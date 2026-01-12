"""
Dataset Manager
Handles all dataset creation, saving, and management operations
"""

import cv2
import numpy as np
import json
import zipfile
import requests
import shutil
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import random
import logging

logger = logging.getLogger(__name__)

try:
    from media_manager_client import MediaManagerClient
    media_manager_available = True
except ImportError:
    media_manager_available = False
    logger.warning("MediaManagerClient not available, will save locally")


class DatasetManager:
    """Manages dataset structure and saving operations"""
    
    def __init__(self, config: Dict, media_manager_client: Optional[object] = None):
        """
        Initialize dataset manager

        Args:
            config: Configuration dictionary
            media_manager_client: Optional MediaManagerClient for remote uploads
        """
        self.config = config
        self.media_manager = media_manager_client
        self.use_media_manager = media_manager_client is not None and config.get('media_manager', {}).get('enabled', True)

        storage_config = config.get('storage', {})

        # Dataset directories (still needed for local backup/metadata)
        self.dataset_dir = Path(config.get('dataset', {}).get('output_dir', 'face_dataset'))
        self.images_dir = Path(storage_config.get('images_dir', 'face_dataset/images'))
        self.labels_dir = Path(storage_config.get('labels_dir', 'face_dataset/labels'))
        self.raw_images_dir = Path(storage_config.get('raw_images_dir', 'face_dataset/raw_images'))
        self.windshield_crops_dir = Path(storage_config.get('windshield_crops_dir', 'face_dataset/windshield_crops'))
        self.visualizations_dir = Path(storage_config.get('visualizations_dir', 'face_dataset/visualizations'))
        self.metadata_dir = Path(storage_config.get('metadata_dir', 'face_dataset/metadata'))
        self.download_extras = storage_config.get('download_extras', False)

        # Train/val split ratio
        self.train_val_split = config.get('dataset', {}).get('train_val_split', 0.8)
        self.save_empty = config.get('dataset', {}).get('save_empty', False)

        # Statistics
        self.stats = {
            "total_images": 0,
            "train_images": 0,
            "val_images": 0,
            "total_faces": 0,
            "total_windshields": 0,
            "started_at": datetime.now().isoformat()
        }

        # Only create directories if not using media manager or if backup is enabled
        self._setup_metadata_directories()
        if not self.use_media_manager or config.get('storage', {}).get('local_backup', False):
            self._setup_directories()

        self._load_stats()

        if self.use_media_manager:
            logger.info(" Dataset Manager configured for remote media manager uploads")
    
    def _setup_directories(self):
        """Create all necessary directories"""
        (self.images_dir / "train").mkdir(parents=True, exist_ok=True)
        (self.images_dir / "val").mkdir(parents=True, exist_ok=True)
        (self.labels_dir / "train").mkdir(parents=True, exist_ok=True)
        (self.labels_dir / "val").mkdir(parents=True, exist_ok=True)
        self.raw_images_dir.mkdir(parents=True, exist_ok=True)
        self.windshield_crops_dir.mkdir(parents=True, exist_ok=True)
        self.visualizations_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Dataset directories created at: {self.dataset_dir}")

    def _setup_metadata_directories(self):
        """Create minimal directories needed for stats/labels even in media-manager mode."""
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        (self.labels_dir / "train").mkdir(parents=True, exist_ok=True)
        (self.labels_dir / "val").mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
    
    def _save_stats(self):
        """Save statistics to JSON file"""
        stats_file = self.dataset_dir / "stats.json"
        self.stats["last_updated"] = datetime.now().isoformat()
        with open(stats_file, 'w') as f:
            json.dump(self.stats, f, indent=2)
    
    def _load_stats(self):
        """Load statistics from JSON file"""
        stats_file = self.dataset_dir / "stats.json"
        if stats_file.exists():
            with open(stats_file, 'r') as f:
                loaded_stats = json.load(f)
                self.stats.update(loaded_stats)
            logger.info(f"Loaded existing stats: {self.stats['total_images']} images")
    
    def save_sample(
        self,
        full_image: np.ndarray,
        windshield_crop: np.ndarray,
        faces: List[Dict],
        windshield_info: Optional[Dict],
        source_name: str = "upload",
        source_url: Optional[str] = None
    ) -> Dict:
        """
        Save a complete sample to the dataset

        Args:
            full_image: Original full image
            windshield_crop: Cropped windshield region
            faces: List of face detections
            windshield_info: Windshield detection info
            source_name: Source identifier
            source_url: Original URL if image came from URL (avoids re-uploading)

        Returns:
            Info about the saved sample
        """
        # Generate unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"face_{timestamp}_{source_name[:20]}"  # Limit filename length

        # Decide split (train/val)
        split = "val" if random.random() > self.train_val_split else "train"

        upload_results = {}

        if not faces and not self.save_empty:
            logger.info(f"Skipping dataset save for {filename}: no faces detected")
            return {
                "filename": filename,
                "split": split,
                "num_faces": 0,
                "windshield_detected": windshield_info is not None,
                "windshield_confidence": windshield_info['confidence'] if windshield_info else None,
                "storage_mode": "media_manager" if self.use_media_manager else "local",
                "skipped": True,
                "skip_reason": "no_faces"
            }

        if self.use_media_manager:
            # Upload to media manager instead of saving locally
            logger.info(f"Processing images for {filename}")

            # If source_url is provided, use it instead of re-uploading raw image
            if source_url:
                logger.info(f"Using source URL instead of re-uploading: {source_url}")
                raw_result_url = source_url
            else:
                # Upload raw image only if no source URL provided
                raw_result = self.media_manager.upload_image(
                    full_image,
                    f"{filename}_raw.jpg",
                    "raw_image"
                )
                raw_result_url = raw_result.get('url') if raw_result else None

            # Upload windshield crop (the main training image)
            crop_result = self.media_manager.upload_image(
                windshield_crop,
                f"{filename}_crop.jpg",
                "windshield_crop"
            )

            # Upload visualization if windshield was detected
            vis_result = None
            if windshield_info:
                vis_image = self._create_visualization(full_image, faces, windshield_info['bbox'])
                vis_result = self.media_manager.upload_image(
                    vis_image,
                    f"{filename}_vis.jpg",
                    "visualization"
                )

            upload_results = {
                'raw_image_url': raw_result_url,
                'windshield_crop_url': crop_result.get('url') if crop_result else None,
                'visualization_url': vis_result.get('url') if vis_result else None,
                'used_source_url': source_url is not None
            }

            # Save YOLO labels as metadata (can be stored in database or saved locally)
            labels = []
            for face in faces:
                yolo_line = self._convert_to_yolo_format(face["bbox"])
                labels.append(yolo_line)

            upload_results['yolo_labels'] = labels
            label_path = self.labels_dir / split / f"{filename}.txt"
            with open(label_path, 'w') as f:
                for line in labels:
                    f.write(line + "\n")

        else:
            # Save locally (original behavior)
            # Save original full image
            raw_path = self.raw_images_dir / f"{filename}.jpg"
            cv2.imwrite(str(raw_path), full_image)

            # Save windshield crop to dataset (training data)
            image_path = self.images_dir / split / f"{filename}.jpg"
            cv2.imwrite(str(image_path), windshield_crop)

            # Save windshield crop separately
            crop_path = self.windshield_crops_dir / f"{filename}.jpg"
            cv2.imwrite(str(crop_path), windshield_crop)

            # Save YOLO labels
            label_path = self.labels_dir / split / f"{filename}.txt"
            with open(label_path, 'w') as f:
                for face in faces:
                    yolo_line = self._convert_to_yolo_format(face["bbox"])
                    f.write(yolo_line + "\n")

            # Save visualization
            vis_path = None
            if windshield_info:
                vis_image = self._create_visualization(full_image, faces, windshield_info['bbox'])
                vis_path = self.visualizations_dir / f"vis_{filename}.jpg"
                cv2.imwrite(str(vis_path), vis_image)

            upload_results = {
                'saved_to': str(image_path),
                'windshield_crop': str(crop_path),
                'visualization': str(vis_path) if vis_path else None
            }

        # Update statistics
        self.stats["total_images"] += 1
        if split == "train":
            self.stats["train_images"] += 1
        else:
            self.stats["val_images"] += 1
        self.stats["total_faces"] += len(faces)
        if windshield_info:
            self.stats["total_windshields"] += 1

        self._save_stats()

        metadata = {
            "filename": filename,
            "split": split,
            "num_faces": len(faces),
            "windshield_detected": windshield_info is not None,
            "windshield_confidence": windshield_info['confidence'] if windshield_info else None,
            "storage_mode": "media_manager" if self.use_media_manager else "local",
            **upload_results
        }
        metadata_path = self.metadata_dir / f"{filename}.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        return {
            **metadata
        }
    
    def _convert_to_yolo_format(self, bbox: Dict) -> str:
        """Convert normalized bbox to YOLO format"""
        left = bbox["left"]
        top = bbox["top"]
        right = bbox["right"]
        bottom = bbox["bottom"]
        
        width = right - left
        height = bottom - top
        x_center = left + width / 2
        y_center = top + height / 2
        
        return f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
    
    def _create_visualization(
        self,
        image: np.ndarray,
        faces: List[Dict],
        windshield_bbox: List[int]
    ) -> np.ndarray:
        """Create visualization with windshield and face boxes"""
        vis_image = image.copy()
        img_height, img_width = image.shape[:2]
        
        # Draw windshield bbox
        x1, y1, x2, y2 = windshield_bbox
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), (255, 0, 0), 3)
        cv2.putText(vis_image, "Windshield", (x1, y1-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # Draw face bboxes (relative to windshield crop, so offset them)
        for face in faces:
            bbox = face["bbox"]
            # Calculate absolute coordinates
            fx1 = int(x1 + bbox["left"] * (x2 - x1))
            fy1 = int(y1 + bbox["top"] * (y2 - y1))
            fx2 = int(x1 + bbox["right"] * (x2 - x1))
            fy2 = int(y1 + bbox["bottom"] * (y2 - y1))
            
            cv2.rectangle(vis_image, (fx1, fy1), (fx2, fy2), (0, 255, 0), 2)
            label = f"Face #{face['face_id']}"
            cv2.putText(vis_image, label, (fx1, fy1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return vis_image
    
    def create_yaml_config(self) -> str:
        """Create YAML configuration file for YOLO training"""
        yaml_content = f"""# Face Detection Dataset (Windshield Crops)
# Auto-generated by Clarifai Auto-Collect

path: {self.dataset_dir.absolute()}
train: images/train
val: images/val

names:
  0: face

nc: 1

# Dataset Info
# Images: Windshield crops from vehicle images
# Labels: Face bounding boxes in YOLO format
# Detection: Faces detected using Clarifai API
"""
        yaml_path = self.dataset_dir / "data.yaml"
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)
        
        logger.info(f"Created YAML config: {yaml_path}")
        return str(yaml_path)

    def _create_yaml_config_at(self, dataset_root: Path) -> str:
        """Create YAML configuration file for YOLO training at a specified root."""
        yaml_content = f"""# Face Detection Dataset (Windshield Crops)
# Auto-generated by Clarifai Auto-Collect

path: {dataset_root.absolute()}
train: images/train
val: images/val

names:
  0: face

nc: 1

# Dataset Info
# Images: Windshield crops from vehicle images
# Labels: Face bounding boxes in YOLO format
# Detection: Faces detected using Clarifai API
"""
        yaml_path = dataset_root / "data.yaml"
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)
        return str(yaml_path)

    def _download_file(self, url: str, dest_path: Path) -> bool:
        """Download a file from a URL to a destination path."""
        try:
            response = requests.get(url, timeout=30, stream=True)
            if response.status_code != 200:
                logger.warning(f"Failed to download {url}: {response.status_code}")
                return False
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            with open(dest_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024 * 128):
                    if chunk:
                        f.write(chunk)
            return True
        except Exception as e:
            logger.warning(f"Download failed for {url}: {e}")
            return False

    def _create_zip_from_dir(self, dataset_root: Path, zip_path: Path) -> str:
        """Create ZIP file from a specific dataset root directory."""
        logger.info("Creating dataset ZIP file...")
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in dataset_root.rglob('*'):
                if file_path.is_file():
                    arcname = file_path.relative_to(dataset_root.parent)
                    zipf.write(file_path, arcname)
        logger.info(f"Created ZIP: {zip_path}")
        return str(zip_path)
    
    def create_zip(self) -> str:
        """Create ZIP file of entire dataset"""
        zip_path = Path("face_dataset.zip")
        local_images_available = self.images_dir.exists() and any(self.images_dir.rglob("*.jpg"))

        if local_images_available:
            # Create YAML first
            self.create_yaml_config()
            return self._create_zip_from_dir(self.dataset_dir, zip_path)

        if not self.use_media_manager:
            raise RuntimeError("Local dataset images not available.")

        metadata_files = list(self.metadata_dir.glob("*.json"))
        if not metadata_files:
            raise RuntimeError(
                "No metadata available to reconstruct dataset. Process at least one image."
            )

        temp_root = Path("face_dataset_tmp_download")
        dataset_root = temp_root / self.dataset_dir.name
        (dataset_root / "images" / "train").mkdir(parents=True, exist_ok=True)
        (dataset_root / "images" / "val").mkdir(parents=True, exist_ok=True)
        (dataset_root / "labels" / "train").mkdir(parents=True, exist_ok=True)
        (dataset_root / "labels" / "val").mkdir(parents=True, exist_ok=True)
        if self.download_extras:
            (dataset_root / "raw_images").mkdir(parents=True, exist_ok=True)
            (dataset_root / "windshield_crops").mkdir(parents=True, exist_ok=True)
            (dataset_root / "visualizations").mkdir(parents=True, exist_ok=True)

        for metadata_path in metadata_files:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

            filename = metadata.get("filename")
            split = metadata.get("split", "train")
            if not filename:
                continue

            crop_url = metadata.get("windshield_crop_url")
            raw_url = metadata.get("raw_image_url")
            vis_url = metadata.get("visualization_url")

            if crop_url:
                self._download_file(
                    crop_url,
                    dataset_root / "images" / split / f"{filename}.jpg"
                )
                if self.download_extras:
                    self._download_file(
                        crop_url,
                        dataset_root / "windshield_crops" / f"{filename}.jpg"
                    )
            if self.download_extras and raw_url:
                self._download_file(raw_url, dataset_root / "raw_images" / f"{filename}.jpg")
            if self.download_extras and vis_url:
                self._download_file(vis_url, dataset_root / "visualizations" / f"vis_{filename}.jpg")

            label_path = self.labels_dir / split / f"{filename}.txt"
            temp_label_path = dataset_root / "labels" / split / f"{filename}.txt"
            if label_path.exists():
                temp_label_path.write_text(label_path.read_text())
            else:
                yolo_labels = metadata.get("yolo_labels", [])
                if yolo_labels:
                    temp_label_path.write_text("\n".join(yolo_labels) + "\n")

        self._create_yaml_config_at(dataset_root)
        try:
            return self._create_zip_from_dir(dataset_root, zip_path)
        finally:
            shutil.rmtree(temp_root, ignore_errors=True)
    
    def clear_dataset(self):
        """Clear all dataset files and reset statistics"""
        if self.dataset_dir.exists():
            shutil.rmtree(self.dataset_dir)
        
        self._setup_directories()
        
        self.stats = {
            "total_images": 0,
            "train_images": 0,
            "val_images": 0,
            "total_faces": 0,
            "total_windshields": 0,
            "started_at": datetime.now().isoformat()
        }
        self._save_stats()
        
        logger.info("Dataset cleared")
    
    def get_stats(self) -> Dict:
        """Get current dataset statistics"""
        return {
            **self.stats,
            "avg_faces_per_image": self.stats["total_faces"] / max(1, self.stats["total_images"]),
            "windshield_detection_rate": self.stats["total_windshields"] / max(1, self.stats["total_images"]),
            "ready_to_download": self.stats["total_images"] > 0
        }
    
    def get_visualization_path(self, filename: str) -> Optional[Path]:
        """Get path to visualization file"""
        vis_path = self.visualizations_dir / f"vis_{filename}.jpg"
        if vis_path.exists():
            return vis_path
        
        # Try without prefix
        vis_path = self.visualizations_dir / f"{filename}.jpg"
        if vis_path.exists():
            return vis_path
        
        return None
