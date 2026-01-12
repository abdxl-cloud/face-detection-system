"""
Windshield Detection Module
Supports multiple YOLO models with ensemble methods
"""

import cv2
import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

try:
    from ultralytics import YOLO
    yolo_available = True
except ImportError:
    yolo_available = False


class WindshieldDetector:
    """Multi-model YOLO windshield detector with ensemble support"""
    
    def __init__(self, config: Dict):
        """
        Initialize windshield detector with multiple models
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.models = []
        self.ensemble_config = config.get('ensemble', {})
        
        if not yolo_available:
            logger.warning("Ultralytics YOLO not available. Windshield detection disabled.")
            return
        
        self._load_models()
    
    def _load_models(self):
        """Load all enabled YOLO models"""
        model_configs = self.config.get('windshield_models', [])
        
        if not model_configs:
            logger.warning("No windshield models configured")
            return
        
        loaded_count = 0
        for model_cfg in model_configs:
            if not model_cfg.get('enabled', True):
                logger.info(f"Model '{model_cfg.get('name')}' is disabled, skipping")
                continue
            
            try:
                model_name = model_cfg.get('name', 'unknown')
                model_path = model_cfg['model_path']
                model_type = model_cfg.get('model_type', 'yolov8')
                
                logger.info(f"Loading {model_type} model: {model_name} from {model_path}")
                
                # Load YOLO model
                model = YOLO(model_path)
                
                # Store model with configuration
                model_info = {
                    'name': model_name,
                    'model': model,
                    'model_type': model_type,
                    'conf_threshold': model_cfg.get('conf_threshold', 0.35),
                    'iou_threshold': model_cfg.get('iou_threshold', 0.45),
                    'device': model_cfg.get('device', 'cuda'),
                    'weight': model_cfg.get('weight', 1.0),
                    'index': loaded_count
                }
                
                self.models.append(model_info)
                loaded_count += 1
                
                logger.info(f" Loaded {model_type} model '{model_name}' (weight: {model_info['weight']})")
                
            except Exception as e:
                logger.error(f"Failed to load model {model_cfg.get('name')}: {e}")
                continue
        
        if loaded_count == 0:
            logger.warning("No windshield models were loaded successfully")
        else:
            logger.info(f" Loaded {loaded_count} windshield detection model(s)")
            logger.info(f"Ensemble method: {self.ensemble_config.get('method', 'union')}")
    
    def is_available(self) -> bool:
        """Check if windshield detection is available"""
        return len(self.models) > 0
    
    def detect(self, image: np.ndarray) -> Optional[Dict]:
        """
        Detect windshield in image using all enabled models
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Dict with bbox and confidence, or None if no detection
        """
        if not self.is_available():
            return None
        
        all_detections = []
        
        # Get detections from all models
        for model_info in self.models:
            try:
                results = model_info['model'].predict(
                    image,
                    conf=model_info['conf_threshold'],
                    iou=model_info['iou_threshold'],
                    verbose=False
                )[0]
                
                # Extract detections
                for box in results.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    
                    all_detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': conf,
                        'model': model_info['index'],
                        'model_name': model_info['name'],
                        'weight': model_info['weight']
                    })
                
                logger.info(f"Model '{model_info['name']}' detected {len(results.boxes)} windshield(s)")
                
            except Exception as e:
                logger.error(f"Error in model '{model_info['name']}': {e}")
                continue
        
        if not all_detections:
            logger.info("No windshields detected by any model")
            return None
        
        # Apply ensemble method
        ensemble_method = self.ensemble_config.get('method', 'union')
        
        if ensemble_method == 'best_confidence':
            final_detection = self._ensemble_best_confidence(all_detections)
        elif ensemble_method == 'weighted':
            final_detection = self._ensemble_weighted(all_detections)
        elif ensemble_method == 'nms':
            final_detection = self._ensemble_nms(all_detections)
        else:  # union (default)
            final_detection = self._ensemble_union(all_detections)
        
        return final_detection
    
    def _ensemble_best_confidence(self, detections: List[Dict]) -> Optional[Dict]:
        """Select detection with highest confidence"""
        if not detections:
            return None
        
        best = max(detections, key=lambda d: d['confidence'])
        
        return {
            'bbox': best['bbox'],
            'confidence': best['confidence'],
            'method': 'best_confidence',
            'num_detections': len(detections)
        }
    
    def _ensemble_weighted(self, detections: List[Dict]) -> Optional[Dict]:
        """Weighted average of all detections"""
        if not detections:
            return None
        
        total_weight = sum(d['weight'] * d['confidence'] for d in detections)
        
        # Weighted average of bboxes
        weighted_bbox = [0, 0, 0, 0]
        for det in detections:
            weight = det['weight'] * det['confidence']
            for i in range(4):
                weighted_bbox[i] += det['bbox'][i] * weight
        
        weighted_bbox = [int(coord / total_weight) for coord in weighted_bbox]
        avg_confidence = total_weight / sum(d['weight'] for d in detections)
        
        return {
            'bbox': weighted_bbox,
            'confidence': avg_confidence,
            'method': 'weighted',
            'num_detections': len(detections)
        }
    
    def _ensemble_nms(self, detections: List[Dict]) -> Optional[Dict]:
        """Non-Maximum Suppression"""
        if not detections:
            return None
        
        # Convert to numpy arrays for NMS
        boxes = np.array([d['bbox'] for d in detections])
        scores = np.array([d['confidence'] for d in detections])
        
        # Simple NMS implementation
        nms_threshold = self.ensemble_config.get('nms_iou_threshold', 0.5)
        keep = self._nms(boxes, scores, nms_threshold)
        
        if len(keep) == 0:
            return None
        
        # Return best after NMS
        best_idx = keep[0]
        best = detections[best_idx]
        
        return {
            'bbox': best['bbox'],
            'confidence': best['confidence'],
            'method': 'nms',
            'num_detections': len(keep)
        }
    
    def _ensemble_union(self, detections: List[Dict]) -> Optional[Dict]:
        """Union of all detections (largest bbox containing all)"""
        if not detections:
            return None
        
        # Find bounding box that contains all detections
        x1_min = min(d['bbox'][0] for d in detections)
        y1_min = min(d['bbox'][1] for d in detections)
        x2_max = max(d['bbox'][2] for d in detections)
        y2_max = max(d['bbox'][3] for d in detections)
        
        avg_confidence = sum(d['confidence'] for d in detections) / len(detections)
        
        return {
            'bbox': [x1_min, y1_min, x2_max, y2_max],
            'confidence': avg_confidence,
            'method': 'union',
            'num_detections': len(detections)
        }
    
    def _nms(self, boxes: np.ndarray, scores: np.ndarray, threshold: float) -> List[int]:
        """Non-Maximum Suppression implementation"""
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            
            inds = np.where(iou <= threshold)[0]
            order = order[inds + 1]
        
        return keep
    
    def crop_windshield(self, image: np.ndarray, bbox: List[int]) -> np.ndarray:
        """
        Crop windshield region from image
        
        Args:
            image: Full image
            bbox: [x1, y1, x2, y2]
        
        Returns:
            Cropped windshield image
        """
        x1, y1, x2, y2 = bbox
        
        # Ensure coordinates are within image bounds
        h, w = image.shape[:2]
        x1 = max(0, min(x1, w))
        y1 = max(0, min(y1, h))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))
        
        # Crop
        windshield_crop = image[y1:y2, x1:x2]
        
        return windshield_crop
    
    def get_model_info(self) -> List[Dict]:
        """Get information about loaded models"""
        return [
            {
                'name': m['name'],
                'model_type': m['model_type'],
                'confidence_threshold': m['conf_threshold'],
                'iou_threshold': m['iou_threshold'],
                'device': m['device'],
                'weight': m['weight']
            }
            for m in self.models
        ]
        