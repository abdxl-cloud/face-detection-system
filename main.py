"""
Clarifai Auto-Collect API - Main Application
Multi-model windshield detection + Clarifai face detection
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, HttpUrl, Field
from typing import List, Optional
import asyncio
import cv2
import numpy as np
import yaml
import logging
from pathlib import Path

from windshield_detector import WindshieldDetector
from clarifai_client import ClarifaiFaceDetector
from dataset_manager import DatasetManager
from media_manager_client import MediaManagerClient
from utils import download_image_from_url, image_to_bytes

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load configuration
with open("config.yaml", 'r') as f:
    config = yaml.safe_load(f)

# Initialize FastAPI app
app = FastAPI(
    title="Face Detection API",
    version="5.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Initialize components
windshield_detector = WindshieldDetector(config)
face_detector = ClarifaiFaceDetector(config)

# Initialize media manager client if enabled
media_manager_client = None
if config.get('media_manager', {}).get('enabled', False):
    try:
        media_manager_client = MediaManagerClient(config)
        logger.info("Media Manager client initialized")
    except Exception as e:
        logger.warning(f"Media Manager initialization failed: {e}. Will save locally.")
        media_manager_client = None

dataset_manager = DatasetManager(config, media_manager_client)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def extract_and_upload_faces(windshield_crop: np.ndarray, faces: List[dict], filename_base: str) -> List[dict]:
    """
    Extract individual face crops and upload them to media manager

    Args:
        windshield_crop: Windshield cropped image
        faces: List of face detections with normalized bboxes
        filename_base: Base filename for uploads

    Returns:
        List of face data with URLs
    """
    img_height, img_width = windshield_crop.shape[:2]
    face_results = []

    for face in faces:
        bbox = face["bbox"]
        face_id = face["face_id"]

        # Calculate pixel coordinates
        x = int(bbox["left"] * img_width)
        y = int(bbox["top"] * img_height)
        w = int((bbox["right"] - bbox["left"]) * img_width)
        h = int((bbox["bottom"] - bbox["top"]) * img_height)

        # Ensure coordinates are within image bounds
        x = max(0, min(x, img_width))
        y = max(0, min(y, img_height))
        x2 = min(x + w, img_width)
        y2 = min(y + h, img_height)

        # Crop individual face
        face_crop = windshield_crop[y:y2, x:x2]

        # Upload face crop to media manager
        face_url = None
        if media_manager_client and face_crop.size > 0:
            result = media_manager_client.upload_image(
                face_crop,
                f"{filename_base}_face_{face_id}.jpg",
                "face_crop"
            )
            face_url = result.get('url') if result else None

        face_results.append({
            "face_id": face_id,
            "image_url": face_url,
            "bbox": {
                "x": x,
                "y": y,
                "width": w,
                "height": h
            }
        })

    return face_results


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class ImageURL(BaseModel):
    url: HttpUrl

    class Config:
        json_schema_extra = {
            "example": {
                "url": "https://example.com/car-image.jpg"
            }
        }


class URLBatch(BaseModel):
    urls: List[str]

    class Config:
        json_schema_extra = {
            "example": {
                "urls": ["https://example.com/car1.jpg", "https://example.com/car2.jpg"]
            }
        }


class BBoxResponse(BaseModel):
    x: int
    y: int
    width: int
    height: int

    class Config:
        json_schema_extra = {
            "example": {
                "x": 145,
                "y": 89,
                "width": 180,
                "height": 240
            }
        }


class FaceResponse(BaseModel):
    face_id: int
    image_url: str
    bbox: BBoxResponse

    class Config:
        json_schema_extra = {
            "example": {
                "face_id": 0,
                "image_url": "https://etraffica-media-manager.ngrok.app/api/uploads/face_0.jpg",
                "bbox": {
                    "x": 145,
                    "y": 89,
                    "width": 180,
                    "height": 240
                }
            }
        }


class ProcessResponse(BaseModel):
    status: str
    request_id: str
    faces: List[FaceResponse]
    original_image_url: str
    total_faces: int

    class Config:
        json_schema_extra = {
            "example": {
                "status": "success",
                "request_id": "face_20260109_120530_car",
                "faces": [
                    {
                        "face_id": 0,
                        "image_url": "https://etraffica-media-manager.ngrok.app/api/uploads/face_0.jpg",
                        "bbox": {
                            "x": 145,
                            "y": 89,
                            "width": 180,
                            "height": 240
                        }
                    }
                ],
                "original_image_url": "https://example.com/car-image.jpg",
                "total_faces": 1
            }
        }


class BatchResultItem(BaseModel):
    url: str
    status: str
    request_id: Optional[str] = None
    faces: List[FaceResponse] = Field(default_factory=list)
    total_faces: int = 0
    error: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "url": "https://example.com/car1.jpg",
                "status": "success",
                "request_id": "face_20260109_120530_car1",
                "faces": [
                    {
                        "face_id": 0,
                        "image_url": "https://etraffica-media-manager.ngrok.app/api/uploads/face_20260109_120530_car1_face_0.jpg",
                        "bbox": {
                            "x": 145,
                            "y": 89,
                            "width": 180,
                            "height": 240
                        }
                    }
                ],
                "total_faces": 1
            }
        }


class BatchProcessResponse(BaseModel):
    status: str
    total_processed: int
    successful: int
    failed: int
    results: List[BatchResultItem]

    class Config:
        json_schema_extra = {
            "example": {
                "status": "completed",
                "total_processed": 2,
                "successful": 2,
                "failed": 0,
                "results": [
                    {
                        "url": "https://example.com/car1.jpg",
                        "status": "success",
                        "request_id": "face_20260109_120530_car1",
                        "faces": [
                            {
                                "face_id": 0,
                                "image_url": "https://etraffica-media-manager.ngrok.app/api/uploads/face_20260109_120530_car1_face_0.jpg",
                                "bbox": {"x": 145, "y": 89, "width": 180, "height": 240}
                            }
                        ],
                        "total_faces": 1
                    },
                    {
                        "url": "https://example.com/car2.jpg",
                        "status": "success",
                        "request_id": "face_20260109_120531_car2",
                        "faces": [
                            {
                                "face_id": 0,
                                "image_url": "https://etraffica-media-manager.ngrok.app/api/uploads/face_20260109_120531_car2_face_0.jpg",
                                "bbox": {"x": 150, "y": 95, "width": 175, "height": 235}
                            }
                        ],
                        "total_faces": 1
                    }
                ]
            }
        }


# ============================================================================
# WEB INTERFACE
# ============================================================================

@app.get("/", response_class=HTMLResponse)
async def root():
    """Web interface"""
    windshield_status = "Active" if windshield_detector.is_available() else "Disabled"
    num_models = len(windshield_detector.models) if windshield_detector.is_available() else 0
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Clarifai Auto-Collect</title>
        <style>
            body {{ font-family: Arial; max-width: 1000px; margin: 0 auto; padding: 20px;
                   background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }}
            .container {{ background: white; padding: 30px; border-radius: 10px; 
                        box-shadow: 0 10px 40px rgba(0,0,0,0.2); margin-bottom: 20px; }}
            h1 {{ color: #667eea; text-align: center; }}
            .status {{ display: inline-block; padding: 5px 15px; border-radius: 15px;
                      font-weight: bold; margin: 5px; }}
            .status-ok {{ background: #c6f6d5; color: #22543d; }}
            pre {{ background: #2d3748; color: #68d391; padding: 15px; border-radius: 5px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Clarifai Auto-Collect + Multi-YOLO</h1>
            <p style="text-align: center;">Multi-Model Windshield Detection + Face Recognition</p>
            <div style="text-align: center;">
                <span class="status status-ok">Windshield: {windshield_status} ({num_models} models)</span>
                <span class="status status-ok">Faces: Clarifai</span>
            </div>
        </div>
        
        <div class="container">
            <h2>Quick Start</h2>
            <h3>Upload Image</h3>
            <pre>curl -X POST "http://localhost:8000/upload" -F "file=@car.jpg"</pre>
            
            <h3>Add from URL</h3>
            <pre>curl -X POST "http://localhost:8000/add-url" \\
  -H "Content-Type: application/json" \\
  -d '{{"url": "https://example.com/car.jpg"}}'</pre>
            
            <h3>Check Stats</h3>
            <pre>curl "http://localhost:8000/stats"</pre>
            
            <h3>Download Dataset</h3>
            <pre>curl "http://localhost:8000/download" -o dataset.zip</pre>
        </div>
        
        <div class="container">
            <h2>API Documentation</h2>
            <a href="/docs"><button style="background: #667eea; color: white; padding: 10px 20px; 
                   border: none; border-radius: 5px; cursor: pointer;">Open Swagger UI</button></a>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html)


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.post(
    "/upload",
    response_model=ProcessResponse,
    tags=["Face Detection"],
    summary="Upload Image File",
    responses={
        200: {
            "description": "Successfully processed uploaded image and detected faces",
            "content": {
                "application/json": {
                    "example": {
                        "status": "success",
                        "request_id": "face_20260109_120530_uploaded",
                        "faces": [
                            {
                                "face_id": 0,
                                "image_url": "https://etraffica-media-manager.ngrok.app/api/uploads/face_20260109_120530_uploaded_face_0.jpg",
                                "bbox": {"x": 145, "y": 89, "width": 180, "height": 240}
                            }
                        ],
                        "original_image_url": "https://etraffica-media-manager.ngrok.app/api/uploads/face_20260109_120530_uploaded_raw.jpg",
                        "total_faces": 1
                    }
                }
            }
        },
        400: {"description": "Invalid image file"},
        500: {"description": "Internal server error during processing"}
    }
)
async def upload_image(file: UploadFile = File(..., description="Image file (JPEG, PNG, etc.)")):
    """
    ## Upload and Process Image File

    Upload an image file, detect windshield region, extract faces, and upload individual face crops.

    ### Process Flow:
    1. Read uploaded image file
    2. Detect windshield region (if available)
    3. Detect faces within windshield area
    4. Crop each face individually
    5. Upload all images (raw + face crops) to media manager
    6. Return URLs to cropped face images

    ### Parameters:
    - **file**: Image file to process (multipart/form-data)

    ### Returns:
    - **status**: Request status (success/error)
    - **request_id**: Unique identifier for this request
    - **faces**: Array of detected faces with URLs and bounding boxes
    - **original_image_url**: URL to uploaded raw image
    - **total_faces**: Number of faces detected

    ### Example Response:
    Each face in the response includes:
    - Direct URL to cropped face image
    - Bounding box coordinates (x, y, width, height) in pixels
    - Face ID for reference
    """
    try:
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        full_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if full_image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        # Detect windshield
        windshield_info = None
        windshield_crop = full_image

        if windshield_detector.is_available():
            windshield_info = windshield_detector.detect(full_image)
            if windshield_info:
                windshield_crop = windshield_detector.crop_windshield(
                    full_image,
                    windshield_info['bbox']
                )

        # Detect faces
        crop_bytes = image_to_bytes(windshield_crop)
        faces, _ = face_detector.detect_faces(crop_bytes)

        # Generate request ID
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = file.filename or "uploaded"
        request_id = f"face_{timestamp}_{filename[:20]}"

        # Extract and upload individual face crops
        face_results = extract_and_upload_faces(windshield_crop, faces, request_id)

        # Save to dataset
        save_result = dataset_manager.save_sample(
            full_image,
            windshield_crop,
            faces,
            windshield_info,
            filename
        )

        # Get the uploaded raw image URL
        original_url = save_result.get('raw_image_url') if save_result else None

        # Return simplified response with face crops
        return {
            "status": "success",
            "request_id": request_id,
            "faces": face_results,
            "original_image_url": original_url,
            "total_faces": len(face_results)
        }

    except Exception as e:
        logger.error(f"Error processing upload: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/add-url",
    response_model=ProcessResponse,
    tags=["Face Detection"],
    summary="Process Image from URL",
    responses={
        200: {
            "description": "Successfully processed image and detected faces",
            "content": {
                "application/json": {
                    "example": {
                        "status": "success",
                        "request_id": "face_20260109_120530_car",
                        "faces": [
                            {
                                "face_id": 0,
                                "image_url": "https://etraffica-media-manager.ngrok.app/api/uploads/face_20260109_120530_car_face_0.jpg",
                                "bbox": {"x": 145, "y": 89, "width": 180, "height": 240}
                            },
                            {
                                "face_id": 1,
                                "image_url": "https://etraffica-media-manager.ngrok.app/api/uploads/face_20260109_120530_car_face_1.jpg",
                                "bbox": {"x": 420, "y": 95, "width": 175, "height": 235}
                            }
                        ],
                        "original_image_url": "https://example.com/car-image.jpg",
                        "total_faces": 2
                    }
                }
            }
        },
        400: {"description": "Invalid URL or image format"},
        500: {"description": "Error downloading or processing image"}
    }
)
async def add_url(data: ImageURL):
    """
    ## Process Image from URL

    Download image from URL, detect windshield, extract faces, and upload individual face crops.

    ### Process Flow:
    1. Download image from provided URL
    2. Detect windshield region (if available)
    3. Detect faces within windshield area
    4. Crop each face individually
    5. Upload face crops to media manager
    6. Return URLs to cropped face images

    ### Efficiency:
    - **Original image is NOT re-uploaded** - the source URL is stored directly
    - Only face crops are uploaded to media manager
    - Saves bandwidth and storage space

    ### Parameters:
    - **url**: Direct URL to image file (must be publicly accessible)

    ### Returns:
    - **status**: Request status (success/error)
    - **request_id**: Unique identifier for this request
    - **faces**: Array of detected faces with URLs and bounding boxes
    - **original_image_url**: Original source URL (not re-uploaded)
    - **total_faces**: Number of faces detected

    ### Example Usage:
    ```bash
    curl -X POST "http://localhost:8000/add-url" \\
      -H "Content-Type: application/json" \\
      -d '{"url": "https://example.com/vehicle-photo.jpg"}'
    ```

    ### Notes:
    - Image URL must be publicly accessible
    - Supported formats: JPEG, PNG, BMP, etc.
    - Maximum recommended image size: 10MB
    """
    try:
        # Download image
        full_image = download_image_from_url(str(data.url))

        # Detect windshield
        windshield_info = None
        windshield_crop = full_image

        if windshield_detector.is_available():
            windshield_info = windshield_detector.detect(full_image)
            if windshield_info:
                windshield_crop = windshield_detector.crop_windshield(
                    full_image,
                    windshield_info['bbox']
                )

        # Detect faces
        crop_bytes = image_to_bytes(windshield_crop)
        faces, _ = face_detector.detect_faces(crop_bytes)

        # Generate request ID
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        url_filename = str(data.url).split('/')[-1] or "url_image"
        request_id = f"face_{timestamp}_{url_filename[:20]}"

        # Extract and upload individual face crops
        face_results = extract_and_upload_faces(windshield_crop, faces, request_id)

        # Save to dataset (metadata only) - pass source_url to avoid re-uploading
        dataset_manager.save_sample(
            full_image,
            windshield_crop,
            faces,
            windshield_info,
            url_filename,
            source_url=str(data.url)
        )

        # Return simplified response with face crops
        return {
            "status": "success",
            "request_id": request_id,
            "faces": face_results,
            "original_image_url": str(data.url),
            "total_faces": len(face_results)
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing URL: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/batch-urls",
    response_model=BatchProcessResponse,
    tags=["Face Detection"],
    summary="Batch Process Multiple URLs",
    responses={
        200: {
            "description": "Batch processed multiple URLs",
            "content": {
                "application/json": {
                    "example": {
                        "status": "completed",
                        "total_processed": 2,
                        "successful": 2,
                        "failed": 0,
                        "results": [
                            {
                                "url": "https://example.com/car1.jpg",
                                "status": "success",
                                "request_id": "face_20260109_120530_car1",
                                "faces": [
                                    {
                                        "face_id": 0,
                                        "image_url": "https://etraffica-media-manager.ngrok.app/api/uploads/face_20260109_120530_car1_face_0.jpg",
                                        "bbox": {"x": 145, "y": 89, "width": 180, "height": 240}
                                    }
                                ],
                                "total_faces": 1
                            },
                            {
                                "url": "https://example.com/car2.jpg",
                                "status": "success",
                                "request_id": "face_20260109_120531_car2",
                                "faces": [
                                    {
                                        "face_id": 0,
                                        "image_url": "https://etraffica-media-manager.ngrok.app/api/uploads/face_20260109_120531_car2_face_0.jpg",
                                        "bbox": {"x": 150, "y": 95, "width": 175, "height": 235}
                                    }
                                ],
                                "total_faces": 1
                            }
                        ]
                    }
                }
            }
        },
        400: {"description": "Invalid request or URLs"},
        500: {"description": "Error processing batch"}
    }
)
async def batch_urls(data: URLBatch):
    """
    ## Batch Process Multiple Image URLs

    Process multiple images in a single request. Ideal for bulk operations.

    ### Process Flow:
    1. Receive array of image URLs
    2. Process each URL sequentially
    3. For each image:
       - Download from URL
       - Detect windshield region
       - Detect and extract faces
       - Upload face crops to media manager
    4. Return aggregated results

    ### Features:
    - **Sequential Processing**: Images processed one by one with delay
    - **Error Handling**: Failed images don't stop the batch
    - **Individual Results**: Each URL gets its own result object
    - **Efficiency**: Original URLs not re-uploaded

    ### Parameters:
    - **urls**: Array of image URLs to process

    ### Returns:
    - **status**: Overall batch status (completed)
    - **total_processed**: Number of URLs in batch
    - **successful**: Number successfully processed
    - **failed**: Number that failed
    - **results**: Array of individual results for each URL

    ### Example Usage:
    ```bash
    curl -X POST "http://localhost:8000/batch-urls" \\
      -H "Content-Type: application/json" \\
      -d '{
        "urls": [
          "https://example.com/car1.jpg",
          "https://example.com/car2.jpg",
          "https://example.com/car3.jpg"
        ]
      }'
    ```

    ### Configuration:
    - Default delay between images: 0.5 seconds (configurable)
    - Each URL processed independently
    - Failed URLs return error details in results array

    ### Notes:
    - All URLs must be publicly accessible
    - Large batches may take time to process
    - Consider using webhooks for async processing of large batches
    """
    results = []
    batch_delay = config.get('processing', {}).get('batch_delay', 0.5)

    for url in data.urls:
        try:
            full_image = download_image_from_url(url)

            windshield_info = None
            windshield_crop = full_image

            if windshield_detector.is_available():
                windshield_info = windshield_detector.detect(full_image)
                if windshield_info:
                    windshield_crop = windshield_detector.crop_windshield(
                        full_image,
                        windshield_info['bbox']
                    )

            crop_bytes = image_to_bytes(windshield_crop)
            faces, _ = face_detector.detect_faces(crop_bytes)

            # Generate request ID
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            url_filename = url.split('/')[-1] or "url_image"
            request_id = f"face_{timestamp}_{url_filename[:20]}"

            # Extract and upload individual face crops
            face_results = extract_and_upload_faces(windshield_crop, faces, request_id)

            # Save to dataset
            dataset_manager.save_sample(
                full_image,
                windshield_crop,
                faces,
                windshield_info,
                url_filename,
                source_url=url
            )

            results.append({
                "url": url,
                "status": "success",
                "request_id": request_id,
                "faces": face_results,
                "total_faces": len(face_results)
            })

            await asyncio.sleep(batch_delay)

        except HTTPException as e:
            results.append({
                "url": url,
                "status": "failed",
                "error": str(e.detail)
            })
        except Exception as e:
            results.append({
                "url": url,
                "status": "failed",
                "error": str(e)
            })

    return {
        "status": "completed",
        "total_processed": len(data.urls),
        "successful": len([r for r in results if r["status"] == "success"]),
        "failed": len([r for r in results if r["status"] == "failed"]),
        "results": results
    }


@app.get(
    "/stats",
    tags=["System"],
    summary="Get Dataset Statistics"
)
async def get_stats():
    """
    ## Get Dataset Statistics

    Retrieve statistics about processed images and detected faces.

    ### Returns:
    - **dataset**: Dataset name
    - **windshield_detection**: Status and configuration
    - **total_images**: Total number of processed images
    - **total_faces**: Total number of detected faces
    - **train_images**: Training set size
    - **val_images**: Validation set size
    - **avg_faces_per_image**: Average faces per image
    - **windshield_detection_rate**: Percentage of images with detected windshields

    ### Example Response:
    ```json
    {
      "dataset": "face_dataset",
      "windshield_detection": {
        "enabled": true,
        "num_models": 1,
        "ensemble_method": "union"
      },
      "total_images": 150,
      "total_faces": 245,
      "train_images": 120,
      "val_images": 30,
      "avg_faces_per_image": 1.63,
      "windshield_detection_rate": 0.95
    }
    ```
    """
    stats = dataset_manager.get_stats()

    return {
        "dataset": "face_dataset",
        "windshield_detection": {
            "enabled": windshield_detector.is_available(),
            "num_models": len(windshield_detector.models) if windshield_detector.is_available() else 0,
            "ensemble_method": config.get('ensemble', {}).get('method', 'union')
        },
        **stats
    }


@app.get(
    "/models",
    tags=["System"],
    summary="Get Model Information"
)
async def get_models():
    """
    ## Get Loaded Model Information

    Retrieve information about loaded YOLO windshield detection models.

    ### Returns:
    - **windshield_models**: Array of loaded YOLO models with configuration
    - **ensemble_config**: Ensemble method configuration
    - **status**: Overall model status (active/disabled)

    ### Model Information Includes:
    - Model name and type (YOLOv8, YOLOv11, etc.)
    - Confidence and IOU thresholds
    - Device (CPU/CUDA)
    - Model weight for ensemble voting

    ### Example Response:
    ```json
    {
      "windshield_models": [
        {
          "name": "model_1",
          "model_type": "yolov12",
          "confidence_threshold": 0.35,
          "iou_threshold": 0.45,
          "device": "cuda",
          "weight": 1.0
        }
      ],
      "ensemble_config": {
        "method": "union",
        "nms_iou_threshold": 0.5
      },
      "status": "active"
    }
    ```
    """
    if not windshield_detector.is_available():
        return {"windshield_models": [], "status": "disabled"}

    return {
        "windshield_models": windshield_detector.get_model_info(),
        "ensemble_config": windshield_detector.ensemble_config,
        "status": "active"
    }


@app.get(
    "/download",
    tags=["Dataset Management"],
    summary="Download Dataset ZIP"
)
async def download_dataset():
    """
    ## Download Complete Dataset

    Download the entire dataset as a ZIP file including images, labels, and configuration.

    ### ZIP Contents:
    - **images/**: Training and validation images (windshield crops)
    - **labels/**: YOLO format labels for each image
    - **raw_images/**: Original full images
    - **windshield_crops/**: Windshield region crops
    - **visualizations/**: Images with bounding boxes drawn
    - **data.yaml**: YOLO training configuration file
    - **stats.json**: Dataset statistics

    ### Returns:
    - ZIP file download (application/zip)

    ### Notes:
    - Only available when using local storage mode
    - Dataset must contain at least one image
    - Use for training YOLO face detection models
    """
    stats = dataset_manager.get_stats()

    if stats["total_images"] == 0:
        raise HTTPException(
            status_code=400,
            detail="No images in dataset. Upload some images first!"
        )

    try:
        zip_path = dataset_manager.create_zip()
        return FileResponse(
            zip_path,
            media_type="application/zip",
            filename="face_dataset.zip"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/visualize/{filename}",
    tags=["Dataset Management"],
    summary="Get Visualization Image"
)
async def get_visualization(filename: str):
    """
    ## Get Visualization Image

    Retrieve visualization image showing detected windshield and faces with bounding boxes.

    ### Parameters:
    - **filename**: Request ID or filename (without extension)

    ### Returns:
    - JPEG image with drawn bounding boxes

    ### Notes:
    - Visualizations show both windshield and face detections
    - Only available for locally stored images
    """
    vis_path = dataset_manager.get_visualization_path(filename)

    if vis_path is None:
        raise HTTPException(status_code=404, detail="Visualization not found")

    return FileResponse(vis_path, media_type="image/jpeg")


@app.delete(
    "/clear",
    tags=["Dataset Management"],
    summary="Clear Dataset"
)
async def clear_dataset():
    """
    ## Clear All Dataset Data

    Delete all images, labels, and statistics from the local dataset.

    ### Warning:
    - This action is **irreversible**
    - All local files will be permanently deleted
    - Does NOT affect images uploaded to media manager

    ### Returns:
    - **status**: Success/failure
    - **message**: Confirmation message

    ### Notes:
    - Only affects local storage
    - Use with caution in production environments
    """
    try:
        dataset_manager.clear_dataset()
        return {
            "status": "success",
            "message": "Dataset cleared successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/health",
    tags=["System"],
    summary="Health Check"
)
async def health_check():
    """
    ## System Health Check

    Check the operational status of all system components.

    ### Returns:
    - **status**: Overall system status (healthy/unhealthy)
    - **clarifai_available**: Clarifai SDK status
    - **windshield_detection_available**: YOLO models status
    - **num_windshield_models**: Number of loaded models
    - **dataset_stats**: Current dataset statistics
    - **media_manager_enabled**: Media manager status

    ### Example Response:
    ```json
    {
      "status": "healthy",
      "clarifai_available": true,
      "windshield_detection_available": true,
      "num_windshield_models": 1,
      "media_manager_enabled": true,
      "dataset_stats": {
        "total_images": 150,
        "total_faces": 245
      }
    }
    ```

    ### Use Cases:
    - Monitoring and alerting
    - Load balancer health checks
    - Service discovery
    - Integration testing
    """
    return {
        "status": "healthy",
        "clarifai_available": ClarifaiFaceDetector.is_available(),
        "windshield_detection_available": windshield_detector.is_available(),
        "num_windshield_models": len(windshield_detector.models) if windshield_detector.is_available() else 0,
        "media_manager_enabled": media_manager_client is not None,
        "dataset_stats": dataset_manager.get_stats()
    }

if __name__ == "__main__":
    import uvicorn

    print("\n" + "=" * 70)
    print("Clarifai Auto-Collect with Multi-Model Windshield Detection")
    print("=" * 70)

    if windshield_detector.is_available():
        print(f"\nWindshield Detection: {len(windshield_detector.models)} model(s) loaded")
        for model_info in windshield_detector.get_model_info():
            print(f"   - {model_info['name']} ({model_info['model_type']})")
        print(f"   Ensemble method: {config.get('ensemble', {}).get('method', 'union')}")
    else:
        print("\nWindshield detection disabled (no models loaded)")

    print("\nFace Detection: Clarifai API")

    # Show storage mode
    if media_manager_client:
        print(f"\nStorage: Media Manager API ({config.get('media_manager', {}).get('base_url')})")
    else:
        print("\nStorage: Local file system")

    stats = dataset_manager.get_stats()
    print(f"\nCurrent dataset: {stats['total_images']} images, {stats['total_faces']} faces")

    server_config = config.get('server', {})
    host = server_config.get('host', '0.0.0.0')
    port = server_config.get('port', 8000)

    print(f"\nStarting server at http://{host}:{port}")
    print(f"Documentation: http://{host}:{port}/docs")
    print("=" * 70 + "\n")

    uvicorn.run(app, host=host, port=port)
