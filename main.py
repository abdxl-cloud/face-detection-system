"""
Clarifai Auto-Collect API - Main Application
Multi-model windshield detection + Clarifai face detection
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, HttpUrl
from typing import List
import cv2
import numpy as np
import yaml
import logging
from pathlib import Path
import time

from windshield_detector import WindshieldDetector
from clarifai_client import ClarifaiFaceDetector
from dataset_manager import DatasetManager
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
    title="Clarifai Auto-Collect with Multi-Model Windshield Detection",
    description="Upload → Detect Windshields (Multi-YOLO) → Crop → Detect Faces (Clarifai) → Save",
    version="5.0.0"
)

# Initialize components
windshield_detector = WindshieldDetector(config)
face_detector = ClarifaiFaceDetector(config)
dataset_manager = DatasetManager(config)


# ============================================================================
# REQUEST MODELS
# ============================================================================

class ImageURL(BaseModel):
    url: HttpUrl


class URLBatch(BaseModel):
    urls: List[str]


# ============================================================================
# WEB INTERFACE
# ============================================================================

@app.get("/", response_class=HTMLResponse)
async def root():
    """Web interface"""
    windshield_status = "✅ Active" if windshield_detector.is_available() else "⚠️ Disabled"
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
            <h1>🎯 Clarifai Auto-Collect + Multi-YOLO</h1>
            <p style="text-align: center;">Multi-Model Windshield Detection + Face Recognition</p>
            <div style="text-align: center;">
                <span class="status status-ok">Windshield: {windshield_status} ({num_models} models)</span>
                <span class="status status-ok">Faces: ✅ Clarifai</span>
            </div>
        </div>
        
        <div class="container">
            <h2>🚀 Quick Start</h2>
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
            <h2>📖 API Documentation</h2>
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

@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    """
    Upload image → Detect windshield → Crop → Detect faces → Save
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
        windshield_crop = full_image  # Default to full image
        
        if windshield_detector.is_available():
            windshield_info = windshield_detector.detect(full_image)
            
            if windshield_info:
                windshield_crop = windshield_detector.crop_windshield(
                    full_image, 
                    windshield_info['bbox']
                )
                logger.info(f"Windshield detected: {windshield_info}")
            else:
                logger.info("No windshield detected, using full image")
        
        # Detect faces on windshield crop
        crop_bytes = image_to_bytes(windshield_crop)
        faces, detection_time = face_detector.detect_faces(crop_bytes)
        
        # Add pixel coordinates for faces
        img_height, img_width = windshield_crop.shape[:2]
        for face in faces:
            bbox = face["bbox"]
            face["bbox_pixels"] = {
                "x": int(bbox["left"] * img_width),
                "y": int(bbox["top"] * img_height),
                "w": int((bbox["right"] - bbox["left"]) * img_width),
                "h": int((bbox["bottom"] - bbox["top"]) * img_height)
            }
        
        # Save to dataset
        save_info = dataset_manager.save_sample(
            full_image,
            windshield_crop,
            faces,
            windshield_info,
            file.filename
        )
        
        return {
            "status": "success",
            "message": "Image processed and saved successfully!",
            "windshield_detection": {
                "detected": windshield_info is not None,
                "confidence": windshield_info['confidence'] if windshield_info else None,
                "method": windshield_info.get('method') if windshield_info else None,
                "num_models": windshield_info.get('num_detections') if windshield_info else 0
            },
            "face_detection": {
                "num_faces": len(faces),
                "faces": faces,
                "detection_time": detection_time
            },
            "saved": save_info,
            "dataset_stats": dataset_manager.get_stats()
        }
        
    except Exception as e:
        logger.error(f"Error processing upload: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/add-url")
async def add_url(data: ImageURL):
    """
    Add image from URL → Detect windshield → Crop → Detect faces → Save
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
        faces, detection_time = face_detector.detect_faces(crop_bytes)
        
        # Save to dataset
        url_filename = str(data.url).split('/')[-1] or "url_image"
        save_info = dataset_manager.save_sample(
            full_image,
            windshield_crop,
            faces,
            windshield_info,
            url_filename
        )
        
        return {
            "status": "success",
            "url": str(data.url),
            "windshield_detection": {
                "detected": windshield_info is not None,
                "confidence": windshield_info['confidence'] if windshield_info else None
            },
            "face_detection": {
                "num_faces": len(faces),
                "detection_time": detection_time
            },
            "saved": save_info,
            "dataset_stats": dataset_manager.get_stats()
        }
        
    except Exception as e:
        logger.error(f"Error processing URL: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch-urls")
async def batch_urls(data: URLBatch):
    """Batch process multiple URLs"""
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
            
            url_filename = url.split('/')[-1] or "url_image"
            save_info = dataset_manager.save_sample(
                full_image,
                windshield_crop,
                faces,
                windshield_info,
                url_filename
            )
            
            results.append({
                "url": url,
                "status": "success",
                "windshield_detected": windshield_info is not None,
                "num_faces": len(faces),
                "saved_as": save_info["filename"]
            })
            
            time.sleep(batch_delay)
            
        except Exception as e:
            results.append({
                "url": url,
                "status": "failed",
                "error": str(e)
            })
    
    return {
        "status": "completed",
        "total_urls": len(data.urls),
        "successful": len([r for r in results if r["status"] == "success"]),
        "failed": len([r for r in results if r["status"] == "failed"]),
        "results": results,
        "dataset_stats": dataset_manager.get_stats()
    }


@app.get("/stats")
async def get_stats():
    """Get dataset statistics"""
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


@app.get("/models")
async def get_models():
    """Get information about loaded models"""
    if not windshield_detector.is_available():
        return {"windshield_models": [], "status": "disabled"}
    
    return {
        "windshield_models": windshield_detector.get_model_info(),
        "ensemble_config": windshield_detector.ensemble_config,
        "status": "active"
    }


@app.get("/download")
async def download_dataset():
    """Download complete dataset as ZIP"""
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


@app.get("/visualize/{filename}")
async def get_visualization(filename: str):
    """Get visualization image"""
    vis_path = dataset_manager.get_visualization_path(filename)
    
    if vis_path is None:
        raise HTTPException(status_code=404, detail="Visualization not found")
    
    return FileResponse(vis_path, media_type="image/jpeg")


@app.delete("/clear")
async def clear_dataset():
    """Clear all dataset data"""
    try:
        dataset_manager.clear_dataset()
        return {
            "status": "success",
            "message": "Dataset cleared successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "clarifai_available": ClarifaiFaceDetector.is_available(),
        "windshield_detection_available": windshield_detector.is_available(),
        "num_windshield_models": len(windshield_detector.models) if windshield_detector.is_available() else 0,
        "dataset_stats": dataset_manager.get_stats()
    }


if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*70)
    print("🎯 Clarifai Auto-Collect with Multi-Model Windshield Detection")
    print("="*70)
    
    if windshield_detector.is_available():
        print(f"\n✅ Windshield Detection: {len(windshield_detector.models)} model(s) loaded")
        for model_info in windshield_detector.get_model_info():
            print(f"   • {model_info['name']} ({model_info['model_type']})")
        print(f"   Ensemble method: {config.get('ensemble', {}).get('method', 'union')}")
    else:
        print("\n⚠️  Windshield detection disabled (no models loaded)")
    
    print(f"\n✅ Face Detection: Clarifai API")
    
    stats = dataset_manager.get_stats()
    print(f"\n📊 Current dataset: {stats['total_images']} images, {stats['total_faces']} faces")
    
    server_config = config.get('server', {})
    host = server_config.get('host', '0.0.0.0')
    port = server_config.get('port', 8000)
    
    print(f"\n🌐 Starting server at http://{host}:{port}")
    print(f"📖 Documentation: http://{host}:{port}/docs")
    print("="*70 + "\n")
    
    uvicorn.run(app, host=host, port=port)
