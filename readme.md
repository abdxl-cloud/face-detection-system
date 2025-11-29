# Clarifai Auto-Collect with Multi-Model Windshield Detection

A FastAPI application that automatically collects face detection training data by:
1. Detecting windshields using **multiple YOLO models** (ensemble)
2. Cropping windshield regions
3. Detecting faces using **Clarifai API**
4. Automatically saving to YOLO-format dataset

## 🎯 Features

- ✅ **Multi-Model YOLO Windshield Detection** - Use 1-N models with ensemble
- ✅ **4 Ensemble Methods** - Union, NMS, Weighted, Best Confidence
- ✅ **Clarifai Face Detection** - High-accuracy face detection
- ✅ **Automatic Dataset Creation** - No manual work required
- ✅ **YOLO Format Export** - Ready for training
- ✅ **Batch Processing** - Handle multiple images
- ✅ **Visualizations** - See detections with bounding boxes

## 📁 File Structure

```
.
├── clarifai_config.yaml      # Configuration file
├── clarifai_main.py           # Main FastAPI application
├── windshield_detector.py     # Multi-model YOLO windshield detection
├── clarifai_client.py         # Clarifai API client
├── dataset_manager.py         # Dataset management
├── utils.py                   # Helper functions
└── README.md                  # This file
```

## 🚀 Installation

### 1. Install Dependencies

```bash
pip install fastapi uvicorn pydantic
pip install opencv-python numpy pyyaml requests
pip install ultralytics  # For YOLO
pip install clarifai-grpc  # For Clarifai API
```

### 2. Configure

Edit `clarifai_config.yaml`:

```yaml
# Add your Clarifai API key
clarifai:
  api_key: "YOUR_API_KEY_HERE"

# Configure windshield detection models
windshield_models:
  - name: "model_1"
    enabled: true
    model_path: "models/windshield_model_1.pt"
    conf_threshold: 0.35
    weight: 1.0
    
  - name: "model_2"
    enabled: true  # Enable/disable as needed
    model_path: "models/windshield_model_2.pt"
    conf_threshold: 0.4
    weight: 1.2
```

### 3. Place YOLO Models

Put your YOLO windshield detection models in the `models/` directory:

```bash
mkdir models
# Copy your models
cp /path/to/windshield_model.pt models/windshield_model_1.pt
```

## 🏃 Running

```bash
python clarifai_main.py
```

Server starts at: `http://localhost:8000`

## 📖 API Usage

### Upload Image

```bash
curl -X POST "http://localhost:8000/upload" \
  -F "file=@car_photo.jpg"
```

### Add from URL

```bash
curl -X POST "http://localhost:8000/add-url" \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com/car.jpg"}'
```

### Batch URLs

```bash
curl -X POST "http://localhost:8000/batch-urls" \
  -H "Content-Type: application/json" \
  -d '{
    "urls": [
      "https://example.com/car1.jpg",
      "https://example.com/car2.jpg"
    ]
  }'
```

### Check Statistics

```bash
curl http://localhost:8000/stats
```

### View Model Info

```bash
curl http://localhost:8000/models
```

### Download Dataset

```bash
curl http://localhost:8000/download -o dataset.zip
```

### Clear Dataset

```bash
curl -X DELETE http://localhost:8000/clear
```

## ⚙️ Configuration Options

### Ensemble Methods

Choose how multiple windshield detections are combined:

```yaml
ensemble:
  method: "union"  # Options: union, nms, weighted, best_confidence
```

- **union**: Largest bbox containing all detections
- **nms**: Non-Maximum Suppression
- **weighted**: Weighted average based on model weights
- **best_confidence**: Pick detection with highest confidence

### Model Configuration

Each model can be configured independently:

```yaml
windshield_models:
  - name: "yolov8_model"
    enabled: true
    model_path: "models/windshield_v8.pt"
    model_type: "yolov8"
    conf_threshold: 0.35
    iou_threshold: 0.45
    device: "cuda"  # or "cpu"
    weight: 1.0  # Used in weighted ensemble
```

## 📊 Dataset Structure

```
face_dataset/
├── images/
│   ├── train/          # Training windshield crops
│   └── val/            # Validation windshield crops
├── labels/
│   ├── train/          # YOLO format labels
│   └── val/            # YOLO format labels
├── raw_images/         # Original full images
├── windshield_crops/   # All windshield crops
├── visualizations/     # Annotated images
├── data.yaml           # YOLO config
└── stats.json          # Dataset statistics
```

## 🔄 Workflow

1. **Upload Image** → Full vehicle image uploaded
2. **Windshield Detection** → All enabled YOLO models detect windshield
3. **Ensemble** → Detections combined using configured method
4. **Crop** → Windshield region extracted
5. **Face Detection** → Clarifai detects faces in crop
6. **Save** → Everything saved automatically to dataset

## 📈 Training with Dataset

After downloading the dataset:

```bash
# Unzip
unzip dataset.zip

# Train with YOLOv8
yolo task=detect mode=train \
  model=yolov8n.pt \
  data=face_dataset/data.yaml \
  epochs=100 \
  imgsz=640

# Train with YOLOv11
yolo task=detect mode=train \
  model=yolo11n.pt \
  data=face_dataset/data.yaml \
  epochs=100
```

## 🔍 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web interface |
| `/upload` | POST | Upload image file |
| `/add-url` | POST | Add image from URL |
| `/batch-urls` | POST | Process multiple URLs |
| `/stats` | GET | Get dataset statistics |
| `/models` | GET | Get loaded models info |
| `/download` | GET | Download dataset ZIP |
| `/visualize/{filename}` | GET | View visualization |
| `/clear` | DELETE | Clear all data |
| `/health` | GET | Health check |
| `/docs` | GET | Swagger UI |

## 🎛️ Advanced Configuration

### Multiple Models Example

```yaml
windshield_models:
  - name: "yolov8_base"
    enabled: true
    model_path: "models/yolov8_windshield.pt"
    weight: 1.0
    
  - name: "yolov11_improved"
    enabled: true
    model_path: "models/yolov11_windshield.pt"
    weight: 1.5  # Higher weight if more accurate
    
  - name: "yolov12_latest"
    enabled: true
    model_path: "models/yolov12_windshield.pt"
    weight: 2.0  # Highest weight for best model
```

### Ensemble Settings

```yaml
ensemble:
  method: "weighted"
  nms_iou_threshold: 0.5
  min_models_agreement: 2  # Require at least 2 models
  confidence_threshold: 0.3
```

## 📝 Response Example

```json
{
  "status": "success",
  "windshield_detection": {
    "detected": true,
    "confidence": 0.89,
    "method": "weighted",
    "num_models": 3
  },
  "face_detection": {
    "num_faces": 2,
    "detection_time": 1.23
  },
  "saved": {
    "filename": "face_20240115_123456",
    "split": "train",
    "num_faces": 2,
    "windshield_detected": true
  },
  "dataset_stats": {
    "total_images": 150,
    "total_faces": 287,
    "total_windshields": 142
  }
}
```

## 🐛 Troubleshooting

### No windshield detection

- Check model paths in config
- Verify models are enabled
- Check YOLO installation: `pip install ultralytics`

### Clarifai errors

- Verify API key in config
- Check installation: `pip install clarifai-grpc`
- Check API quota/rate limits

### CUDA errors

- Set `device: "cpu"` in model config
- Or install CUDA-compatible PyTorch

## 📄 License

For educational and development purposes.

## 🤝 Contributing

Feel free to improve the ensemble methods, add new features, or optimize the detection pipeline!
