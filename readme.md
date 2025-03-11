# Automatic License Plate Recognition (ALPR) Module for CodeProject.AI Server

This is an Automatic License Plate Recognition (ALPR) module for [CodeProject.AI Server](https://www.codeproject.com/Articles/5322557/CodeProject-AI-Server-AI-the-easy-way). The module can detect license plates in images, recognize characters, identify states, and detect vehicles with make/model classifications.

## Features

- License plate detection for both day and night plates
- Character recognition on license plates
- State classification for license plates (when enabled)
- Vehicle detection and make/model classification (when enabled)
- Support for GPU acceleration via CUDA (NVIDIA) or MPS (Apple Silicon)
- Configurable confidence thresholds and plate aspect ratios
- Support for both PyTorch and ONNX model formats
- Debug image saving for visualizing detection pipeline stages

## API Endpoint

The module provides a single API endpoint with different operation modes:

```
POST /v1/vision/alpr
```

Parameters:
- `operation`: The type of analysis to perform
  - `plate`: Detect only license plates
  - `vehicle`: Detect only vehicles and their make/model
  - `full`: Complete analysis (both license plates and vehicles)
- `min_confidence`: Minimum confidence threshold for detections (0.0 to 1.0)

## Technical Details

This module uses YOLOv8 models for various detection and recognition tasks:

- **plate_detector.pt/onnx**: Detects license plates in the image
- **state_classifier.pt/onnx**: Identifies the US state for a license plate
- **char_detector.pt/onnx**: Detects individual characters on the license plate
- **char_classifier.pt/onnx**: Recognizes each character (OCR)
- **vehicle_detector.pt/onnx**: Detects vehicles in the image
- **vehicle_classifier.pt/onnx**: Identifies vehicle make and model

Both PyTorch (.pt) and ONNX (.onnx) model formats are supported.

## Configuration

The module supports several configuration options through environment variables:

- `ENABLE_STATE_DETECTION`: Enable/disable state identification
- `ENABLE_VEHICLE_DETECTION`: Enable/disable vehicle detection
- `PLATE_ASPECT_RATIO`: Set a specific aspect ratio for license plates
- `CORNER_DILATION_PIXELS`: Configure corner dilation for license plate extraction
- `USE_ONNX`: Use ONNX models for faster inference (default: false, uses PyTorch)
- `ONNX_MODELS_DIR`: Directory path for ONNX models (default: "models/onnx")
- `SAVE_DEBUG_IMAGES`: Enable/disable saving debug images (default: false)
- `DEBUG_IMAGES_DIR`: Directory path for debug images (default: "debug_images")
- Various confidence thresholds for different detection components

## Debug Image Support

This module includes a debug image feature that saves intermediate results during license plate detection and recognition. This is useful for:

- Troubleshooting detection issues
- Visualizing each stage of the pipeline
- Understanding how the models are processing images
- Tuning confidence thresholds and other parameters

When enabled, debug images are saved to the configured directory for each step of the detection process:
- Original input images
- Resized input images for each model
- Model outputs and predictions
- Plate detections with corners
- Cropped license plates
- Character detections within plates
- State classification results
- Vehicle detections and classifications
- Final annotated results

To enable debug images, set `SAVE_DEBUG_IMAGES` to "True" in the environment variables or use the "Debug Image Saving" option in the UI. You can also configure a custom directory path using the `DEBUG_IMAGES_DIR` setting.

## ONNX Support

This module can use ONNX models for potentially faster inference, especially on edge devices or when GPU acceleration is not available. ONNX models are stored in the "models/onnx" directory and can be enabled through the "Model Format" option in the UI or by setting the `USE_ONNX` environment variable to "True".

Benefits of ONNX:
- Faster inference on many platforms
- Better compatibility with different hardware
- Optimized runtime performance

## Requirements

- Python 3.8+ 
- PyTorch
- OpenCV
- Ultralytics YOLOv8
- ONNX Runtime (optional, for ONNX models)