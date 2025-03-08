"""
License plate detection module using YOLOv8 keypoint detection model.
"""
import os
import cv2
import numpy as np
import torch
from typing import Dict, List, Any, Optional
from ultralytics import YOLO

from ..config import ALPRConfig
from ..exceptions import ModelLoadingError, InferenceError
from ..utils.image_processing import dilate_corners


class PlateDetector:
    """
    License plate detector using YOLOv8 keypoint detection model.
    Detects both day and night license plates in images.
    """
    
    def __init__(self, config: ALPRConfig):
        """
        Initialize the license plate detector.
        
        Args:
            config: ALPR configuration object
            
        Raises:
            ModelLoadingError: If model loading fails
        """
        self.config = config
        self.model_path = config.get_model_path("plate_detector")
        self.confidence_threshold = config.plate_detector_confidence
        self.corner_dilation_pixels = config.corner_dilation_pixels
        self.resolution = (640, 640)  # Standard YOLOv8 input resolution
        
        # Initialize the model
        try:
            self.model = YOLO(self.model_path, task='pose')
        except Exception as e:
            raise ModelLoadingError(self.model_path, e)
    
    def detect(self, image: np.ndarray) -> Dict[str, List[Dict[str, Any]]]:
        """
        Detect license plates in the image.
        
        Args:
            image: Input image as numpy array (BGR format)
            
        Returns:
            Dictionary with 'day_plates' and 'night_plates' lists containing detection results
            
        Raises:
            InferenceError: If detection fails
        """
        # Resize image for plate detector
        h, w = image.shape[:2]
        img_resized = cv2.resize(image, self.resolution)
        
        try:
            # Run YOLOv8 keypoint detection model to detect plate corners
            results = self.model(img_resized, conf=self.confidence_threshold, verbose=False)[0]
        except Exception as e:
            raise InferenceError("plate_detector", e)
        
        # Process the results to extract plate corners
        day_plates = []
        night_plates = []
        
        # The model returns keypoints for each detected plate (4 corners)
        if hasattr(results, 'keypoints') and results.keypoints is not None:
            for i, keypoints in enumerate(results.keypoints.data):
                if len(keypoints) >= 4:  # Ensure we have at least 4 keypoints (corners)
                    # Get the 4 corner points
                    corners = keypoints[:4].cpu().numpy()  # Format: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
                    
                    # Scale the keypoints back to original image size
                    scale_x = w / self.resolution[0]
                    scale_y = h / self.resolution[1]
                    
                    scaled_corners = []
                    for corner in corners:
                        # Handle different possible formats of keypoint data
                        try:
                            if len(corner) >= 3:  # Format may include confidence value or other data
                                x, y = corner[0], corner[1]
                            else:
                                x, y = corner
                            
                            scaled_corners.append([float(x * scale_x), float(y * scale_y)])
                        except Exception as e:
                            # Use a default value to avoid breaking the pipeline
                            scaled_corners.append([0.0, 0.0])
                    
                    # Convert to numpy array for dilation
                    scaled_corners_np = np.array(scaled_corners, dtype=np.float32)
                    
                    # Apply dilation to the corners
                    dilated_corners_np = dilate_corners(scaled_corners_np, self.corner_dilation_pixels)
                    
                    # Convert back to list format
                    dilated_corners = dilated_corners_np.tolist()
                    
                    # Store both original and dilated corners for visualization
                    original_corners = scaled_corners.copy()
                    
                    # Get the detection box if available
                    detection_box = None
                    if hasattr(results.boxes, 'xyxy') and i < len(results.boxes.xyxy):
                        box = results.boxes.xyxy[i].cpu().numpy()
                        # Scale box coordinates
                        x1, y1, x2, y2 = box
                        x1 = int(x1 * scale_x) - 15
                        y1 = int(y1 * scale_y) - 15
                        x2 = int(x2 * scale_x) + 15
                        y2 = int(y2 * scale_y) + 15
                        detection_box = [x1, y1, x2, y2]  # [x1, y1, x2, y2] format
                    
                    # Determine if it's a day plate or night plate based on the class
                    # Assuming class 0 is day plate and class 1 is night plate
                    if hasattr(results.boxes, 'cls') and i < len(results.boxes.cls):
                        plate_class = int(results.boxes.cls[i].item())
                        
                        plate_info = {
                            "corners": dilated_corners,  # Use dilated corners for processing
                            "original_corners": original_corners,  # Keep original corners for reference
                            "detection_box": detection_box,
                            "confidence": float(results.boxes.conf[i].item()) if hasattr(results.boxes, 'conf') else 0.0
                        }
                        
                        if plate_class == 0:  # Day plate
                            day_plates.append(plate_info)
                        else:  # Night plate
                            night_plates.append(plate_info)
        
        return {
            "day_plates": day_plates,
            "night_plates": night_plates
        }
    
    def __call__(self, image: np.ndarray) -> Dict[str, List[Dict[str, Any]]]:
        """
        Convenience method to call detect().
        
        Args:
            image: Input image
            
        Returns:
            Detection results
        """
        return self.detect(image)
