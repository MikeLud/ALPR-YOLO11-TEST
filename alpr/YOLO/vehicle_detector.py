"""
Vehicle detection and classification using YOLOv8.
"""
import cv2
import numpy as np
import torch
from typing import List, Dict, Any, Tuple
from ultralytics import YOLO

from ..config import ALPRConfig
from ..exceptions import ModelLoadingError, InferenceError, VehicleDetectionError


class VehicleDetector:
    """
    Vehicle detector and classifier using YOLOv8.
    Detects vehicles and identifies make/model.
    """
    
    def __init__(self, config: ALPRConfig):
        """
        Initialize the vehicle detector and classifier.
        
        Args:
            config: ALPR configuration object
            
        Raises:
            ModelLoadingError: If model loading fails
        """
        self.config = config
        self.vehicle_detector_path = config.get_model_path("vehicle_detector")
        self.vehicle_classifier_path = config.get_model_path("vehicle_classifier")
        self.vehicle_detector_confidence = config.vehicle_detector_confidence
        self.vehicle_classifier_confidence = config.vehicle_classifier_confidence
        
        # Model resolutions
        self.vehicle_detector_resolution = (640, 640)
        self.vehicle_classifier_resolution = (224, 224)
        
        # Skip initialization if vehicle detection is disabled
        if not config.enable_vehicle_detection:
            self.detector_model = None
            self.classifier_model = None
            return
            
        # Initialize the models
        try:
            self.detector_model = YOLO(self.vehicle_detector_path, task='detect')
        except Exception as e:
            raise ModelLoadingError(self.vehicle_detector_path, e)
            
        try:
            self.classifier_model = YOLO(self.vehicle_classifier_path, task='classify')
        except Exception as e:
            raise ModelLoadingError(self.vehicle_classifier_path, e)
    
    def detect_vehicles(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect vehicles in the image.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            List of dictionaries with vehicle information
            
        Raises:
            InferenceError: If detection fails
        """
        if self.detector_model is None:
            return []
        
        # Resize image for vehicle detector
        img_resized = cv2.resize(image, self.vehicle_detector_resolution)
        
        try:
            # Run vehicle detection model
            results = self.detector_model(
                img_resized, 
                conf=self.vehicle_detector_confidence, 
                verbose=False
            )[0]
        except Exception as e:
            raise InferenceError("vehicle_detector", e)
        
        # Process the results to extract vehicle bounding boxes
        vehicles = []
        
        if hasattr(results, 'boxes') and hasattr(results.boxes, 'xyxy'):
            for i, box in enumerate(results.boxes.xyxy):
                x1, y1, x2, y2 = box.cpu().numpy()
                
                # Scale the coordinates back to the original image size
                h, w = image.shape[:2]
                scale_x = w / self.vehicle_detector_resolution[0]
                scale_y = h / self.vehicle_detector_resolution[1]
                
                x1 = int(x1 * scale_x)
                y1 = int(y1 * scale_y)
                x2 = int(x2 * scale_x)
                y2 = int(y2 * scale_y)
                
                # Ensure the box coordinates are within the image bounds
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)
                
                # Skip invalid boxes
                if x1 >= x2 or y1 >= y2:
                    continue
                
                confidence = float(results.boxes.conf[i].item()) if hasattr(results.boxes, 'conf') else 0.0
                
                # Extract the vehicle region
                vehicle_img = image[y1:y2, x1:x2]
                
                vehicles.append({
                    "box": [x1, y1, x2, y2],
                    "confidence": confidence,
                    "image": vehicle_img
                })
        
        return vehicles
    
    def classify_vehicle(self, vehicle_image: np.ndarray) -> Dict[str, Any]:
        """
        Classify vehicle make and model.
        
        Args:
            vehicle_image: Vehicle image to classify
            
        Returns:
            Dictionary with make, model, and confidence
            
        Raises:
            InferenceError: If classification fails
            VehicleDetectionError: If vehicle image is invalid
        """
        if self.classifier_model is None or vehicle_image.size == 0:
            return {"make": "Unknown", "model": "Unknown", "confidence": 0.0}
        
        # Resize vehicle image for classifier
        try:
            vehicle_resized = cv2.resize(vehicle_image, self.vehicle_classifier_resolution)
        except Exception as e:
            raise VehicleDetectionError(f"Failed to resize vehicle image: {str(e)}")
        
        try:
            # Run vehicle classification model
            results = self.classifier_model(
                vehicle_resized, 
                conf=self.vehicle_classifier_confidence, 
                verbose=False
            )[0]
        except Exception as e:
            raise InferenceError("vehicle_classifier", e)
        
        # Get the predicted class and confidence
        if hasattr(results, 'probs') and hasattr(results.probs, 'top1'):
            vehicle_idx = int(results.probs.top1)
            confidence = float(results.probs.top1conf.item())
            
            # Convert class index to make and model
            vehicle_names = self.classifier_model.names
            make_model = vehicle_names[vehicle_idx]
            
            # Split make and model (assuming format "Make_Model")
            make, model = make_model.split("_", 1) if "_" in make_model else (make_model, "Unknown")
            
            return {"make": make, "model": model, "confidence": confidence}
        
        return {"make": "Unknown", "model": "Unknown", "confidence": 0.0}
    
    def detect_and_classify(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect vehicles and classify their make/model.
        
        Args:
            image: Input image
            
        Returns:
            List of dictionaries with vehicle information
        """
        # Detect vehicles
        vehicles = self.detect_vehicles(image)
        
        # Classify each vehicle
        vehicle_results = []
        for vehicle_info in vehicles:
            try:
                classification = self.classify_vehicle(vehicle_info["image"])
                
                # Create result with bounding box and classification
                result = {
                    "box": vehicle_info["box"],
                    "confidence": vehicle_info["confidence"],
                    "make": classification["make"],
                    "model": classification["model"],
                    "classification_confidence": classification["confidence"]
                }
                vehicle_results.append(result)
            except Exception as e:
                # Skip this vehicle if classification fails
                continue
                
        return vehicle_results
    
    def __call__(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Convenience method to call detect_and_classify().
        
        Args:
            image: Input image
            
        Returns:
            Detection and classification results
        """
        return self.detect_and_classify(image)
