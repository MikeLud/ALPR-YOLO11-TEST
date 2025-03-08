"""
State classification for license plates using YOLOv8.
"""
import cv2
import numpy as np
import torch
from typing import Dict, Any, Tuple
from ultralytics import YOLO

from ..config import ALPRConfig
from ..exceptions import ModelLoadingError, InferenceError


class StateClassifier:
    """
    State classifier for license plates using YOLOv8.
    Identifies the state of origin for a license plate.
    """
    
    def __init__(self, config: ALPRConfig):
        """
        Initialize the state classifier.
        
        Args:
            config: ALPR configuration object
            
        Raises:
            ModelLoadingError: If model loading fails
        """
        self.config = config
        self.model_path = config.get_model_path("state_classifier")
        self.confidence_threshold = config.state_classifier_confidence
        self.resolution = (224, 224)  # Standard resolution for classification
        
        # Skip initialization if state detection is disabled
        if not config.enable_state_detection:
            self.model = None
            return
            
        # Initialize the model
        try:
            self.model = YOLO(self.model_path, task='classify')
        except Exception as e:
            raise ModelLoadingError(self.model_path, e)
    
    def classify(self, plate_image: np.ndarray) -> Dict[str, Any]:
        """
        Classify the state of the license plate.
        
        Args:
            plate_image: License plate image as numpy array
            
        Returns:
            Dictionary with state name and confidence
            
        Raises:
            InferenceError: If classification fails
        """
        if self.model is None:
            return {"state": "Unknown", "confidence": 0.0}
        
        # Resize plate image for state classifier
        plate_resized = cv2.resize(plate_image, self.resolution)
        
        try:
            # Run state classification model
            results = self.model(plate_resized, conf=self.confidence_threshold, verbose=False)[0]
        except Exception as e:
            raise InferenceError("state_classifier", e)
        
        # Get the predicted class and confidence
        if hasattr(results, 'probs') and hasattr(results.probs, 'top1'):
            state_idx = int(results.probs.top1)
            confidence = float(results.probs.top1conf.item())
            
            # Convert class index to state name
            state_names = self.model.names
            state_name = state_names[state_idx]
            
            return {"state": state_name, "confidence": confidence}
        
        return {"state": "Unknown", "confidence": 0.0}
    
    def __call__(self, plate_image: np.ndarray) -> Dict[str, Any]:
        """
        Convenience method to call classify().
        
        Args:
            plate_image: License plate image
            
        Returns:
            Classification results
        """
        return self.classify(plate_image)
