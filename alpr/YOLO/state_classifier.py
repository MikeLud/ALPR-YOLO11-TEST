"""
State classification for license plates using YOLOv8.
"""
import cv2
import numpy as np
import torch
from typing import Dict, Any, Tuple
from ultralytics import YOLO

from .base import YOLOBase
from ..config import ALPRConfig
from ..exceptions import ModelLoadingError, InferenceError


class StateClassifier(YOLOBase):
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
            self.onnx_session = None
            return
            
        # Initialize the model
        try:
            super().__init__(
                model_path=self.model_path,
                task='classify',
                use_onnx=config.use_onnx,
                use_cuda=config.use_cuda
            )
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
        if self.model is None and self.onnx_session is None:
            return {"state": "Unknown", "confidence": 0.0}
        
        # Resize plate image for state classifier
        plate_resized = cv2.resize(plate_image, self.resolution)
        
        try:
            if self.use_onnx:
                return self._classify_onnx(plate_resized)
            else:
                return self._classify_pytorch(plate_resized)
        except Exception as e:
            raise InferenceError("state_classifier", e)
    
    def _classify_pytorch(self, plate_image: np.ndarray) -> Dict[str, Any]:
        """Classify state using PyTorch model"""
        # Run state classification model
        results = self.model(plate_image, conf=self.confidence_threshold, verbose=False)[0]
        
        # Get the predicted class and confidence
        if hasattr(results, 'probs') and hasattr(results.probs, 'top1'):
            state_idx = int(results.probs.top1)
            confidence = float(results.probs.top1conf.item())
            
            # Convert class index to state name
            state_names = self.model.names
            state_name = state_names[state_idx]
            
            return {"state": state_name, "confidence": confidence}
        
        return {"state": "Unknown", "confidence": 0.0}
    
    def _classify_onnx(self, plate_image: np.ndarray) -> Dict[str, Any]:
        """Classify state using ONNX model"""
        # Preprocess image for ONNX
        input_tensor = self._preprocess_image(plate_image, self.resolution)
        
        # Run inference
        outputs = self.onnx_session.run(
            self.output_names, 
            {self.input_name: input_tensor}
        )
        
        # Parse ONNX outputs
        # For classification, output is typically class probabilities
        probs = outputs[0]
        
        if probs is not None and len(probs) > 0:
            # Get the index with highest probability
            state_idx = np.argmax(probs[0])
            confidence = float(probs[0][state_idx])
            
            # Get state name from class index
            state_name = self.names.get(str(state_idx), self.names.get(state_idx, "Unknown"))
            
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
