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
from ..utils.image_processing import save_debug_image


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
        self.save_debug_images = config.save_debug_images
        self.debug_images_dir = config.debug_images_dir
        
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
        
        # Save resized input image for debugging if enabled
        if self.save_debug_images:
            save_debug_image(
                image=plate_resized,
                debug_dir=self.debug_images_dir,
                prefix="state_classifier",
                suffix="resized_input",
                draw_objects=None,
                draw_type=None
            )
        
        try:
            if self.use_onnx:
                result = self._classify_onnx(plate_resized)
            else:
                result = self._classify_pytorch(plate_resized)
                
            # Save classification result visualization if debug is enabled
            if self.save_debug_images:
                # Create a visualization of the state classification result
                result_img = plate_resized.copy()
                h, w = result_img.shape[:2]
                
                # Add some space at the bottom for the label
                label_bg = np.zeros((60, w, 3), dtype=np.uint8)
                result_img = np.vstack([result_img, label_bg])
                
                # Draw the state and confidence
                state_text = f"State: {result['state']}"
                confidence_text = f"Confidence: {result['confidence']:.2f}"
                
                cv2.putText(result_img, state_text, (10, h+30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(result_img, confidence_text, (10, h+55), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                
                save_debug_image(
                    image=result_img,
                    debug_dir=self.debug_images_dir,
                    prefix="state_classifier",
                    suffix=f"result_{result['state']}",
                    draw_objects=None,
                    draw_type=None
                )
                
            return result
        except Exception as e:
            raise InferenceError("state_classifier", e)
    
    def _classify_pytorch(self, plate_image: np.ndarray) -> Dict[str, Any]:
        """Classify state using PyTorch model"""
        # Run state classification model
        results = self.model(plate_image, conf=self.confidence_threshold, verbose=False)[0]
        
        # Save model output visualization if debug is enabled
        if self.save_debug_images and hasattr(results, 'plot'):
            try:
                # Plot the results using the model's built-in plotting
                plot_img = results.plot()
                save_debug_image(
                    image=plot_img,
                    debug_dir=self.debug_images_dir,
                    prefix="state_classifier",
                    suffix="model_output",
                    draw_objects=None,
                    draw_type=None
                )
            except Exception as e:
                print(f"Error plotting state classification results: {e}")
        
        # Get the predicted class and confidence
        if hasattr(results, 'probs') and hasattr(results.probs, 'top1'):
            state_idx = int(results.probs.top1)
            confidence = float(results.probs.top1conf.item())
            
            # Convert class index to state name
            state_names = self.model.names
            state_name = state_names[state_idx]
            
            # Save top probabilities for debugging if enabled
            if self.save_debug_images and hasattr(results.probs, 'data'):
                try:
                    # Get top 5 predictions
                    probs_tensor = results.probs.data
                    values, indices = torch.topk(probs_tensor, min(5, len(probs_tensor)))
                    
                    # Create a blank image to show alternatives
                    alt_img = np.ones((200, 400, 3), dtype=np.uint8) * 255
                    
                    # Title
                    cv2.putText(alt_img, "Top State Predictions:", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                    
                    # Draw each state prediction
                    for i in range(len(values)):
                        idx = int(indices[i].item())
                        conf = float(values[i].item())
                        name = state_names[idx]
                        text = f"{name}: {conf:.4f}"
                        cv2.putText(alt_img, text, (20, 70 + i*25), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
                    
                    # Save the alternatives image
                    save_debug_image(
                        image=alt_img,
                        debug_dir=self.debug_images_dir,
                        prefix="state_classifier",
                        suffix="top_probs",
                        draw_objects=None,
                        draw_type=None
                    )
                except Exception as e:
                    print(f"Error creating top states visualization: {e}")
            
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
            
            # Save top probabilities for debugging if enabled
            if self.save_debug_images:
                try:
                    # Sort indices by probability
                    sorted_indices = np.argsort(-probs[0])[:5]  # Top 5 indices
                    
                    # Create a blank image to show alternatives
                    alt_img = np.ones((200, 400, 3), dtype=np.uint8) * 255
                    
                    # Title
                    cv2.putText(alt_img, "Top State Predictions (ONNX):", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                    
                    # Draw each state prediction
                    for i, idx in enumerate(sorted_indices):
                        conf = float(probs[0][idx])
                        name = self.names.get(str(idx), self.names.get(idx, "Unknown"))
                        text = f"{name}: {conf:.4f}"
                        cv2.putText(alt_img, text, (20, 70 + i*25), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
                    
                    # Save the alternatives image
                    save_debug_image(
                        image=alt_img,
                        debug_dir=self.debug_images_dir,
                        prefix="state_classifier",
                        suffix="top_probs_onnx",
                        draw_objects=None,
                        draw_type=None
                    )
                except Exception as e:
                    print(f"Error creating top states visualization (ONNX): {e}")
            
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
