"""
License plate detection module using YOLOv8 keypoint detection model.
"""
import os
import cv2
import numpy as np
import torch
from typing import Dict, List, Any, Optional, Tuple
from ultralytics import YOLO

from .base import YOLOBase
from ..config import ALPRConfig
from ..exceptions import ModelLoadingError, InferenceError
from ..utils.image_processing import dilate_corners, save_debug_image


class PlateDetector(YOLOBase):
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
        self.save_debug_images = config.save_debug_images
        self.debug_images_dir = config.debug_images_dir
        
        # Initialize the model
        try:
            super().__init__(
                model_path=self.model_path,
                task='pose',
                use_onnx=config.use_onnx,
                use_cuda=config.use_cuda
            )
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
        # Get original image dimensions
        h, w = image.shape[:2]
        
        # Save input image for debugging if enabled
        if self.save_debug_images:
            save_debug_image(
                image=image,
                debug_dir=self.debug_images_dir,
                prefix="plate_detector",
                suffix="input",
                draw_objects=None,
                draw_type=None
            )
        
        try:
            if self.use_onnx:
                results = self._detect_onnx(image)
            else:
                # Run YOLOv8 keypoint detection model to detect plate corners
                # Use original image without resizing
                results = self.model(image, conf=self.confidence_threshold, verbose=False)[0]
                
                # Save model output visualization if debugging is enabled
                if self.save_debug_images and hasattr(results, 'plot'):
                    try:
                        # Plot the results using the model's built-in plotting
                        plot_img = results.plot()
                        save_debug_image(
                            image=plot_img,
                            debug_dir=self.debug_images_dir,
                            prefix="plate_detector",
                            suffix="model_output",
                            draw_objects=None,
                            draw_type=None
                        )
                    except Exception as e:
                        print(f"Error plotting plate detection results: {e}")
        except Exception as e:
            raise InferenceError("plate_detector", e)
        
        # Process the results to extract plate corners
        day_plates = []
        night_plates = []
        
        if self.use_onnx:
            # Process ONNX results
            if 'keypoints' in results and len(results['keypoints']) > 0:
                for i, keypoints in enumerate(results['keypoints']):
                    if len(keypoints) >= 4:  # Ensure we have at least 4 keypoints (corners)
                        # Get class and confidence
                        class_id = results['classes'][i]
                        confidence = results['confidences'][i]
                        
                        # Get the bounding box if available
                        detection_box = None
                        if 'boxes' in results and i < len(results['boxes']):
                            box = results['boxes'][i]
                            x1, y1, x2, y2 = box
                            detection_box = [
                                int(x1) - 15,
                                int(y1) - 15,
                                int(x2) + 15,
                                int(y2) + 15
                            ]
                        
                        # Use keypoints directly without scaling
                        corners = []
                        for kp in keypoints[:4]:
                            if len(kp) >= 2:
                                x, y = kp[0], kp[1]
                                corners.append([float(x), float(y)])
                            else:
                                corners.append([0.0, 0.0])
                        
                        # Convert to numpy array for dilation
                        corners_np = np.array(corners, dtype=np.float32)
                        
                        # Apply dilation to the corners
                        dilated_corners_np = dilate_corners(corners_np, self.corner_dilation_pixels)
                        
                        # Convert back to list format
                        dilated_corners = dilated_corners_np.tolist()
                        original_corners = corners.copy()
                        
                        plate_info = {
                            "corners": dilated_corners,
                            "original_corners": original_corners,
                            "detection_box": detection_box,
                            "confidence": float(confidence)
                        }
                        
                        if class_id == 0:  # Day plate
                            day_plates.append(plate_info)
                        else:  # Night plate
                            night_plates.append(plate_info)
        else:
            # Process PyTorch YOLOv8 results
            # The model returns keypoints for each detected plate (4 corners)
            if hasattr(results, 'keypoints') and results.keypoints is not None:
                for i, keypoints in enumerate(results.keypoints.data):
                    if len(keypoints) >= 4:  # Ensure we have at least 4 keypoints (corners)
                        # Get the 4 corner points
                        corners = keypoints[:4].cpu().numpy()  # Format: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
                        
                        # No scaling needed since we're using original image
                        scaled_corners = []
                        for corner in corners:
                            # Handle different possible formats of keypoint data
                            try:
                                if len(corner) >= 3:  # Format may include confidence value or other data
                                    x, y = corner[0], corner[1]
                                else:
                                    x, y = corner
                                
                                scaled_corners.append([float(x), float(y)])
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
                            # Add padding to the box
                            x1, y1, x2, y2 = box
                            x1 = int(x1) - 15
                            y1 = int(y1) - 15
                            x2 = int(x2) + 15
                            y2 = int(y2) + 15
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
        
        # Save debug images with both day and night plates if enabled
        if self.save_debug_images:
            # Create debug image with plate detections
            debug_img = image.copy()
            
            # Draw day plates in green
            for plate in day_plates:
                corners = np.array(plate['corners'], dtype=np.int32)
                cv2.polylines(debug_img, [corners], True, (0, 255, 0), 2)
                
                # Add "Day" label and confidence
                if len(corners) > 0:
                    x, y = corners[0]
                    cv2.putText(debug_img, f"Day ({plate['confidence']:.2f})", (int(x), int(y) - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Draw night plates in blue
            for plate in night_plates:
                corners = np.array(plate['corners'], dtype=np.int32)
                cv2.polylines(debug_img, [corners], True, (255, 0, 0), 2)
                
                # Add "Night" label and confidence
                if len(corners) > 0:
                    x, y = corners[0]
                    cv2.putText(debug_img, f"Night ({plate['confidence']:.2f})", (int(x), int(y) - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            # Save the debug image
            save_debug_image(
                image=debug_img,
                debug_dir=self.debug_images_dir,
                prefix="plate_detector",
                suffix="processed_output",
                draw_objects=None,
                draw_type=None
            )
        
        return {
            "day_plates": day_plates,
            "night_plates": night_plates
        }
    
    def _detect_onnx(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Detect plates using ONNX model.
        
        Args:
            image: Input image
            
        Returns:
            Dictionary with detection results
        """
        # Preprocess image for ONNX without resizing
        input_tensor = self._preprocess_image(image)
        
        # Run inference
        outputs = self.onnx_session.run(
            self.output_names, 
            {self.input_name: input_tensor}
        )
        
        # Structure to hold results similar to YOLO output format
        results = {
            'boxes': [],        # Format: [x1, y1, x2, y2]
            'classes': [],      # Class IDs
            'confidences': [],  # Confidence scores
            'keypoints': []     # Keypoints (4 corners for plates)
        }
        
        # Parse ONNX outputs (format depends on the exported model)
        # This implementation assumes a specific output format - adjust based on your model
        
        # Get boxes, scores, classes from appropriate outputs
        # Example parsing - adjust based on actual ONNX model output structure
        if len(outputs) >= 3:  # Typical detection output has boxes, scores, classes
            boxes = outputs[0]
            scores = outputs[1]
            classes = outputs[2].astype(np.int32)
            
            # If keypoints are included, they'll usually be in a separate output
            keypoints = outputs[3] if len(outputs) > 3 else None
            
            # Filter by confidence threshold
            for i in range(len(scores)):
                if scores[i] >= self.confidence_threshold:
                    results['boxes'].append(boxes[i])
                    results['classes'].append(classes[i])
                    results['confidences'].append(scores[i])
                    if keypoints is not None:
                        results['keypoints'].append(keypoints[i])
        
        return results
    
    def __call__(self, image: np.ndarray) -> Dict[str, List[Dict[str, Any]]]:
        """
        Convenience method to call detect().
        
        Args:
            image: Input image
            
        Returns:
            Detection results
        """
        return self.detect(image)
