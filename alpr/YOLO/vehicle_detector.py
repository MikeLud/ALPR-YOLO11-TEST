"""
Vehicle detection and classification using YOLOv8.
"""
import cv2
import numpy as np
import torch
from typing import List, Dict, Any, Tuple
from ultralytics import YOLO

from .base import YOLOBase
from ..config import ALPRConfig
from ..exceptions import ModelLoadingError, InferenceError, VehicleDetectionError
from ..utils.image_processing import save_debug_image


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
        self.save_debug_images = config.save_debug_images
        self.debug_images_dir = config.debug_images_dir
        
        # Model resolutions
        self.vehicle_detector_resolution = (640, 640)
        self.vehicle_classifier_resolution = (224, 224)
        
        # Skip initialization if vehicle detection is disabled
        if not config.enable_vehicle_detection:
            self.detector = None
            self.classifier = None
            return
            
        # Initialize the models
        try:
            self.detector = VehicleDetectorYOLO(
                model_path=self.vehicle_detector_path,
                task='detect',
                use_onnx=config.use_onnx,
                use_cuda=config.use_cuda,
                resolution=self.vehicle_detector_resolution,
                confidence=self.vehicle_detector_confidence,
                save_debug_images=self.save_debug_images,
                debug_images_dir=self.debug_images_dir
            )
        except Exception as e:
            raise ModelLoadingError(self.vehicle_detector_path, e)
            
        try:
            self.classifier = VehicleClassifierYOLO(
                model_path=self.vehicle_classifier_path,
                task='classify',
                use_onnx=config.use_onnx,
                use_cuda=config.use_cuda,
                resolution=self.vehicle_classifier_resolution,
                confidence=self.vehicle_classifier_confidence,
                save_debug_images=self.save_debug_images,
                debug_images_dir=self.debug_images_dir
            )
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
        if self.detector is None:
            return []
        
        return self.detector.detect(image)
    
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
        if self.classifier is None or vehicle_image.size == 0:
            return {"make": "Unknown", "model": "Unknown", "confidence": 0.0}
        
        return self.classifier.classify(vehicle_image)
    
    def detect_and_classify(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect vehicles and classify their make/model.
        
        Args:
            image: Input image
            
        Returns:
            List of dictionaries with vehicle information
        """
        # Save input image if debug is enabled
        if self.save_debug_images:
            save_debug_image(
                image=image,
                debug_dir=self.debug_images_dir,
                prefix="vehicle_process",
                suffix="input",
                draw_objects=None,
                draw_type=None
            )
        
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
        
        # Save final detection and classification results if debug is enabled
        if self.save_debug_images and vehicle_results:
            debug_img = image.copy()
            
            # Draw each detected and classified vehicle
            for i, vehicle in enumerate(vehicle_results):
                x1, y1, x2, y2 = vehicle["box"]
                # Draw box
                cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                
                # Draw make/model
                make_model = f"{vehicle['make']} {vehicle['model']}"
                confidence = f"Det: {vehicle['confidence']:.2f}, Cls: {vehicle['classification_confidence']:.2f}"
                cv2.putText(debug_img, make_model, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.putText(debug_img, confidence, (x1, y1-30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            # Save the debug image
            save_debug_image(
                image=debug_img,
                debug_dir=self.debug_images_dir,
                prefix="vehicle_result",
                suffix=f"detected_{len(vehicle_results)}",
                draw_objects=None,
                draw_type=None
            )
                
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


class VehicleDetectorYOLO(YOLOBase):
    """Vehicle detector using YOLOv8."""
    
    def __init__(self, model_path: str, task: str, use_onnx: bool, use_cuda: bool, 
                 resolution: Tuple[int, int], confidence: float,
                 save_debug_images: bool = False, debug_images_dir: str = None):
        """
        Initialize the vehicle detector.
        
        Args:
            model_path: Path to the model file
            task: Task type ('detect')
            use_onnx: Whether to use ONNX model
            use_cuda: Whether to use CUDA
            resolution: Input resolution for the model
            confidence: Confidence threshold
            save_debug_images: Whether to save debug images
            debug_images_dir: Directory for debug images
        """
        super().__init__(model_path, task, use_onnx, use_cuda)
        self.resolution = resolution
        self.confidence_threshold = confidence
        self.save_debug_images = save_debug_images
        self.debug_images_dir = debug_images_dir
    
    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect vehicles in the image.
        
        Args:
            image: Input image
            
        Returns:
            List of vehicle detections
        """
        # Resize image for vehicle detector
        img_resized = cv2.resize(image, self.resolution)
        
        # Save resized input image if debug is enabled
        if self.save_debug_images:
            save_debug_image(
                image=img_resized,
                debug_dir=self.debug_images_dir,
                prefix="vehicle_detector",
                suffix="resized_input",
                draw_objects=None,
                draw_type=None
            )
        
        try:
            if self.use_onnx:
                return self._detect_onnx(image, img_resized)
            else:
                return self._detect_pytorch(image, img_resized)
        except Exception as e:
            raise InferenceError("vehicle_detector", e)
    
    def _detect_pytorch(self, original_image: np.ndarray, resized_image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect vehicles using PyTorch model"""
        # Run vehicle detection model
        results = self.model(
            resized_image, 
            conf=self.confidence_threshold, 
            verbose=False
        )[0]
        
        # Save model output visualization if debug is enabled
        if self.save_debug_images and hasattr(results, 'plot'):
            try:
                # Plot the results using the model's built-in plotting
                plot_img = results.plot()
                save_debug_image(
                    image=plot_img,
                    debug_dir=self.debug_images_dir,
                    prefix="vehicle_detector",
                    suffix="model_output",
                    draw_objects=None,
                    draw_type=None
                )
            except Exception as e:
                print(f"Error plotting vehicle detection results: {e}")
        
        # Process the results to extract vehicle bounding boxes
        vehicles = []
        
        if hasattr(results, 'boxes') and hasattr(results.boxes, 'xyxy'):
            for i, box in enumerate(results.boxes.xyxy):
                x1, y1, x2, y2 = box.cpu().numpy()
                
                # Scale the coordinates back to the original image size
                h, w = original_image.shape[:2]
                scale_x = w / self.resolution[0]
                scale_y = h / self.resolution[1]
                
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
                vehicle_img = original_image[y1:y2, x1:x2]
                
                # Save individual vehicle crop if debug is enabled
                if self.save_debug_images:
                    save_debug_image(
                        image=vehicle_img,
                        debug_dir=self.debug_images_dir,
                        prefix="vehicle_crop",
                        suffix=f"vehicle_{i}",
                        draw_objects=None,
                        draw_type=None
                    )
                
                vehicles.append({
                    "box": [x1, y1, x2, y2],
                    "confidence": confidence,
                    "image": vehicle_img
                })
        
        return vehicles
    
    def _detect_onnx(self, original_image: np.ndarray, resized_image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect vehicles using ONNX model"""
        # Preprocess image for ONNX
        input_tensor = self._preprocess_image(resized_image, self.resolution)
        
        # Run inference
        outputs = self.onnx_session.run(
            self.output_names, 
            {self.input_name: input_tensor}
        )
        
        # Parse ONNX outputs
        # For detection models: outputs typically include boxes, scores, classes
        boxes = outputs[0]
        scores = outputs[1] if len(outputs) > 1 else None
        
        # Process detections
        vehicles = []
        
        if boxes is not None:
            h, w = original_image.shape[:2]
            
            for i in range(len(boxes)):
                # Skip detections below confidence threshold
                if scores is not None and scores[i] < self.confidence_threshold:
                    continue
                
                # Get box coordinates
                x1, y1, x2, y2 = boxes[i]
                
                # Scale coordinates to original image
                x1 = int(x1 * w / self.resolution[0])
                y1 = int(y1 * h / self.resolution[1])
                x2 = int(x2 * w / self.resolution[0])
                y2 = int(y2 * h / self.resolution[1])
                
                # Ensure coordinates are within image bounds
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)
                
                # Skip invalid boxes
                if x1 >= x2 or y1 >= y2:
                    continue
                
                # Extract vehicle image
                vehicle_img = original_image[y1:y2, x1:x2]
                
                # Save individual vehicle crop if debug is enabled
                if self.save_debug_images:
                    save_debug_image(
                        image=vehicle_img,
                        debug_dir=self.debug_images_dir,
                        prefix="vehicle_crop",
                        suffix=f"vehicle_{i}_onnx",
                        draw_objects=None,
                        draw_type=None
                    )
                
                # Store detection
                vehicles.append({
                    "box": [x1, y1, x2, y2],
                    "confidence": float(scores[i]) if scores is not None else 0.0,
                    "image": vehicle_img
                })
        
        return vehicles


class VehicleClassifierYOLO(YOLOBase):
    """Vehicle classifier using YOLOv8."""
    
    def __init__(self, model_path: str, task: str, use_onnx: bool, use_cuda: bool, 
                 resolution: Tuple[int, int], confidence: float,
                 save_debug_images: bool = False, debug_images_dir: str = None):
        """
        Initialize the vehicle classifier.
        
        Args:
            model_path: Path to the model file
            task: Task type ('classify')
            use_onnx: Whether to use ONNX model
            use_cuda: Whether to use CUDA
            resolution: Input resolution for the model
            confidence: Confidence threshold
            save_debug_images: Whether to save debug images
            debug_images_dir: Directory for debug images
        """
        super().__init__(model_path, task, use_onnx, use_cuda)
        self.resolution = resolution
        self.confidence_threshold = confidence
        self.save_debug_images = save_debug_images
        self.debug_images_dir = debug_images_dir
    
    def classify(self, vehicle_image: np.ndarray) -> Dict[str, Any]:
        """
        Classify vehicle make and model.
        
        Args:
            vehicle_image: Vehicle image
            
        Returns:
            Dictionary with make, model, and confidence
        """
        if vehicle_image.size == 0:
            return {"make": "Unknown", "model": "Unknown", "confidence": 0.0}
        
        # Resize vehicle image for classifier
        try:
            vehicle_resized = cv2.resize(vehicle_image, self.resolution)
            
            # Save resized vehicle image if debug is enabled
            if self.save_debug_images:
                save_debug_image(
                    image=vehicle_resized,
                    debug_dir=self.debug_images_dir,
                    prefix="vehicle_classifier",
                    suffix="resized_input",
                    draw_objects=None,
                    draw_type=None
                )
        except Exception as e:
            raise VehicleDetectionError(f"Failed to resize vehicle image: {str(e)}")
        
        try:
            if self.use_onnx:
                result = self._classify_onnx(vehicle_resized)
            else:
                result = self._classify_pytorch(vehicle_resized)
                
            # Save classification result visualization if debug is enabled
            if self.save_debug_images:
                # Create a visualization of the classification result
                result_img = vehicle_resized.copy()
                h, w = result_img.shape[:2]
                
                # Add a label at the bottom
                label_bg = np.zeros((80, w, 3), dtype=np.uint8)
                result_img = np.vstack([result_img, label_bg])
                
                # Draw the make and model
                make_model = f"{result['make']} {result['model']}"
                confidence = f"Confidence: {result['confidence']:.2f}"
                
                cv2.putText(result_img, make_model, (10, h+30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(result_img, confidence, (10, h+60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                
                save_debug_image(
                    image=result_img,
                    debug_dir=self.debug_images_dir,
                    prefix="vehicle_classifier",
                    suffix=f"result_{result['make']}_{result['model']}",
                    draw_objects=None,
                    draw_type=None
                )
                
            return result
        except Exception as e:
            raise InferenceError("vehicle_classifier", e)
    
    def _classify_pytorch(self, vehicle_image: np.ndarray) -> Dict[str, Any]:
        """Classify vehicle using PyTorch model"""
        # Run vehicle classification model
        results = self.model(
            vehicle_image, 
            conf=self.confidence_threshold, 
            verbose=False
        )[0]
        
        # Get the predicted class and confidence
        if hasattr(results, 'probs') and hasattr(results.probs, 'top1'):
            vehicle_idx = int(results.probs.top1)
            confidence = float(results.probs.top1conf.item())
            
            # Convert class index to make and model
            vehicle_names = self.model.names
            make_model = vehicle_names[vehicle_idx]
            
            # Split make and model (assuming format "Make_Model")
            make, model = make_model.split("_", 1) if "_" in make_model else (make_model, "Unknown")
            
            return {"make": make, "model": model, "confidence": confidence}
        
        return {"make": "Unknown", "model": "Unknown", "confidence": 0.0}
    
    def _classify_onnx(self, vehicle_image: np.ndarray) -> Dict[str, Any]:
        """Classify vehicle using ONNX model"""
        # Preprocess image for ONNX
        input_tensor = self._preprocess_image(vehicle_image, self.resolution)
        
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
            vehicle_idx = np.argmax(probs[0])
            confidence = float(probs[0][vehicle_idx])
            
            # Get vehicle name from class index
            vehicle_name = self.names.get(str(vehicle_idx), self.names.get(vehicle_idx, "Unknown"))
            
            # Split make and model (assuming format "Make_Model")
            make, model = vehicle_name.split("_", 1) if "_" in vehicle_name else (vehicle_name, "Unknown")
            
            return {"make": make, "model": model, "confidence": confidence}
        
        return {"make": "Unknown", "model": "Unknown", "confidence": 0.0}
