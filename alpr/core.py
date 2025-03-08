"""
Core ALPR system that coordinates the detection and recognition pipeline.
"""
import cv2
import numpy as np
import time
import concurrent.futures
from typing import Dict, List, Any, Optional, Tuple

from .config import ALPRConfig
from .exceptions import ALPRException, ModelLoadingError
from .YOLO.plate_detector import PlateDetector
from .YOLO.character_detector import CharacterDetector
from .YOLO.state_classifier import StateClassifier
from .YOLO.vehicle_detector import VehicleDetector
from .utils.image_processing import four_point_transform


class ALPRSystem:
    """
    Automatic License Plate Recognition system.
    Coordinates the detection and recognition pipeline.
    """
    
    def __init__(self, config: ALPRConfig):
        """
        Initialize the ALPR system.
        
        Args:
            config: Configuration for the ALPR system
            
        Raises:
            ModelLoadingError: If any model fails to load
        """
        self.config = config
        
        # Initialize detector components
        self.plate_detector = PlateDetector(config)
        self.character_detector = CharacterDetector(config)
        
        # Initialize optional components based on configuration
        self.state_classifier = None
        if config.enable_state_detection:
            self.state_classifier = StateClassifier(config)
            
        self.vehicle_detector = None
        if config.enable_vehicle_detection:
            self.vehicle_detector = VehicleDetector(config)
    
    def detect_license_plates(self, image: np.ndarray) -> Dict[str, List[Dict[str, Any]]]:
        """
        Detect license plates in the image.
        
        Args:
            image: Input image as numpy array (BGR format)
            
        Returns:
            Dictionary with 'day_plates' and 'night_plates' lists
        """
        return self.plate_detector.detect(image)
    
    def process_plate(self, 
                     image: np.ndarray, 
                     plate_info: Dict[str, Any], 
                     is_day_plate: bool) -> Dict[str, Any]:
        """
        Process a single license plate.
        
        Args:
            image: Original image
            plate_info: Plate detection information
            is_day_plate: Whether it's a day plate or night plate
            
        Returns:
            Dictionary with plate processing results
        """
        # Extract corners
        plate_corners = plate_info["corners"]
        
        # Crop the license plate using 4-point transform
        plate_image = four_point_transform(image, plate_corners, self.config.plate_aspect_ratio)
        
        # Initialize result dictionary
        plate_result = {
            "corners": plate_corners,
            "is_day_plate": is_day_plate,
            "plate": "",
            "confidence": 0.0,
            "aspect_ratio": self.config.plate_aspect_ratio  # Store used aspect ratio for reference
        }
        
        # Include original corners if available
        if "original_corners" in plate_info:
            plate_result["original_corners"] = plate_info["original_corners"]
        
        # Include detection box if available
        if "detection_box" in plate_info:
            plate_result["detection_box"] = plate_info["detection_box"]
        
        # If it's a day plate, also determine the state
        if is_day_plate and self.state_classifier:
            state_result = self.state_classifier.classify(plate_image)
            plate_result["state"] = state_result["state"]
            plate_result["state_confidence"] = state_result["confidence"]
        
        # Detect and recognize characters in the plate
        char_result = self.character_detector.process_plate(plate_image)
        
        # Update plate result with character recognition data
        plate_result.update({
            "characters": char_result["characters"],
            "license_number": char_result["license_number"],
            "confidence": char_result["confidence"],
            "top_plates": char_result["top_plates"]
        })
        
        # Store plate dimensions for debugging
        if plate_image is not None:
            h, w = plate_image.shape[:2]
            plate_result["dimensions"] = {"width": w, "height": h, "actual_ratio": w/h if h > 0 else 0}
        
        return plate_result
    
    def process_image(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Process an image to detect and recognize license plates, vehicle make/model.
        
        Args:
            image: Input image
            
        Returns:
            Dictionary with processing results
        """
        # Create a copy of the image to avoid modifying the original
        image_copy = image.copy()
        
        start_time = time.perf_counter()
        
        # Detect license plates in the image
        plate_detection = self.detect_license_plates(image_copy)
        
        # Initialize results
        results = {
            "day_plates": [],
            "night_plates": [],
            "vehicles": []
        }
        
        # Process plates using multi-threading
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Submit day plate processing tasks
            day_plate_futures = [
                executor.submit(self.process_plate, image_copy, plate, True)
                for plate in plate_detection["day_plates"]
            ]
            
            # Submit night plate processing tasks
            night_plate_futures = [
                executor.submit(self.process_plate, image_copy, plate, False)
                for plate in plate_detection["night_plates"]
            ]
            
            # Collect day plate results
            for future in concurrent.futures.as_completed(day_plate_futures):
                try:
                    results["day_plates"].append(future.result())
                except Exception as e:
                    print(f"Error processing day plate: {e}")
            
            # Collect night plate results
            for future in concurrent.futures.as_completed(night_plate_futures):
                try:
                    results["night_plates"].append(future.result())
                except Exception as e:
                    print(f"Error processing night plate: {e}")
        
        # If day plates were detected and vehicle detection is enabled,
        # detect and classify vehicles
        if results["day_plates"] and self.vehicle_detector:
            vehicle_results = self.vehicle_detector.detect_and_classify(image_copy)
            results["vehicles"] = vehicle_results
        
        # Add timing information
        results["processing_time_ms"] = int((time.perf_counter() - start_time) * 1000)
        
        return results
    
    def detect_license_plate(self, 
                            image: np.ndarray, 
                            threshold: float = 0.4) -> Dict[str, Any]:
        """
        Detect license plates and prepare response for the API.
        
        Args:
            image: Input image as PIL Image
            threshold: Minimum confidence threshold
            
        Returns:
            API response with predictions
        """
        start_process_time = time.perf_counter()
        
        # Convert PIL Image to numpy array for OpenCV
        image_np = np.array(image)
        # Convert RGB to BGR (OpenCV format)
        if len(image_np.shape) == 3 and image_np.shape[2] == 3:  # Color image
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        
        # Process the image
        start_inference_time = time.perf_counter()
        
        # Process the image to find license plates
        plate_detection = self.detect_license_plates(image_np)
        
        # Process each plate
        results = {
            "day_plates": [],
            "night_plates": []
        }
        
        for plate_type in ["day_plates", "night_plates"]:
            for plate_info in plate_detection[plate_type]:
                if plate_info["confidence"] >= threshold:
                    plate_result = self.process_plate(image_np, plate_info, plate_type == "day_plates")
                    if plate_result["confidence"] >= threshold:
                        results[plate_type].append(plate_result)
        
        inferenceMs = int((time.perf_counter() - start_inference_time) * 1000)
        
        # Extract plate numbers and coordinates for client response
        plates = []
        for plate_type in ["day_plates", "night_plates"]:
            for plate in results[plate_type]:
                # Only include plates with confidence above threshold
                if plate["confidence"] >= threshold:
                    # Use detection_box if available, otherwise calculate from corners
                    if "detection_box" in plate and plate["detection_box"] is not None:
                        # If detection_box is available, use it directly
                        x1, y1, x2, y2 = plate["detection_box"]
                        plate_data = {
                            "confidence": plate["confidence"],
                            "is_day_plate": plate["is_day_plate"],
                            "label": plate["license_number"],
                            "plate": plate["license_number"],
                            "x_min": x1,
                            "y_min": y1,
                            "x_max": x2,
                            "y_max": y2
                        }
                    else:
                        # Otherwise, calculate the bounding box from the corners
                        corners = plate["corners"]
                        # Convert corners to numpy array if not already
                        corners_arr = np.array(corners)
                        x_min = np.min(corners_arr[:, 0])
                        y_min = np.min(corners_arr[:, 1])
                        x_max = np.max(corners_arr[:, 0])
                        y_max = np.max(corners_arr[:, 1])
                        
                        plate_data = {
                            "confidence": plate["confidence"],
                            "is_day_plate": plate["is_day_plate"],
                            "label": plate["license_number"],
                            "plate": plate["license_number"],
                            "x_min": float(x_min),
                            "y_min": float(y_min),
                            "x_max": float(x_max),
                            "y_max": float(y_max)
                        }
                    
                    if "state" in plate:
                        plate_data["state"] = plate["state"]
                        plate_data["state_confidence"] = plate["state_confidence"]
                    
                    # Add top plate alternatives
                    if "top_plates" in plate:
                        plate_data["top_plates"] = plate["top_plates"]
                        
                    plates.append(plate_data)
        
        # Create a response message
        if len(plates) > 0:
            message = f"Found {len(plates)} license plates"
            if len(plates) <= 3:
                message += ": " + ", ".join([p["label"] for p in plates])
        else:
            message = "No license plates detected"
            
        return {
            "success": True,
            "processMs": int((time.perf_counter() - start_process_time) * 1000),
            "inferenceMs": inferenceMs,
            "predictions": plates,
            "message": message,
            "count": len(plates)
        }
