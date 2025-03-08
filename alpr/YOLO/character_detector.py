"""
Character detection and recognition for license plates.
"""
import cv2
import numpy as np
import torch
from typing import List, Dict, Any, Tuple, Optional
from ultralytics import YOLO

from .base import YOLOBase
from ..config import ALPRConfig
from ..exceptions import ModelLoadingError, InferenceError, CharacterRecognitionError


class CharacterDetector:
    """
    Character detector for license plates using YOLOv8.
    Detects individual characters on license plates.
    """
    
    def __init__(self, config: ALPRConfig):
        """
        Initialize the character detector.
        
        Args:
            config: ALPR configuration object
            
        Raises:
            ModelLoadingError: If model loading fails
        """
        self.config = config
        self.char_detector_path = config.get_model_path("char_detector")
        self.char_classifier_path = config.get_model_path("char_classifier")
        self.char_detector_confidence = config.char_detector_confidence
        self.char_classifier_confidence = config.char_classifier_confidence
        
        # Model resolutions
        self.char_detector_resolution = (160, 160)
        self.char_classifier_resolution = (32, 32)
        
        # Initialize the detector model
        try:
            self.char_detector = CharDetector(
                model_path=self.char_detector_path,
                task='detect',
                use_onnx=config.use_onnx,
                use_cuda=config.use_cuda,
                resolution=self.char_detector_resolution,
                confidence=self.char_detector_confidence
            )
        except Exception as e:
            raise ModelLoadingError(self.char_detector_path, e)
            
        # Initialize the classifier model
        try:
            self.char_classifier = CharClassifier(
                model_path=self.char_classifier_path,
                task='classify',
                use_onnx=config.use_onnx,
                use_cuda=config.use_cuda,
                resolution=self.char_classifier_resolution,
                confidence=self.char_classifier_confidence
            )
        except Exception as e:
            raise ModelLoadingError(self.char_classifier_path, e)
    
    def detect_characters(self, plate_image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect characters in the license plate image.
        
        Args:
            plate_image: License plate image as numpy array
            
        Returns:
            List of dictionaries with character box coordinates and images
            
        Raises:
            InferenceError: If detection fails
        """
        return self.char_detector.detect(plate_image)
    
    def organize_characters(self, characters: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Organize characters into a coherent structure, handling multiple lines and vertical characters.
        
        Args:
            characters: List of character dictionaries from detect_characters()
            
        Returns:
            List of characters in reading order
        """
        if not characters:
            return []
        
        # Extract bounding box coordinates
        boxes = np.array([[c["box"][0], c["box"][1], c["box"][2], c["box"][3]] for c in characters])
        
        # Calculate center points of all boxes
        centers = np.array([[(box[0] + box[2]) / 2, (box[1] + box[3]) / 2] for box in boxes])
        
        # Calculate heights and widths
        heights = boxes[:, 3] - boxes[:, 1]
        widths = boxes[:, 2] - boxes[:, 0]
        
        # Determine if there are multiple lines
        # We'll use a simple heuristic: if there are centers with y-coordinates that differ by more than
        # the average character height, then we have multiple lines
        avg_height = np.mean(heights)
        y_diffs = np.abs(centers[:, 1][:, np.newaxis] - centers[:, 1])
        multiple_lines = np.any(y_diffs > 1.5 * avg_height)
        
        # Determine if there are vertical characters
        # Vertical characters typically have height > width
        aspect_ratios = heights / widths
        vertical_chars = aspect_ratios > 1.5  # Characters with aspect ratio > 1.5 are considered vertical
        
        organized_chars = []
        
        if multiple_lines:
            # Cluster characters by y-coordinate (row)
            # Using a simple approach: characters with similar y-center are on the same line
            y_centers = centers[:, 1]
            sorted_indices = np.argsort(y_centers)
            
            # Group characters by line
            lines = []
            current_line = [sorted_indices[0]]
            
            for i in range(1, len(sorted_indices)):
                idx = sorted_indices[i]
                prev_idx = sorted_indices[i-1]
                
                if abs(y_centers[idx] - y_centers[prev_idx]) > 0.5 * avg_height:
                    # Start a new line
                    lines.append(current_line)
                    current_line = [idx]
                else:
                    # Continue the current line
                    current_line.append(idx)
            
            lines.append(current_line)
            
            # Sort characters within each line by x-coordinate (left to right)
            for line in lines:
                line_chars = [characters[idx] for idx in sorted(line, key=lambda idx: centers[idx][0])]
                organized_chars.extend(line_chars)
        else:
            # Single line with possible vertical characters at the beginning or end
            
            # Check for vertical characters at the beginning
            start_vertical_indices = []
            for i, is_vertical in enumerate(vertical_chars):
                if is_vertical and centers[i][0] < np.median(centers[:, 0]):
                    start_vertical_indices.append(i)
            
            # Check for vertical characters at the end
            end_vertical_indices = []
            for i, is_vertical in enumerate(vertical_chars):
                if is_vertical and centers[i][0] > np.median(centers[:, 0]):
                    end_vertical_indices.append(i)
            
            # Remaining horizontal characters
            horizontal_indices = [i for i in range(len(characters)) 
                                if i not in start_vertical_indices and i not in end_vertical_indices]
            
            # Sort vertical characters at the beginning by x-coordinate
            start_vertical_indices.sort(key=lambda idx: centers[idx][0])
            
            # Sort horizontal characters by x-coordinate
            horizontal_indices.sort(key=lambda idx: centers[idx][0])
            
            # Sort vertical characters at the end by x-coordinate
            end_vertical_indices.sort(key=lambda idx: centers[idx][0])
            
            # Combine all indices in the correct order
            all_indices = start_vertical_indices + horizontal_indices + end_vertical_indices
            organized_chars = [characters[idx] for idx in all_indices]
        
        return organized_chars
    
    def classify_character(self, char_image: np.ndarray) -> List[Tuple[str, float]]:
        """
        Classify a character using OCR and return top predictions.
        
        Args:
            char_image: Character image to classify
            
        Returns:
            List of tuples containing (character, confidence) for top predictions
            
        Raises:
            InferenceError: If classification fails
            CharacterRecognitionError: If character image is invalid
        """
        return self.char_classifier.classify(char_image)
    
    def process_plate(self, plate_image: np.ndarray) -> Dict[str, Any]:
        """
        Detect and recognize characters on a license plate.
        
        Args:
            plate_image: License plate image
            
        Returns:
            Dictionary with character detections and plate number
        """
        # 1. Detect characters
        characters = self.detect_characters(plate_image)
        
        # 2. Organize characters (handle multiple lines and vertical characters)
        organized_chars = self.organize_characters(characters)
        
        # 3. Classify each character
        char_results = []
        for char_info in organized_chars:
            top_chars = self.classify_character(char_info["image"])
            char_results.append({
                "char": top_chars[0][0] if top_chars else "?",  # Use the top prediction as the main character
                "confidence": top_chars[0][1] if top_chars else 0.0,
                "top_predictions": top_chars,  # Store all top predictions
                "box": char_info["box"]
            })
        
        # 4. Construct the license number by concatenating the characters
        license_number = ''.join(cr["char"] for cr in char_results)
        avg_confidence = sum(cr["confidence"] for cr in char_results) / len(char_results) if char_results else 0.0
        
        # 5. Generate alternative plate combinations
        top_plates = self._generate_top_plates(char_results)
        
        return {
            "characters": char_results,
            "license_number": license_number,
            "confidence": avg_confidence,
            "top_plates": top_plates
        }
    
    def _generate_top_plates(self, 
                            char_results: List[Dict[str, Any]], 
                            max_combinations: int = 5) -> List[Dict[str, Any]]:
        """
        Generate multiple possible license plate combinations using top character predictions.
        
        Args:
            char_results: List of character results with top_predictions
            max_combinations: Maximum number of combinations to return
            
        Returns:
            List of alternative plate combinations with plate number and confidence
        """
        if not char_results:
            return []
        
        # Identify positions with uncertain character predictions
        uncertain_positions = []
        for i, char_result in enumerate(char_results):
            top_preds = char_result.get("top_predictions", [])
            
            # If we have at least 2 predictions with good confidence
            if len(top_preds) >= 2 and top_preds[1][1] >= 0.01:
                confidence_diff = top_preds[0][1] - top_preds[1][1]
                uncertain_positions.append((i, confidence_diff))
        
        # Sort by smallest confidence difference (most uncertain first)
        uncertain_positions.sort(key=lambda x: x[1])
        
        # Create base plate using top1 predictions
        base_plate = ''.join(cr["char"] for cr in char_results)
        base_confidence = sum(cr["confidence"] for cr in char_results) / len(char_results) if char_results else 0.0
        
        # Start with the base plate
        combinations = [{"plate": base_plate, "confidence": base_confidence}]
        
        # Generate alternative plates by substituting at uncertain positions
        for pos_idx, _ in uncertain_positions[:min(3, len(uncertain_positions))]:
            char_result = char_results[pos_idx]
            top_preds = char_result.get("top_predictions", [])[1:3]  # Use 2nd and 3rd predictions
            
            # Generate new plates by substituting at this position
            new_combinations = []
            for existing in combinations:
                for alt_char, alt_conf in top_preds:
                    if alt_conf >= 0.02:
                        plate_chars = list(existing["plate"])
                        if pos_idx < len(plate_chars):
                            # Calculate new confidence
                            old_char_conf = char_results[pos_idx]["confidence"]
                            plate_chars[pos_idx] = alt_char
                            
                            # Adjust confidence by replacing the character's contribution
                            char_count = len(char_results)
                            new_conf = existing["confidence"] - (old_char_conf / char_count) + (alt_conf / char_count)
                            
                            new_plate = ''.join(plate_chars)
                            new_combinations.append({"plate": new_plate, "confidence": new_conf})
                
            combinations.extend(new_combinations)
            
            # If we have enough combinations, stop
            if len(combinations) >= max_combinations:
                break
        
        # Sort by confidence and take top N
        combinations.sort(key=lambda x: x["confidence"], reverse=True)
        return combinations[:max_combinations]


class CharDetector(YOLOBase):
    """Character detector for license plates using YOLOv8 or ONNX."""
    
    def __init__(self, model_path: str, task: str, use_onnx: bool, use_cuda: bool, 
                 resolution: Tuple[int, int], confidence: float):
        """
        Initialize the character detector.
        
        Args:
            model_path: Path to the model file
            task: Task type
            use_onnx: Whether to use ONNX
            use_cuda: Whether to use CUDA
            resolution: Input resolution for the model
            confidence: Confidence threshold
        """
        super().__init__(model_path, task, use_onnx, use_cuda)
        self.resolution = resolution
        self.confidence_threshold = confidence
    
    def detect(self, plate_image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect characters in the license plate image.
        
        Args:
            plate_image: License plate image
            
        Returns:
            List of character detections
        """
        # Resize plate image for detector
        plate_resized = cv2.resize(plate_image, self.resolution)
        
        try:
            if self.use_onnx:
                return self._detect_onnx(plate_image, plate_resized)
            else:
                return self._detect_pytorch(plate_image, plate_resized)
        except Exception as e:
            raise InferenceError("char_detector", e)
    
    def _detect_pytorch(self, original_image: np.ndarray, resized_image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect characters using PyTorch model"""
        # Run character detection model
        results = self.model.predict(
            resized_image, 
            conf=self.confidence_threshold, 
            verbose=False
        )[0]
        
        # Process the results to extract character bounding boxes
        characters = []
        
        if hasattr(results, 'boxes') and hasattr(results.boxes, 'xyxy'):
            for i, box in enumerate(results.boxes.xyxy):
                x1, y1, x2, y2 = box.cpu().numpy()
                
                # Scale the coordinates back to the original plate image size
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
                
                # Extract the character region
                char_img = original_image[y1:y2, x1:x2]
                
                characters.append({
                    "box": [x1, y1, x2, y2],
                    "confidence": confidence,
                    "image": char_img
                })
        
        return characters
    
    def _detect_onnx(self, original_image: np.ndarray, resized_image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect characters using ONNX model"""
        # Preprocess image for ONNX
        input_tensor = self._preprocess_image(resized_image, self.resolution)
        
        # Run inference
        outputs = self.onnx_session.run(
            self.output_names, 
            {self.input_name: input_tensor}
        )
        
        # Parse ONNX outputs (format depends on your exported model)
        # This implementation assumes a specific output format
        # For detection models: outputs typically include boxes, scores, classes
        boxes = outputs[0]
        scores = outputs[1] if len(outputs) > 1 else None
        
        # Process detections
        characters = []
        
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
                
                # Extract character image
                char_img = original_image[y1:y2, x1:x2]
                
                # Store detection
                characters.append({
                    "box": [x1, y1, x2, y2],
                    "confidence": float(scores[i]) if scores is not None else 0.0,
                    "image": char_img
                })
        
        return characters


class CharClassifier(YOLOBase):
    """Character classifier for license plates using YOLOv8 or ONNX."""
    
    def __init__(self, model_path: str, task: str, use_onnx: bool, use_cuda: bool, 
                 resolution: Tuple[int, int], confidence: float):
        """
        Initialize the character classifier.
        
        Args:
            model_path: Path to the model file
            task: Task type
            use_onnx: Whether to use ONNX
            use_cuda: Whether to use CUDA
            resolution: Input resolution for the model
            confidence: Confidence threshold
        """
        super().__init__(model_path, task, use_onnx, use_cuda)
        self.resolution = resolution
        self.confidence_threshold = confidence
    
    def classify(self, char_image: np.ndarray) -> List[Tuple[str, float]]:
        """
        Classify a character and return top predictions.
        
        Args:
            char_image: Character image
            
        Returns:
            List of (character, confidence) tuples
        """
        if char_image.size == 0:
            return [("?", 0.0)]
        
        # Resize character image for classifier
        try:
            char_resized = cv2.resize(char_image, self.resolution)
        except Exception as e:
            raise CharacterRecognitionError(f"Failed to resize character image: {str(e)}")
        
        try:
            if self.use_onnx:
                return self._classify_onnx(char_resized)
            else:
                return self._classify_pytorch(char_resized)
        except Exception as e:
            raise InferenceError("char_classifier", e)
    
    def _classify_pytorch(self, char_image: np.ndarray) -> List[Tuple[str, float]]:
        """Classify character using PyTorch model"""
        # Run character classification model
        results = self.model.predict(
            char_image, 
            conf=self.confidence_threshold, 
            verbose=False
        )[0]
        
        top_predictions = []
        
        # Extract top5 predictions if available
        if hasattr(results, 'probs'):
            probs = results.probs
            
            # Try to access probability data
            if hasattr(probs, 'data'):
                try:
                    # Convert to tensor if it's not already
                    probs_tensor = probs.data
                    if not isinstance(probs_tensor, torch.Tensor):
                        probs_tensor = torch.tensor(probs_tensor)
                    
                    # Get top 5 predictions
                    values, indices = torch.topk(probs_tensor, min(5, len(probs_tensor)))
                    
                    # Convert to list of (char, confidence) tuples
                    char_names = self.model.names
                    for i in range(len(values)):
                        idx = int(indices[i].item())
                        conf = float(values[i].item())
                        
                        # Only include predictions with confidence > 0.02
                        if conf >= 0.02:
                            character = char_names[idx]
                            top_predictions.append((character, conf))
                except Exception as e:
                    # Fallback to top1 if available
                    if hasattr(probs, 'top1') and hasattr(probs, 'top1conf'):
                        idx = int(probs.top1)
                        conf = float(probs.top1conf.item())
                        
                        char_names = self.model.names
                        character = char_names[idx]
                        top_predictions.append((character, conf))
            elif hasattr(probs, 'top1') and hasattr(probs, 'top1conf'):
                # Fallback to top1 if data attribute not available
                idx = int(probs.top1)
                conf = float(probs.top1conf.item())
                
                char_names = self.model.names
                character = char_names[idx]
                top_predictions.append((character, conf))
        
        # If no predictions were found or all had low confidence
        if not top_predictions:
            top_predictions.append(("?", 0.0))
        
        return top_predictions
    
    def _classify_onnx(self, char_image: np.ndarray) -> List[Tuple[str, float]]:
        """Classify character using ONNX model"""
        # Preprocess image for ONNX
        input_tensor = self._preprocess_image(char_image, self.resolution)
        
        # Run inference
        outputs = self.onnx_session.run(
            self.output_names, 
            {self.input_name: input_tensor}
        )
        
        # Parse ONNX outputs
        # For classification, output is typically class probabilities
        probs = outputs[0]
        
        # Get top 5 predictions
        top_predictions = []
        
        if probs is not None:
            # Get indices of top 5 probabilities
            top_indices = np.argsort(probs[0])[::-1][:5]
            
            # Create list of (character, confidence) tuples
            for idx in top_indices:
                conf = float(probs[0][idx])
                
                # Only include predictions with confidence > 0.02
                if conf >= 0.02:
                    character = self.names.get(str(idx), self.names.get(idx, "?"))
                    top_predictions.append((character, conf))
        
        # If no predictions were found or all had low confidence
        if not top_predictions:
            top_predictions.append(("?", 0.0))
        
        return top_predictions
