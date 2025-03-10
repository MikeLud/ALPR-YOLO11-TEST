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
from ..utils.image_processing import save_debug_image


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
        self.save_debug_images = config.save_debug_images
        self.debug_images_dir = config.debug_images_dir
        
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
                confidence=self.char_detector_confidence,
                save_debug_images=self.save_debug_images,
                debug_images_dir=self.debug_images_dir
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
                confidence=self.char_classifier_confidence,
                save_debug_images=self.save_debug_images,
                debug_images_dir=self.debug_images_dir
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
        
        Implements a robust algorithm for ordering characters left-to-right and top-to-bottom,
        with special handling for various edge cases.
        
        Args:
            characters: List of character dictionaries from detect_characters()
            
        Returns:
            List of characters in reading order
        """
        # Define constants for better readability and easier tuning
        LINE_SEPARATION_THRESHOLD = 0.6   # Multiple of avg height to consider separate lines
        VERTICAL_ASPECT_RATIO = 1.5       # Aspect ratio threshold for vertical characters
        OVERLAP_THRESHOLD = 0.3           # Threshold for determining overlapping characters
        MIN_CHARS_FOR_CLUSTERING = 6      # Minimum chars before using advanced clustering
        
        if not characters:
            return []
        
        try:
            # Extract bounding box coordinates with error handling
            valid_chars = []
            boxes = []
            for i, char in enumerate(characters):
                box = char.get("box", None)
                if box and len(box) == 4 and box[2] > box[0] and box[3] > box[1]:
                    valid_chars.append(char)
                    boxes.append([box[0], box[1], box[2], box[3]])
            
            if not valid_chars:
                return characters  # Return original if no valid characters
            
            boxes = np.array(boxes)
            
            # Calculate important metrics
            centers = np.array([[(box[2] + box[0]) / 2, (box[3] + box[1]) / 2] for box in boxes])
            heights = boxes[:, 3] - boxes[:, 1]
            widths = boxes[:, 2] - boxes[:, 0]
            
            # Special case: only one character
            if len(valid_chars) == 1:
                return valid_chars
                
            # Special case: only two characters - simple left-to-right ordering
            if len(valid_chars) == 2:
                if centers[0][0] <= centers[1][0]:
                    return [valid_chars[0], valid_chars[1]]
                else:
                    return [valid_chars[1], valid_chars[0]]
            
            # Determine if there are multiple lines using adaptive thresholding
            # Calculate median of heights for robustness against outliers
            median_height = np.median(heights)
            avg_height = np.mean(heights)
            reference_height = min(avg_height, median_height)  # Use the smaller one for conservative grouping
            
            # Use a weighted approach for line separation
            y_diffs = np.abs(centers[:, 1][:, np.newaxis] - centers[:, 1])
            
            # Check for multi-line text by analyzing y-coordinate distribution
            y_sorted = np.sort(centers[:, 1])
            y_gaps = y_sorted[1:] - y_sorted[:-1]
            if len(y_gaps) > 0:
                max_gap = np.max(y_gaps)
                is_multiline = max_gap > LINE_SEPARATION_THRESHOLD * reference_height
            else:
                is_multiline = False
                
            # Detect horizontal vs vertical characters
            epsilon = 1e-6  # Avoid division by zero
            aspect_ratios = heights / (widths + epsilon)
            vertical_chars = aspect_ratios > VERTICAL_ASPECT_RATIO
            
            # Check for overlapping characters
            has_overlaps = self._detect_overlapping_characters(boxes, OVERLAP_THRESHOLD)
                
            # Handle differently based on layout complexity
            if is_multiline:
                # Use more advanced clustering for complex layouts
                if len(valid_chars) >= MIN_CHARS_FOR_CLUSTERING:
                    organized_chars = self._cluster_and_order_characters(
                        valid_chars, centers, heights, reference_height)
                else:
                    organized_chars = self._organize_multiline_characters(
                        valid_chars, centers, reference_height)
            else:
                if has_overlaps:
                    # Special handling for overlapping characters
                    organized_chars = self._handle_overlapping_characters(
                        valid_chars, boxes, centers)
                else:
                    # Standard single-line processing
                    organized_chars = self._organize_single_line_characters(
                        valid_chars, centers, vertical_chars)
            
            # Save debug image of the organized characters if enabled
            if self.save_debug_images and organized_chars:
                self._save_character_order_debug_image(characters, organized_chars)
            
            return organized_chars
        
        except Exception as e:
            # Log error and return original characters as fallback
            if hasattr(self, 'logger'):
                self.logger.error(f"Error organizing characters: {str(e)}")
            return characters

    def _detect_overlapping_characters(self, boxes, threshold):
        """
        Detect if there are overlapping characters in the text.
        
        Args:
            boxes: Array of character bounding boxes
            threshold: Overlap threshold ratio
            
        Returns:
            Boolean indicating whether significant overlaps exist
        """
        # Check each pair of boxes for overlap
        for i in range(len(boxes)):
            for j in range(i+1, len(boxes)):
                box1 = boxes[i]
                box2 = boxes[j]
                
                # Calculate intersection area
                x_overlap = max(0, min(box1[2], box2[2]) - max(box1[0], box2[0]))
                y_overlap = max(0, min(box1[3], box2[3]) - max(box1[1], box2[1]))
                overlap_area = x_overlap * y_overlap
                
                # Calculate minimum box area
                box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
                box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
                min_area = min(box1_area, box2_area)
                
                # If overlap is significant relative to character size
                if min_area > 0 and overlap_area / min_area > threshold:
                    return True
        
        return False

    def _handle_overlapping_characters(self, characters, boxes, centers):
        """
        Special handling for overlapping characters.
        
        Args:
            characters: List of character dictionaries
            boxes: Array of character bounding boxes
            centers: Array of character center points
            
        Returns:
            List of characters organized by reading order
        """
        # For overlapping characters, prioritize x-coordinate ordering
        # but handle characters that are approximately at the same x-position
        x_sorted_indices = np.argsort(centers[:, 0])
        
        # Group characters that are very close horizontally
        groups = []
        current_group = [x_sorted_indices[0]]
        
        # Threshold for horizontal grouping (as a fraction of average width)
        avg_width = np.mean(boxes[:, 2] - boxes[:, 0])
        threshold = 0.3 * avg_width
        
        for i in range(1, len(x_sorted_indices)):
            curr_idx = x_sorted_indices[i]
            prev_idx = x_sorted_indices[i-1]
            
            # If characters are horizontally close, group them
            if abs(centers[curr_idx][0] - centers[prev_idx][0]) < threshold:
                current_group.append(curr_idx)
            else:
                # Start a new group
                groups.append(current_group)
                current_group = [curr_idx]
        
        # Add the last group
        if current_group:
            groups.append(current_group)
        
        # For each group of horizontally-aligned characters, sort by y-coordinate
        organized_indices = []
        for group in groups:
            # If group has multiple characters, sort by vertical position
            if len(group) > 1:
                sorted_group = sorted(group, key=lambda idx: centers[idx][1])
            else:
                sorted_group = group
            
            organized_indices.extend(sorted_group)
        
        return [characters[idx] for idx in organized_indices]

    def _cluster_and_order_characters(self, characters, centers, heights, reference_height):
        """
        Use advanced clustering to organize characters in complex layouts.
        
        Args:
            characters: List of character dictionaries
            centers: Array of character center points
            heights: Array of character heights
            reference_height: Reference height for line detection
            
        Returns:
            List of characters organized by reading order
        """
        # Use hierarchical clustering to group characters into lines
        # Scale y-coordinates to make clustering more sensitive to vertical differences
        scaled_centers = centers.copy()
        scaled_centers[:, 1] = scaled_centers[:, 1] * 3  # Emphasize y-coordinate
        
        from scipy.cluster.hierarchy import linkage, fcluster
        
        # Perform hierarchical clustering
        Z = linkage(scaled_centers, 'ward')
        
        # Determine optimal number of clusters (lines)
        # More sophisticated than simple threshold: analyze cluster distances
        max_dist = reference_height * 1.2
        clusters = fcluster(Z, max_dist, criterion='distance')
        
        # Group characters by cluster (line)
        line_groups = {}
        for i, cluster_id in enumerate(clusters):
            if cluster_id not in line_groups:
                line_groups[cluster_id] = []
            line_groups[cluster_id].append(i)
        
        # Sort lines by y-coordinate (top to bottom)
        sorted_lines = []
        for cluster_id, char_indices in line_groups.items():
            # Calculate average y-coordinate for the line
            avg_y = np.mean([centers[idx][1] for idx in char_indices])
            sorted_lines.append((avg_y, char_indices))
        
        sorted_lines.sort()  # Sort by y-coordinate
        
        # Process each line: sort characters from left to right
        organized_chars = []
        for _, char_indices in sorted_lines:
            # Sort characters within a line from left to right
            sorted_indices = sorted(char_indices, key=lambda idx: centers[idx][0])
            line_chars = [characters[idx] for idx in sorted_indices]
            organized_chars.extend(line_chars)
        
        return organized_chars

    def _organize_multiline_characters(self, characters, centers, reference_height):
        """
        Organize characters in multiple lines using refined algorithm.
        
        Args:
            characters: List of character dictionaries
            centers: Array of character center points
            reference_height: Reference height for line separation
            
        Returns:
            List of characters organized by reading order
        """
        # Sort characters by y-coordinate
        y_sorted_indices = np.argsort(centers[:, 1])
        
        # Group characters into lines with adaptive thresholding
        lines = []
        if len(y_sorted_indices) > 0:
            current_line = [y_sorted_indices[0]]
            current_y_avg = centers[y_sorted_indices[0]][1]
            
            for i in range(1, len(y_sorted_indices)):
                idx = y_sorted_indices[i]
                
                # Adaptive threshold based on current line's average y-value
                y_diff = abs(centers[idx][1] - current_y_avg)
                threshold = 0.6 * reference_height  # More conservative threshold
                
                if y_diff > threshold:
                    # Start a new line
                    lines.append(current_line)
                    current_line = [idx]
                    current_y_avg = centers[idx][1]
                else:
                    # Continue the current line
                    current_line.append(idx)
                    # Update rolling average of y-coordinates for current line
                    current_y_avg = (current_y_avg * len(current_line) + centers[idx][1]) / (len(current_line) + 1)
            
            lines.append(current_line)
        
        # Sort characters within each line by x-coordinate (left to right)
        organized_chars = []
        for line in lines:
            # For each line, sort characters by x-coordinate
            sorted_line = sorted(line, key=lambda idx: centers[idx][0])
            line_chars = [characters[idx] for idx in sorted_line]
            organized_chars.extend(line_chars)
        
        return organized_chars

    def _organize_single_line_characters(self, characters, centers, vertical_chars):
        """
        Organize characters in a single line, with improved handling of special cases.
        
        Args:
            characters: List of character dictionaries
            centers: Array of character center points
            vertical_chars: Boolean array indicating which characters are vertical
            
        Returns:
            List of characters organized by reading order
        """
        if len(characters) == 0:
            return []
        
        # Special case: if all characters are vertical or all are horizontal
        if np.all(vertical_chars) or not np.any(vertical_chars):
            # Simple left-to-right sorting
            return [characters[idx] for idx in np.argsort(centers[:, 0])]
        
        # For mixed vertical/horizontal characters:
        
        # First check if vertical characters are likely rotated text
        num_vertical = np.sum(vertical_chars)
        total_chars = len(characters)
        
        # If most characters are vertical (>70%), treat as rotated text
        if num_vertical / total_chars > 0.7:
            # For predominantly vertical text, sort primarily by x, then by y
            # This handles cases like vertically arranged Chinese/Japanese
            indices = np.lexsort((centers[:, 1], centers[:, 0]))
            return [characters[idx] for idx in indices]
        
        # For mixed characters with horizontal dominance, use horizontal/vertical separation
        x_median = np.median(centers[:, 0])
        
        # Categorize vertical characters as start or end based on x-position
        start_vertical = []
        end_vertical = []
        horizontal = []
        
        for i in range(len(characters)):
            if vertical_chars[i]:
                if centers[i][0] < x_median:
                    start_vertical.append(i)
                else:
                    end_vertical.append(i)
            else:
                horizontal.append(i)
        
        # Sort each group by appropriate coordinates
        start_vertical.sort(key=lambda idx: (centers[idx][0], centers[idx][1]))
        horizontal.sort(key=lambda idx: centers[idx][0])
        end_vertical.sort(key=lambda idx: (centers[idx][0], centers[idx][1]))
        
        # Combine all indices in the correct order
        all_indices = start_vertical + horizontal + end_vertical
        
        '''
        with open("log.txt", "a") as text_file:
            text_file.write(str(all_indices) + str(characters) + "\n" + "\n")
        '''
        return [characters[idx] for idx in all_indices]

    def _save_character_order_debug_image(self, characters, organized_chars):
        """
        Generate and save a debug image showing character reading order.
        
        Args:
            characters: Original character list
            organized_chars: Organized character list
        """
        try:
            # Get the original plate image from the first character
            plate_with_char_order = None
            for char in characters:
                if 'plate_image' in char:
                    plate_with_char_order = char['plate_image'].copy()
                    break
            
            if plate_with_char_order is None:
                return
            
            # Draw character boxes with order numbers
            for i, char in enumerate(organized_chars):
                if 'box' in char:
                    x1, y1, x2, y2 = char['box']
                    # Draw box
                    cv2.rectangle(plate_with_char_order, (x1, y1), (x2, y2), (0, 255, 0), 1)
                    # Draw order number
                    cv2.putText(plate_with_char_order, str(i+1), (x1, y1-5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                    
                    # Draw center point for additional debugging
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)
                    cv2.circle(plate_with_char_order, (center_x, center_y), 2, (255, 0, 0), -1)
            
            # Save the debug image
            save_debug_image(
                image=plate_with_char_order,
                debug_dir=self.debug_images_dir,
                prefix="char_organizer",
                suffix="reading_order",
                draw_objects=None,
                draw_type=None
            )
            
            # Save additional debug visualization showing line clustering
            if len(organized_chars) > 2:
                # Create a copy for line visualization
                line_visualization = plate_with_char_order.copy()
                
                # Get centers of all characters
                centers = []
                for char in organized_chars:
                    if 'box' in char:
                        x1, y1, x2, y2 = char['box']
                        centers.append((int((x1 + x2) / 2), int((y1 + y2) / 2)))
                
                # Draw lines connecting characters in order
                for i in range(len(centers) - 1):
                    cv2.line(line_visualization, centers[i], centers[i+1], (0, 0, 255), 1)
                
                # Save this visualization
                save_debug_image(
                    image=line_visualization,
                    debug_dir=self.debug_images_dir,
                    prefix="char_organizer",
                    suffix="character_flow",
                    draw_objects=None,
                    draw_type=None
                )
                
        except Exception as e:
            # Log error but don't halt execution for debug image generation
            if hasattr(self, 'logger'):
                self.logger.warning(f"Error generating debug image: {str(e)}")
            
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
        # Add the original plate image to debug info if needed
        if self.save_debug_images:
            save_debug_image(
                image=plate_image,
                debug_dir=self.debug_images_dir,
                prefix="char_process",
                suffix="plate_input",
                draw_objects=None,
                draw_type=None
            )
        
        # 1. Detect characters
        characters = self.detect_characters(plate_image)
        
        # Add plate_image reference to each character for debugging
        if self.save_debug_images:
            for char in characters:
                char['plate_image'] = plate_image
        
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
        
        # 6. Save full OCR result image if debug is enabled
        if self.save_debug_images:
            # Create a debug image showing the final OCR result
            ocr_result_img = plate_image.copy()
            
            # Draw OCR results
            for i, char_result in enumerate(char_results):
                if 'box' in char_result:
                    x1, y1, x2, y2 = char_result['box']
                    # Draw box
                    cv2.rectangle(ocr_result_img, (x1, y1), (x2, y2), (0, 255, 0), 1)
                    # Draw character
                    cv2.putText(ocr_result_img, char_result['char'], (x1, y1-5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Draw the license number at the bottom
            h, w = ocr_result_img.shape[:2]
            cv2.putText(ocr_result_img, f"License: {license_number} ({avg_confidence:.2f})", 
                       (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Save the debug image
            save_debug_image(
                image=ocr_result_img,
                debug_dir=self.debug_images_dir,
                prefix="char_ocr",
                suffix=f"result_{license_number}",
                draw_objects=None,
                draw_type=None
            )
        
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
        
        # Save debug image with alternative combinations if enabled
        if self.save_debug_images and len(combinations) > 1:
            # Create a blank image to show alternatives
            alt_img = np.ones((200, 400, 3), dtype=np.uint8) * 255
            
            # Draw each alternative
            for i, combo in enumerate(combinations[:5]):  # Show up to 5 alternatives
                text = f"{i+1}. {combo['plate']} ({combo['confidence']:.2f})"
                cv2.putText(alt_img, text, (20, 40 + i*30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            
            # Save the alternatives image
            save_debug_image(
                image=alt_img,
                debug_dir=self.debug_images_dir,
                prefix="char_alternatives",
                suffix=f"top_{len(combinations)}",
                draw_objects=None,
                draw_type=None
            )
        
        return combinations[:max_combinations]


class CharDetector(YOLOBase):
    """Character detector for license plates using YOLOv8 or ONNX."""
    
    def __init__(self, model_path: str, task: str, use_onnx: bool, use_cuda: bool, 
                 resolution: Tuple[int, int], confidence: float, 
                 save_debug_images: bool = False, debug_images_dir: str = None):
        """
        Initialize the character detector.
        
        Args:
            model_path: Path to the model file
            task: Task type
            use_onnx: Whether to use ONNX
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
        
        # Save resized input image if debug is enabled
        if self.save_debug_images:
            save_debug_image(
                image=plate_resized,
                debug_dir=self.debug_images_dir,
                prefix="char_detector",
                suffix="resized_input",
                draw_objects=None,
                draw_type=None
            )
        
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
        
        # Save model visualization if debug is enabled
        if self.save_debug_images and hasattr(results, 'plot'):
            try:
                # Plot the results using the model's built-in plotting
                plot_img = results.plot()
                save_debug_image(
                    image=plot_img,
                    debug_dir=self.debug_images_dir,
                    prefix="char_detector",
                    suffix="model_output",
                    draw_objects=None,
                    draw_type=None
                )
            except Exception as e:
                print(f"Error plotting character detection results: {e}")
        
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
                
                # Save individual character crop if debug is enabled
                if self.save_debug_images:
                    save_debug_image(
                        image=char_img,
                        debug_dir=self.debug_images_dir,
                        prefix="char_crop",
                        suffix=f"char_{i}",
                        draw_objects=None,
                        draw_type=None
                    )
                
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
                
                # Save individual character crop if debug is enabled
                if self.save_debug_images:
                    save_debug_image(
                        image=char_img,
                        debug_dir=self.debug_images_dir,
                        prefix="char_crop",
                        suffix=f"char_{i}_onnx",
                        draw_objects=None,
                        draw_type=None
                    )
                
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
                 resolution: Tuple[int, int], confidence: float,
                 save_debug_images: bool = False, debug_images_dir: str = None):
        """
        Initialize the character classifier.
        
        Args:
            model_path: Path to the model file
            task: Task type
            use_onnx: Whether to use ONNX
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
        
        # Create centered image on black background maintaining aspect ratio
        try:
            height, width = char_image.shape[:2]
            target_height = self.resolution[1]
            
            # Calculate new width maintaining aspect ratio
            new_width = int(width * (target_height / height))
            
            # Resize the image to the new height while maintaining aspect ratio
            resized_char = cv2.resize(char_image, (new_width, target_height))
            
            # Get the number of channels from the input image
            channels = 3 if len(char_image.shape) == 3 else 1
            
            # Create a black background image with resolution size and same number of channels
            if channels == 1:
                black_background = np.zeros((self.resolution[1], self.resolution[0]), dtype=np.uint8)
            else:
                black_background = np.zeros((self.resolution[1], self.resolution[0], channels), dtype=np.uint8)
            
            # Calculate position to center the image
            x_offset = (self.resolution[0] - new_width) // 2
            
            # Ensure the offset is valid
            x_offset = max(0, x_offset)
            
            # Create the portion where the character will be placed
            x_end = min(x_offset + new_width, self.resolution[0])
            width_to_use = x_end - x_offset
            
            # Place the resized character image on the black background
            if channels == 1:
                black_background[:, x_offset:x_end] = resized_char[:, :width_to_use]
            else:
                black_background[:, x_offset:x_end, :] = resized_char[:, :width_to_use, :]
            
            char_resized = black_background
            
            # Save resized character image if debug is enabled
            if self.save_debug_images:
                save_debug_image(
                    image=char_resized,
                    debug_dir=self.debug_images_dir,
                    prefix="char_classifier",
                    suffix="resized_input",
                    draw_objects=None,
                    draw_type=None
                )
        except Exception as e:
            raise CharacterRecognitionError(f"Failed to resize and center character image: {str(e)}")
        
        try:
            if self.use_onnx:
                result = self._classify_onnx(char_resized)
            else:
                result = self._classify_pytorch(char_resized)
                
            # Save classification result visualization if debug is enabled
            if self.save_debug_images:
                # Create a visualization of the classification result
                result_img = char_resized.copy()
                # Scale the image for better visibility (3x)
                result_img = cv2.resize(result_img, (self.resolution[0] * 3, self.resolution[1] * 3))
                
                # Draw the top prediction
                top_char, top_conf = result[0] if result else ("?", 0.0)
                label = f"{top_char}: {top_conf:.2f}"
                cv2.putText(result_img, label, (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Add other top predictions
                y_offset = 60
                for i, (char, conf) in enumerate(result[1:]):
                    alt_label = f"{char}: {conf:.2f}"
                    cv2.putText(result_img, alt_label, (10, y_offset), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 1)
                    y_offset += 25
                
                save_debug_image(
                    image=result_img,
                    debug_dir=self.debug_images_dir,
                    prefix="char_classifier",
                    suffix=f"result_{top_char}",
                    draw_objects=None,
                    draw_type=None
                )
                
            return result
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
