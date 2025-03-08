"""
Image processing utilities for the ALPR system.
"""
import cv2
import numpy as np
from typing import List, Tuple, Optional, Union

from ..exceptions import ImageProcessingError


def resize_image(image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """
    Resize an image to the specified dimensions.
    
    Args:
        image: Input image as numpy array
        size: Target size as (width, height)
        
    Returns:
        Resized image
        
    Raises:
        ImageProcessingError: If resizing fails
    """
    try:
        return cv2.resize(image, size)
    except Exception as e:
        raise ImageProcessingError("resize", e)


def order_points(points: np.ndarray) -> np.ndarray:
    """
    Order points in the sequence: top-left, top-right, bottom-right, bottom-left.
    
    Args:
        points: Array of points (x, y)
        
    Returns:
        Ordered points
        
    Raises:
        ImageProcessingError: If ordering fails
    """
    try:
        # Initialize a list of coordinates that will be ordered
        rect = np.zeros((4, 2), dtype=np.float32)
        
        # The top-left point will have the smallest sum
        # The bottom-right point will have the largest sum
        s = points.sum(axis=1)
        rect[0] = points[np.argmin(s)]
        rect[2] = points[np.argmax(s)]
        
        # Now compute the difference between the points
        # The top-right point will have the smallest difference
        # The bottom-left point will have the largest difference
        diff = np.diff(points, axis=1)
        rect[1] = points[np.argmin(diff)]
        rect[3] = points[np.argmax(diff)]
        
        return rect
    except Exception as e:
        raise ImageProcessingError("order_points", e)


def four_point_transform(
    image: np.ndarray, 
    corners: Union[List[List[float]], np.ndarray], 
    aspect_ratio: Optional[float] = None
) -> np.ndarray:
    """
    Apply a 4-point perspective transform to extract a region (e.g., license plate).
    If aspect_ratio is set, the output will have that aspect ratio with fixed height.
    
    Args:
        image: Original image
        corners: List of 4 corner points [x, y]
        aspect_ratio: Optional width/height ratio for the output
        
    Returns:
        Warped image of the region
        
    Raises:
        ImageProcessingError: If transformation fails
    """
    try:
        # Convert corners to numpy array
        corners = np.array(corners, dtype=np.float32)
        
        # Ensure we have exactly 4 points
        if corners.shape[0] != 4:
            if corners.shape[0] > 4:
                corners = corners[:4]
            else:
                # Not enough points, pad with zeros
                padded_corners = np.zeros((4, 2), dtype=np.float32)
                padded_corners[:corners.shape[0]] = corners
                corners = padded_corners
                
        # Order the points correctly
        rect = order_points(corners)
        (tl, tr, br, bl) = rect
        
        # Compute the width of the new image
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        max_width = max(int(widthA), int(widthB))
        
        # Compute the height of the new image
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        max_height = max(int(heightA), int(heightB))
        
        # Create output dimensions
        output_width = max_width
        output_height = max_height
        
        # Apply aspect ratio if specified (width/height)
        if aspect_ratio is not None and aspect_ratio > 0:
            # Keep height fixed and calculate width based on the desired aspect ratio
            output_width = int(output_height * aspect_ratio)
        
        # Ensure dimensions are at least 1 pixel
        output_width = max(1, output_width)
        output_height = max(1, output_height)
        
        # Construct the set of destination points for the transform
        dst = np.array([
            [0, 0],
            [output_width - 1, 0],
            [output_width - 1, output_height - 1],
            [0, output_height - 1]
        ], dtype=np.float32)
        
        # Compute the perspective transform matrix and apply it
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (output_width, output_height))
        
        return warped
    except Exception as e:
        raise ImageProcessingError("four_point_transform", e)


def dilate_corners(
    corners: np.ndarray, 
    dilation_pixels: int = 5
) -> np.ndarray:
    """
    Dilate the corners by moving them outward from the centroid.
    
    Args:
        corners: Numpy array of shape (4, 2) containing the corner coordinates
        dilation_pixels: Number of pixels to dilate the corners
        
    Returns:
        Dilated corners as a numpy array of the same shape
        
    Raises:
        ImageProcessingError: If dilation fails
    """
    try:
        if dilation_pixels <= 0:
            return corners.copy()
            
        # Calculate the centroid
        centroid = np.mean(corners, axis=0)
        
        # Create a copy of the corners that we will modify
        dilated_corners = corners.copy()
        
        # For each corner, move it away from the centroid
        for i in range(len(corners)):
            # Vector from centroid to corner
            vector = corners[i] - centroid
            
            # Normalize the vector
            vector_length = np.sqrt(np.sum(vector**2))
            if vector_length > 0:  # Avoid division by zero
                unit_vector = vector / vector_length
                
                # Extend the corner by the dilation amount in the direction of the unit vector
                dilated_corners[i] = corners[i] + unit_vector * dilation_pixels
        
        return dilated_corners
    except Exception as e:
        raise ImageProcessingError("dilate_corners", e)
