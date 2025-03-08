"""
YOLOv8 models for Automatic License Plate Recognition.

This module contains the YOLOv8-based detectors and classifiers for license plates,
characters, states, and vehicles. Supports both PyTorch and ONNX formats.
"""

# Export base class
from .base import YOLOBase

# Export detector and classifier classes
from .plate_detector import PlateDetector
from .character_detector import CharacterDetector
from .state_classifier import StateClassifier
from .vehicle_detector import VehicleDetector
