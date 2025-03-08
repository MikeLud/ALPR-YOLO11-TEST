"""
Automatic License Plate Recognition (ALPR) module for CodeProject.AI Server.

This module provides license plate detection, character recognition, state identification,
and vehicle detection capabilities using YOLOv8 models.
"""

__version__ = "1.0.0"
__author__ = "License Plate Recognition Team"

# Export core classes for easy access
from .adapter import ALPRAdapter
from .core import ALPRSystem
from .config import ALPRConfig, load_from_env
from .exceptions import ALPRException

# Export YOLO model classes
from .YOLO.plate_detector import PlateDetector
from .YOLO.character_detector import CharacterDetector
from .YOLO.state_classifier import StateClassifier
from .YOLO.vehicle_detector import VehicleDetector
