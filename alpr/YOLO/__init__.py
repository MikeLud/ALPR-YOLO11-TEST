"""
YOLOv8 models for Automatic License Plate Recognition.

This module contains the YOLOv8-based detectors and classifiers for license plates,
characters, states, and vehicles.
"""

# Export classes for easy access
from .plate_detector import PlateDetector
from .character_detector import CharacterDetector
from .state_classifier import StateClassifier
from .vehicle_detector import VehicleDetector
