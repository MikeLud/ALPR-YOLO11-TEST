"""
Configuration module for the ALPR system.
Handles loading and validating configuration from environment variables and files.
"""
import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from codeproject_ai_sdk import ModuleOptions

@dataclass
class ALPRConfig:
    """Configuration for the ALPR system"""
    # Paths
    app_dir: str = field(default_factory=lambda: os.path.normpath(os.getcwd()))
    models_dir: str = field(default_factory=lambda: os.path.normpath(os.path.join(os.getcwd(), "models")))
    
    # Feature flags
    enable_state_detection: bool = True
    enable_vehicle_detection: bool = True
    
    # Confidence thresholds
    plate_detector_confidence: float = 0.45
    state_classifier_confidence: float = 0.45
    char_detector_confidence: float = 0.40
    char_classifier_confidence: float = 0.40
    vehicle_detector_confidence: float = 0.45
    vehicle_classifier_confidence: float = 0.45
    
    # Processing parameters
    plate_aspect_ratio: Optional[float] = 4.0
    corner_dilation_pixels: int = 5
    
    # Hardware acceleration
    use_cuda: bool = True
    use_mps: bool = True  # Apple Silicon GPU
    use_directml: bool = False  # DirectML for Windows
    
    # Derived properties
    _model_paths: Dict[str, str] = field(default_factory=dict, init=False)
    
    def __post_init__(self):
        """Validate configuration and set derived properties"""
        self.validate()
        
        # Set model paths
        self._model_paths = {
            "plate_detector": os.path.join(self.models_dir, "plate_detector.pt"),
            "state_classifier": os.path.join(self.models_dir, "state_classifier.pt"),
            "char_detector": os.path.join(self.models_dir, "char_detector.pt"),
            "char_classifier": os.path.join(self.models_dir, "char_classifier.pt"),
            "vehicle_detector": os.path.join(self.models_dir, "vehicle_detector.pt"),
            "vehicle_classifier": os.path.join(self.models_dir, "vehicle_classifier.pt"),
        }
    
    def validate(self) -> None:
        """Validate the configuration values"""
        # Validate confidence thresholds
        for name, value in {
            "plate_detector_confidence": self.plate_detector_confidence,
            "state_classifier_confidence": self.state_classifier_confidence,
            "char_detector_confidence": self.char_detector_confidence,
            "char_classifier_confidence": self.char_classifier_confidence,
            "vehicle_detector_confidence": self.vehicle_detector_confidence,
            "vehicle_classifier_confidence": self.vehicle_classifier_confidence
        }.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"Confidence threshold {name} must be between 0.0 and 1.0")
        
        # Validate paths
        if not os.path.exists(self.models_dir):
            raise ValueError(f"Models directory does not exist: {self.models_dir}")
        
        # Validate plate aspect ratio
        if self.plate_aspect_ratio is not None and self.plate_aspect_ratio <= 0:
            raise ValueError(f"Plate aspect ratio must be positive, got {self.plate_aspect_ratio}")
        
        # Validate corner dilation
        if self.corner_dilation_pixels < 0:
            raise ValueError(f"Corner dilation pixels must be non-negative, got {self.corner_dilation_pixels}")
    
    def get_model_path(self, model_name: str) -> str:
        """Get the path to a specific model"""
        if model_name not in self._model_paths:
            raise ValueError(f"Unknown model: {model_name}")
        return self._model_paths[model_name]
    
    def as_dict(self) -> Dict[str, Any]:
        """Convert configuration to a dictionary"""
        return {
            "paths": {
                "app_dir": self.app_dir,
                "models_dir": self.models_dir,
            },
            "features": {
                "enable_state_detection": self.enable_state_detection,
                "enable_vehicle_detection": self.enable_vehicle_detection,
            },
            "confidence_thresholds": {
                "plate_detector": self.plate_detector_confidence,
                "state_classifier": self.state_classifier_confidence,
                "char_detector": self.char_detector_confidence,
                "char_classifier": self.char_classifier_confidence,
                "vehicle_detector": self.vehicle_detector_confidence,
                "vehicle_classifier": self.vehicle_classifier_confidence,
            },
            "processing": {
                "plate_aspect_ratio": self.plate_aspect_ratio,
                "corner_dilation_pixels": self.corner_dilation_pixels,
            },
            "hardware": {
                "use_cuda": self.use_cuda,
                "use_mps": self.use_mps,
                "use_directml": self.use_directml,
            }
        }
        
    def __str__(self) -> str:
        """Convert configuration to a string representation"""
        import json
        return json.dumps(self.as_dict(), indent=2)


def load_from_env() -> ALPRConfig:
    """Load configuration from environment variables"""
    app_dir = os.path.normpath(ModuleOptions.getEnvVariable("APPDIR", os.getcwd()))
    models_dir = os.path.normpath(ModuleOptions.getEnvVariable("MODELS_DIR", f"{app_dir}/models"))
    
    # Feature flags
    enable_state_detection = ModuleOptions.getEnvVariable("ENABLE_STATE_DETECTION", "True").lower() == "true"
    enable_vehicle_detection = ModuleOptions.getEnvVariable("ENABLE_VEHICLE_DETECTION", "True").lower() == "true"
    
    # Confidence thresholds
    plate_detector_confidence = float(ModuleOptions.getEnvVariable("PLATE_DETECTOR_CONFIDENCE", "0.45"))
    state_classifier_confidence = float(ModuleOptions.getEnvVariable("STATE_CLASSIFIER_CONFIDENCE", "0.45"))
    char_detector_confidence = float(ModuleOptions.getEnvVariable("CHAR_DETECTOR_CONFIDENCE", "0.40"))
    char_classifier_confidence = float(ModuleOptions.getEnvVariable("CHAR_CLASSIFIER_CONFIDENCE", "0.40"))
    vehicle_detector_confidence = float(ModuleOptions.getEnvVariable("VEHICLE_DETECTOR_CONFIDENCE", "0.45"))
    vehicle_classifier_confidence = float(ModuleOptions.getEnvVariable("VEHICLE_CLASSIFIER_CONFIDENCE", "0.45"))
    
    # Processing parameters
    plate_aspect_ratio_str = ModuleOptions.getEnvVariable("PLATE_ASPECT_RATIO", "4.0")
    plate_aspect_ratio = float(plate_aspect_ratio_str) if plate_aspect_ratio_str and plate_aspect_ratio_str != "0" else None
    corner_dilation_pixels = int(ModuleOptions.getEnvVariable("CORNER_DILATION_PIXELS", "5"))
    
    # Hardware acceleration
    use_cuda = ModuleOptions.getEnvVariable("USE_CUDA", "True").lower() == "true"
    use_mps = True  # Default to true, will be checked for availability later
    use_directml = False  # Not yet supported
    
    return ALPRConfig(
        app_dir=app_dir,
        models_dir=models_dir,
        enable_state_detection=enable_state_detection,
        enable_vehicle_detection=enable_vehicle_detection,
        plate_detector_confidence=plate_detector_confidence,
        state_classifier_confidence=state_classifier_confidence,
        char_detector_confidence=char_detector_confidence,
        char_classifier_confidence=char_classifier_confidence,
        vehicle_detector_confidence=vehicle_detector_confidence,
        vehicle_classifier_confidence=vehicle_classifier_confidence,
        plate_aspect_ratio=plate_aspect_ratio,
        corner_dilation_pixels=corner_dilation_pixels,
        use_cuda=use_cuda,
        use_mps=use_mps,
        use_directml=use_directml
    )
