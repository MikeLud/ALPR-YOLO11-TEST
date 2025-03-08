"""
Base classes for YOLOv8 models, supporting both PyTorch and ONNX.
"""
import os
import numpy as np
import cv2
from typing import Any, Dict, List, Optional, Tuple, Union
from ultralytics import YOLO

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False


class YOLOBase:
    """
    Base class for YOLO models with support for both PyTorch and ONNX.
    """
    
    def __init__(self, model_path: str, task: str, use_onnx: bool = False, use_cuda: bool = True):
        """
        Initialize a YOLO model (PyTorch or ONNX).
        
        Args:
            model_path: Path to the model file
            task: Task type ('detect', 'classify', 'pose')
            use_onnx: Whether to use ONNX model
            use_cuda: Whether to use CUDA for inference
        """
        self.model_path = model_path
        self.task = task
        self.use_onnx = use_onnx
        self.model = None
        self.onnx_session = None
        self.names = {}  # Class names dictionary
        
        # For ONNX runtime
        self.input_name = None
        self.output_names = None
        
        # Initialize model based on format
        if use_onnx:
            self._init_onnx_model(use_cuda)
        else:
            self._init_pytorch_model()
    
    def _init_pytorch_model(self):
        """Initialize PyTorch YOLO model"""
        self.model = YOLO(self.model_path, task=self.task)
        # Store class names for later use
        if hasattr(self.model, 'names'):
            self.names = self.model.names
    
    def _init_onnx_model(self, use_cuda: bool):
        """Initialize ONNX runtime session"""
        if not ONNX_AVAILABLE:
            raise ImportError("ONNX Runtime is not available. Please install it with 'pip install onnxruntime'")
            
        # Set up ONNX runtime session
        providers = []
        if use_cuda and 'CUDAExecutionProvider' in ort.get_available_providers():
            providers.append('CUDAExecutionProvider')
        providers.append('CPUExecutionProvider')
        
        # Create ONNX session
        self.onnx_session = ort.InferenceSession(self.model_path, providers=providers)
        
        # Get input and output names
        self.input_name = self.onnx_session.get_inputs()[0].name
        self.output_names = [output.name for output in self.onnx_session.get_outputs()]
        
        # Load class names from accompanying JSON file if available
        class_file = os.path.splitext(self.model_path)[0] + '.json'
        if os.path.exists(class_file):
            import json
            with open(class_file, 'r') as f:
                self.names = json.load(f)
        else:
            # Create a default mapping if no class file
            self.names = {i: str(i) for i in range(1000)}  # Default large number of classes
    
    def _preprocess_image(self, image: np.ndarray, input_size: Tuple[int, int]) -> np.ndarray:
        """
        Preprocess image for ONNX inference.
        
        Args:
            image: Input image as numpy array
            input_size: Model input size (width, height)
            
        Returns:
            Preprocessed image as numpy array
        """
        # Resize image
        resized = cv2.resize(image, input_size)
        
        # Convert to RGB if needed
        if len(resized.shape) == 3 and resized.shape[2] == 3:
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Normalize and convert to float32
        normalized = resized.astype(np.float32) / 255.0
        
        # Add batch dimension and ensure proper channel order (NCHW)
        if len(normalized.shape) == 3:
            return np.transpose(normalized, (2, 0, 1))[np.newaxis, ...]
        else:
            return normalized[np.newaxis, np.newaxis, ...]
