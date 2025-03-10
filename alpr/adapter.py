"""
Adapter for the ALPR system to integrate with CodeProject.AI SDK.
"""
import os
import sys
import time
import json
import numpy as np
import cv2
from typing import Dict, Any, List

# For PyTorch on Apple silicon
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Import CodeProject.AI SDK
from codeproject_ai_sdk import RequestData, ModuleRunner, LogMethod, JSON

# Import ALPR system
from .config import ALPRConfig, load_from_env
from .core import ALPRSystem
from .exceptions import ALPRException


class ALPRAdapter(ModuleRunner):
    """
    Adapter class to integrate the ALPR system with CodeProject.AI Server.
    """
    
    def __init__(self):
        """Initialize the ALPR adapter"""
        super().__init__()
        
        # Load configuration from environment variables
        self.config = load_from_env()
        
        # These will be adjusted based on the hardware / packages found
        self.use_CUDA = self.config.use_cuda
        self.use_MPS = self.config.use_mps
        self.use_DirectML = self.config.use_directml

        # Half precision setting from ModuleRunner
        if self.use_CUDA and self.half_precision == 'enable' and \
           not self.system_info.hasTorchHalfPrecision:
            self.half_precision = 'disable'

        # Initialize the ALPR system once for use across requests
        self.alpr_system = None
        
        # Initialize statistics tracking
        self._plates_detected = 0
        self._histogram = {}
        
        # Create debug images directory if enabled
        if self.config.save_debug_images and not os.path.exists(self.config.debug_images_dir):
            os.makedirs(self.config.debug_images_dir, exist_ok=True)
            self.log(LogMethod.Info | LogMethod.Server,
            { 
                "filename": __file__,
                "loglevel": "information",
                "method": sys._getframe().f_code.co_name,
                "message": f"Debug images will be saved to {self.config.debug_images_dir}"
            })

    def initialise(self):
        """Initialize the adapter and ALPR system"""
        # CUDA takes precedence
        if self.use_CUDA:
            self.use_CUDA = self.system_info.hasTorchCuda
            # Potentially solve an issue around CUDNN_STATUS_ALLOC_FAILED errors
            try:
                import cudnn as cudnn
                if cudnn.is_available():
                    cudnn.benchmark = False
            except:
                pass

        # If no CUDA, maybe we're on an Apple Silicon Mac?
        if self.use_CUDA:
            self.use_MPS = False
            self.use_DirectML = False
        else:
            self.use_MPS = self.system_info.hasTorchMPS

        # DirectML currently not supported
        self.use_DirectML = False

        # Update configuration with detected hardware capabilities
        self.config.use_cuda = self.use_CUDA
        self.config.use_mps = self.use_MPS
        self.config.use_directml = self.use_DirectML
        
        # Set inference device information for ModuleRunner
        self.can_use_GPU = self.system_info.hasTorchCuda or self.system_info.hasTorchMPS

        if self.use_CUDA:
            self.inference_device = "GPU"
            self.inference_library = "CUDA"
            device = "cuda"
        elif self.use_MPS:
            self.inference_device = "GPU"
            self.inference_library = "MPS"
            device = "mps"
        else:
            device = "cpu"

        # Initialize the ALPR system
        try:
            self.log(LogMethod.Info | LogMethod.Server,
            { 
                "filename": __file__,
                "loglevel": "information",
                "method": sys._getframe().f_code.co_name,
                "message": f"Initializing ALPR system with models from {self.config.models_dir}"
            })
            
            # Log debug image settings
            if self.config.save_debug_images:
                self.log(LogMethod.Info | LogMethod.Server,
                {
                    "filename": __file__,
                    "loglevel": "information", 
                    "method": sys._getframe().f_code.co_name,
                    "message": f"Debug image saving is enabled. Images will be saved to {self.config.debug_images_dir}"
                })
            
            # Initialize the ALPR system
            self.alpr_system = ALPRSystem(self.config)
            
            self.log(LogMethod.Info | LogMethod.Server,
            {
                "filename": __file__,
                "loglevel": "information", 
                "method": sys._getframe().f_code.co_name,
                "message": f"ALPR system initialized successfully"
            })
            
        except Exception as ex:
            self.report_error(ex, __file__, f"Error initializing ALPR system: {str(ex)}")
            self.alpr_system = None

    def process(self, data: RequestData) -> JSON:
        """
        Process a request from CodeProject.AI Server.
        
        Args:
            data: Request data
            
        Returns:
            JSON response
        """
        response = None

        try:
            # The route to here is /v1/vision/alpr
            img = data.get_image(0)
                        
            # Get thresholds
            plate_threshold = float(data.get_value("min_confidence", "0.4"))
            
            # Only detect license plates
            response = self.detect_license_plate(img, plate_threshold)

        except Exception as ex:
            response = { "success": False, "error": f"Error processing request: {str(ex)}" }
            self.report_error(ex, __file__, f"Error processing request: {str(ex)}")

        return response

    def detect_license_plate(self, img, threshold: float) -> Dict[str, Any]:
        """
        Detect license plates in an image.
        
        Args:
            img: Input image
            threshold: Confidence threshold
            
        Returns:
            JSON response with detection results
        """
        if self.alpr_system is None:
            return {"success": False, "error": "ALPR system not initialized"}
        
        try:
            # Use the ALPR system to detect license plates
            result = self.alpr_system.detect_license_plate(img, threshold)
            
            # Update statistics
            self._plates_detected += len(result.get("predictions", []))
            for plate in result.get("predictions", []):
                license_num = plate["label"]
                if license_num not in self._histogram:
                    self._histogram[license_num] = 1
                else:
                    self._histogram[license_num] += 1
            
            return result
            
        except Exception as ex:
            self.report_error(ex, __file__, f"Error detecting license plates: {str(ex)}")
            return {"success": False, "error": f"Error detecting license plates: {str(ex)}"}
    
    def status(self) -> JSON:
        """
        Get the status of the ALPR adapter.
        
        Returns:
            Status information
        """
        statusData = super().status()
        statusData["platesDetected"] = self._plates_detected
        statusData["histogram"] = self._histogram
        
        # Add debug images information
        statusData["debugImagesEnabled"] = self.config.save_debug_images
        if self.config.save_debug_images:
            statusData["debugImagesDir"] = self.config.debug_images_dir
            # Count the number of debug images if the directory exists
            if os.path.exists(self.config.debug_images_dir):
                debug_files = [f for f in os.listdir(self.config.debug_images_dir) 
                              if f.endswith('.jpg') or f.endswith('.png')]
                statusData["debugImagesCount"] = len(debug_files)
        
        return statusData

    def selftest(self) -> JSON:
        """
        Run a self-test on the ALPR system.
        
        Returns:
            Self-test results
        """
        # If we don't have any test images, just return success if we could initialize the system
        if self.alpr_system is None:
            return {
                "success": False,
                "message": "ALPR system failed to initialize"
            }
        
        test_file = os.path.join("test", "license_plate_test.jpg")
        if not os.path.exists(test_file):
            return {
                "success": True,
                "message": "ALPR system initialized successfully (no test image available)"
            }
        
        # Test with an actual image
        request_data = RequestData()
        request_data.queue = self.queue_name
        request_data.command = "detect"
        request_data.add_file(test_file)
        request_data.add_value("operation", "plate")
        request_data.add_value("min_confidence", 0.4)
        
        result = self.process(request_data)
        print(f"Info: Self-test for {self.module_id}. Success: {result['success']}")
        
        if result['success']:
            message = "ALPR system test successful"
            if result.get('count', 0) > 0:
                message += f" - detected {result['count']} license plates"
                
                # Add debug images information if enabled
                if self.config.save_debug_images:
                    message += f". Debug images saved to {self.config.debug_images_dir}"
        else:
            message = "ALPR system test failed"
            
        return { "success": result['success'], "message": message }


if __name__ == "__main__":
    ALPRAdapter().start_loop()
