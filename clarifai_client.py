"""
Clarifai Face Detection Client
Handles all Clarifai API interactions
"""

import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

try:
    from clarifai_grpc.channel.clarifai_channel import ClarifaiChannel
    from clarifai_grpc.grpc.api import resources_pb2, service_pb2, service_pb2_grpc
    from clarifai_grpc.grpc.api.status import status_code_pb2
    clarifai_available = True
except ImportError:
    clarifai_available = False


class ClarifaiFaceDetector:
    """Clarifai face detection client"""
    
    def __init__(self, config: Dict):
        """
        Initialize Clarifai client
        
        Args:
            config: Configuration dictionary
        """
        if not clarifai_available:
            logger.error("Clarifai SDK not available")
            raise ImportError("Clarifai SDK not installed. Install with: pip install clarifai-grpc")
        
        clarifai_config = config.get('clarifai', {})
        self.api_key = clarifai_config.get('api_key')
        self.user_id = clarifai_config.get('user_id', 'clarifai')
        self.app_id = clarifai_config.get('app_id', 'main')
        self.model_id = clarifai_config.get('model_id', 'face-detection')
        self.version_id = clarifai_config.get('version_id', '6dc7e46bc9124c5c8824be4822abe105')
        
        if not self.api_key:
            raise ValueError("Clarifai API key not configured")
        
        logger.info(f"✓ Clarifai face detector initialized (model: {self.model_id})")
    
    def detect_faces(self, image_bytes: bytes) -> Tuple[List[Dict], float]:
        """
        Detect faces in image using Clarifai API
        
        Args:
            image_bytes: Image as bytes
            
        Returns:
            Tuple of (faces list, detection time in seconds)
        """
        import time
        start_time = time.time()
        
        try:
            # Setup gRPC channel and stub
            channel = ClarifaiChannel.get_grpc_channel()
            stub = service_pb2_grpc.V2Stub(channel)
            metadata = (('authorization', f'Key {self.api_key}'),)
            
            # Create request
            request = service_pb2.PostModelOutputsRequest(
                user_app_id=resources_pb2.UserAppIDSet(
                    user_id=self.user_id,
                    app_id=self.app_id
                ),
                model_id=self.model_id,
                version_id=self.version_id,
                inputs=[
                    resources_pb2.Input(
                        data=resources_pb2.Data(
                            image=resources_pb2.Image(base64=image_bytes)
                        )
                    )
                ]
            )
            
            # Make API call
            response = stub.PostModelOutputs(request, metadata=metadata)
            detection_time = time.time() - start_time
            
            # Check response status
            if response.status.code != status_code_pb2.SUCCESS:
                raise Exception(f"Clarifai API error: {response.status.description}")
            
            # Parse faces
            faces = []
            if response.outputs:
                output = response.outputs[0]
                for i, region in enumerate(output.data.regions):
                    bbox = region.region_info.bounding_box
                    faces.append({
                        "face_id": i,
                        "bbox": {
                            "left": bbox.left_col,
                            "top": bbox.top_row,
                            "right": bbox.right_col,
                            "bottom": bbox.bottom_row
                        }
                    })
            
            logger.info(f"Detected {len(faces)} face(s) in {detection_time:.2f}s")
            return faces, detection_time
            
        except Exception as e:
            logger.error(f"Error detecting faces: {e}")
            raise
    
    @staticmethod
    def is_available() -> bool:
        """Check if Clarifai SDK is available"""
        return clarifai_available