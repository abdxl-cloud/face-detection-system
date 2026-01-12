"""
Media Manager Client
Handles uploading images to the external media manager API
"""

import requests
import logging
import cv2
import numpy as np
from typing import Optional, Dict

logger = logging.getLogger(__name__)


class MediaManagerClient:
    """Client for uploading images to the media manager API"""

    def __init__(self, config: Dict):
        """
        Initialize media manager client

        Args:
            config: Configuration dictionary
        """
        media_config = config.get('media_manager', {})
        self.base_url = media_config.get('base_url', 'https://etraffica-media-manager.ngrok.app')
        self.upload_endpoint = f"{self.base_url}/api/FileManager/upload-image"
        self.enabled = media_config.get('enabled', True)
        self.timeout = media_config.get('timeout', 30)
        self.health_endpoint = media_config.get('health_endpoint')

        logger.info(f"Media Manager Client initialized: {self.upload_endpoint}")

    def _extract_url(self, response_data: Dict) -> Optional[str]:
        """Extract upload URL from known response shapes."""
        if not isinstance(response_data, dict):
            return None

        # Common top-level keys
        url = response_data.get('url') or response_data.get('file_url') or response_data.get('path')
        if url:
            return url

        # Known nested structure: { "data": { "uri": "..." } }
        data = response_data.get('data') if isinstance(response_data.get('data'), dict) else None
        if data:
            return data.get('uri') or data.get('url') or data.get('file_url') or data.get('path')

        return None

    def upload_image(
        self,
        image: np.ndarray,
        filename: str,
        image_type: str = "processed"
    ) -> Optional[Dict]:
        """
        Upload image to media manager

        Args:
            image: Image as numpy array (BGR format)
            filename: Filename for the image
            image_type: Type of image (raw, windshield_crop, visualization, etc.)

        Returns:
            Response dict with URL and metadata, or None if upload fails
        """
        if not self.enabled:
            logger.warning("Media manager uploads are disabled")
            return None

        try:
            # Encode image to JPEG format
            success, encoded_image = cv2.imencode('.jpg', image)
            if not success:
                logger.error("Failed to encode image")
                return None

            image_bytes = encoded_image.tobytes()

            # Prepare multipart form data
            files = {
                'Upload': (filename, image_bytes, 'image/jpeg')
            }

            # Additional metadata (optional)
            data = {
                'type': image_type
            }

            logger.info(f"Uploading {image_type} image: {filename} ({len(image_bytes)} bytes)")

            # Make POST request
            response = requests.post(
                self.upload_endpoint,
                files=files,
                data=data,
                timeout=self.timeout
            )

            # Check response
            if response.status_code == 200:
                response_data = response.json()
                logger.info(f" Image uploaded successfully: {filename}")
                url = self._extract_url(response_data)

                return {
                    'status': 'success',
                    'filename': filename,
                    'type': image_type,
                    'url': url,
                    'response': response_data
                }
            else:
                logger.error(f"Upload failed with status {response.status_code}: {response.text}")
                return None

        except requests.exceptions.Timeout:
            logger.error(f"Upload timeout for {filename}")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Upload request failed for {filename}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error uploading {filename}: {e}")
            return None

    def upload_multiple(
        self,
        images: Dict[str, np.ndarray]
    ) -> Dict[str, Optional[Dict]]:
        """
        Upload multiple images

        Args:
            images: Dictionary of {image_type: image_array}

        Returns:
            Dictionary of {image_type: upload_response}
        """
        results = {}

        for image_type, image in images.items():
            filename = f"{image_type}.jpg"
            result = self.upload_image(image, filename, image_type)
            results[image_type] = result

        return results

    def is_available(self) -> bool:
        """Check if media manager is available"""
        if not self.enabled:
            return False

        # If no health endpoint configured, assume available when enabled.
        if not self.health_endpoint:
            return True

        try:
            response = requests.get(self.health_endpoint, timeout=5)
            return response.status_code == 200
        except:
            return True
