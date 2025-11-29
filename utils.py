"""
Utility Functions
Helper functions for image processing and HTTP requests
"""

import cv2
import numpy as np
import requests
from fastapi import HTTPException
import logging

logger = logging.getLogger(__name__)


def image_to_bytes(image: np.ndarray) -> bytes:
    """
    Convert OpenCV image to bytes
    
    Args:
        image: Image as numpy array (BGR format)
        
    Returns:
        Image as bytes
    """
    _, buffer = cv2.imencode('.jpg', image)
    return buffer.tobytes()


def download_image_from_url(url: str, timeout: int = 10) -> np.ndarray:
    """
    Download image from URL
    
    Args:
        url: Image URL
        timeout: Request timeout in seconds
        
    Returns:
        Image as numpy array (BGR format)
        
    Raises:
        HTTPException: If download or decoding fails
    """
    try:
        logger.info(f"Downloading image from: {url}")
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        
        # Decode image
        image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Failed to decode image")
        
        logger.info(f"Successfully downloaded image: {image.shape}")
        return image
        
    except requests.exceptions.Timeout:
        raise HTTPException(status_code=408, detail="Request timeout while downloading image")
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Failed to download image: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")
        