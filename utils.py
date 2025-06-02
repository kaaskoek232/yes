import os
import cv2
import numpy as np
import logging
from typing import Tuple, Optional, Dict, List
import time
import importlib.util
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("FaceSwapPro")

def check_dependencies() -> Dict[str, bool]:
    """Check if all required dependencies are installed"""
    dependencies = {
        'numpy': False,
        'cv2': False,
        'onnxruntime': False,
        'insightface': False
    }
    
    # Check each dependency
    for dep in dependencies:
        try:
            importlib.import_module(dep)
            dependencies[dep] = True
        except ImportError:
            dependencies[dep] = False
            
    return dependencies
    
def download_models(model_dir: str) -> bool:
    """
    Download required models if not present.
    
    Args:
        model_dir: Directory to store models
        
    Returns:
        True if all models are available
    """
    os.makedirs(model_dir, exist_ok=True)
    
    # Define model URLs and paths
    models = {
        'inswapper_128.onnx': 'https://github.com/facefusion/facefusion-assets/releases/download/models/inswapper_128.onnx',
        'rf_detr_model.onnx': 'https://huggingface.co/cnstark/rt-detr-resnet/resolve/main/rt-detr-resnet50-1x-640x640.onnx',
        # Add more model URLs as needed
    }
    
    all_available = True
    
    # Check each model
    for model_name, model_url in models.items():
        model_path = os.path.join(model_dir, model_name)
        
        if not os.path.exists(model_path):
            logger.info(f"Downloading {model_name}...")
            try:
                # Import only if needed
                import requests
                
                # Download the model
                response = requests.get(model_url, stream=True)
                response.raise_for_status()
                
                # Get file size for progress
                total_size = int(response.headers.get('content-length', 0))
                
                with open(model_path, 'wb') as f:
                    downloaded = 0
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            # Show progress
                            done = int(50 * downloaded / total_size)
                            print(f"\r[{'=' * done}{' ' * (50 - done)}] {downloaded}/{total_size} bytes", end='')
                    print()
                
                logger.info(f"Downloaded {model_name} successfully")
            except Exception as e:
                logger.error(f"Failed to download {model_name}: {e}")
                all_available = False
                
    return all_available
    
def resize_keep_aspect(image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """
    Resize image keeping aspect ratio and padding if necessary.
    
    Args:
        image: Input image
        target_size: Target (width, height)
        
    Returns:
        Resized image
    """
    h, w = image.shape[:2]
    target_w, target_h = target_size
    
    # Calculate scale to maintain aspect ratio
    scale = min(target_w / w, target_h / h)
    
    # New size
    new_w, new_h = int(w * scale), int(h * scale)
    
    # Resize image
    resized = cv2.resize(image, (new_w, new_h))
    
    # Create black canvas
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    
    # Calculate offsets
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    
    # Place resized image on canvas
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    return canvas
    
def is_image_file(path: str) -> bool:
    """Check if a file is an image"""
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
    return os.path.isfile(path) and any(path.lower().endswith(ext) for ext in image_extensions)
    
def is_video_file(path: str) -> bool:
    """Check if a file is a video"""
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']
    return os.path.isfile(path) and any(path.lower().endswith(ext) for ext in video_extensions)
    
def create_thumbnail(image: np.ndarray, size: Tuple[int, int] = (192, 192)) -> np.ndarray:
    """Create a thumbnail from an image"""
    return resize_keep_aspect(image, size)