#!/usr/bin/env python3
"""
FaceSwapPro - Command Line Interface
Author: kaaskoek232
Date: 2025-06-02 19:08:43
"""

import os
import sys
import argparse
import time
import json
import cv2
import numpy as np
import logging
from tqdm import tqdm
from pathlib import Path

# Add project directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from faceswap_pro import (
    EnhancedPixelBoost, ModelConfig, ExpressionTransfer,
    OcclusionHandler, FaceSwapProcessor, CUDA_AVAILABLE
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("FaceSwapPro-CLI")

# Constants
CONFIG_FILE = "config.json"
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def load_config():
    """Load configuration from file"""
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), CONFIG_FILE)
    
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
    
    # Default configuration
    return {
        "resolution": 512,
        "swap_all": True,
        "use_gpu": CUDA_AVAILABLE,
        "face_swap_model": "inswapper_128",
        "tile_blending": True,
        "adaptive_tiling": True,
        "detail_enhancement": True,
        "perceptual_correction": True,
        "overlap_percent": 0.15,
        "expression_method": "NeuroMotion",
        "enable_expression": True,
        "occlusion_method": "MATCHformer",
        "enable_occlusion": True,
        "enable_motion_tracking": True,
        "enable_temporal_smoothing": True,
        "processing_threads": 4
    }


def init_processor(config, model_path=None, use_gpu=None):
    """Initialize the face swapping processor"""
    # Override config with parameters if provided
    if use_gpu is not None:
        config["use_gpu"] = use_gpu
        
    # Use GPU if available and enabled
    execution_provider = 'cuda' if CUDA_AVAILABLE and config.get("use_gpu", True) else 'cpu'
    
    # Get face swap model path
    if not model_path:
        model_name = config.get("face_swap_model", "inswapper_128")
        model_path = os.path.join(MODELS_DIR, f"{model_name}.onnx")
    
    # Check if model exists
    if not os.path.exists(model_path):
        logger.warning(f"Model not found: {model_path}, will attempt to download")
        # In a real implementation, you'd add download logic here
        
    # Configure the model
    if 'inswapper' in model_path.lower():
        model_config = ModelConfig(
            path=model_path,
            size=(128, 128),
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            model_type='inswapper',
            fp16_support='fp16' in model_path.lower()
        )
    elif 'simswap' in model_path.lower():
        model_config = ModelConfig(
            path=model_path,
            size=(256, 256) if '256' in model_path else (512, 512),
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            model_type='simswap',
            requires_embedding_conversion=True
        )
    else:
        # Generic configuration
        model_config = ModelConfig(
            path=model_path,
            size=(256, 256),
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
        
    # Initialize expression transfer if enabled
    expression_transfer = None
    if config.get("enable_expression", True):
        expression_transfer = ExpressionTransfer(
            method=config.get("expression_method", "neuromotion").lower(),
            execution_provider=execution_provider,
            strength=config.get("expression_strength", 0.7),
            preserve_identity=config.get("preserve_identity", 0.8),
            enhance_details=config.get("detail_enhancement", True)
        )
        
    # Initialize occlusion handler if enabled
    occlusion_handler = None
    if config.get("enable_occlusion", True):
        occlusion_handler = OcclusionHandler(
            method=config.get("occlusion_method", "matchformer").lower(),
            threshold=config.get("occlusion_threshold", 0.5),
            refinement=config.get("occlusion_refinement", True),
            temporal_smoothing=config.get("temporal_smoothing", True)
        )
        
    # Initialize the processor
    processor = FaceSwapProcessor(
        face_detector_model='yolov12-face',
        face_recognizer_model='arcface_2025',
        face_swap_model=model_path,
        execution_provider=execution_provider,
        gpu_id=0,
        detection_threshold=0.5,
        recognition_threshold=0.7,
        target_resolution=int(config.get("resolution", 512)),
        enable_motion_tracking=config.get("enable_motion_tracking", True),
        enable_temporal_smoothing=config.get("enable_temporal_smoothing", True)
    )
    
    return processor


def process_image(processor, source_path, target_path, output_path, config):
    """Process a single image"""
    logger.info(f"Processing image: {os.path.basename(target_path)}")
    
    try:
        # Load images
        source_image = cv2.imread(source_path)
        target_image = cv2.imread(target_path)
        
        if source_image is None:
            logger.error(f"Failed to load source image: {source_path}")
            return False
            
        if target_image is None:
            logger.error(f"Failed to load target image: {target_path}")
            return False
            
        # Add source face to processor
        processor.add_face_to_database('source', source_image)
        source_embedding = processor.face_database.get('source')
        
        # Detect faces in target image
        target_faces = processor.detect_faces(target_image)
        
        if not target_faces:
            logger.warning("No faces detected in target image")
            return False
            
        logger.info(f"Detected {len(target_faces)} faces in target image")
        
        # Process image
        result = processor.process_frame(
            frame=target_image,
            source_embedding=source_embedding,
            target_faces=target_faces,
            swap_all=config.get("swap_all", True),
            resolution=config.get("resolution", 512)
        )
        
        # Save result
        cv2.imwrite(output_path, result)
        logger.info(f"Result saved to: {output_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        return False


def process_video(processor, source_path, video_path, output_path, config):
    """Process a video"""
    logger.info(f"Processing video: {os.path.basename(video_path)}")
    
    try:
        # Load source image
        source_image = cv2.imread(source_path)
        
        if source_image is None:
            logger.error(f"Failed to load source image: {source_path}")
            return False
        
        # Setup progress callback for CLI
        def progress_callback(current, total):
            # Updates are handled by tqdm wrapper
            pass
            
        # Add source face to processor
        processor.add_face_to_database('source', source_image)
        
        # Process video
        success = processor.process_video(
            source_face=source_image,
            target_video=video_path,
            output_path=output_path,
            swap_all=config.get("swap_all", True),
            resolution=config.get("resolution", 512),
            progress_callback=progress_callback
        )
        
        if success:
            logger.info(f"Video processing complete. Result saved to: {output_path}")
        else:
            logger.error("Video processing failed")
            
        return success
        
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        return False


def main():
    """Command line interface main function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="FaceSwapPro Command Line Interface")
    parser.add_argument("--source", required=True, help="Source face image path")
    parser.add_argument("--target", required=True, help="Target image or video path")
    parser.add_argument("--output", help="Output path (default: auto-generated)")
    parser.add_argument("--resolution", type=int, help="Processing resolution (default: 512)")
    parser.add_argument("--model", help="Path to face swap model (default: inswapper_128)")
    parser.add_argument("--cpu", action="store_true", help="Force CPU processing")
    parser.add_argument("--no-expression", action="store_true", help="Disable expression transfer")
    parser.add_argument("--no-occlusion", action="store_true", help="Disable occlusion handling")
    parser.add_argument("--selected-face", type=int, help="Only swap the specified face index")
    
    args = parser.parse_args()
    
    # Validate source and target paths
    if not os.path.exists(args.source):
        logger.error(f"Source image not found: {args.source}")
        return 1
        
    if not os.path.exists(args.target):
        logger.error(f"Target file not found: {args.target}")
        return 1
        
    # Load configuration
    config = load_config()
    
    # Override configuration with command line arguments
    if args.resolution:
        config["resolution"] = args.resolution
        
    if args.cpu:
        config["use_gpu"] = False
        
    if args.no_expression:
        config["enable_expression"] = False
        
    if args.no_occlusion:
        config["enable_occlusion"] = False
        
    if args.selected_face is not None:
        config["swap_all"] = False
        config["target_ids"] = [args.selected_face]
        
    # Initialize processor
    processor = init_processor(config, args.model, not args.cpu)
    
    # Determine if target is image or video
    is_video = args.target.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))
    
    # Determine output path if not provided
    if not args.output:
        filename, ext = os.path.splitext(args.target)
        if is_video:
            args.output = f"{filename}_swapped{ext}"
        else:
            args.output = f"{filename}_swapped.png"
    
    # Process based on target type
    start_time = time.time()
    
    if is_video:
        success = process_video(processor, args.source, args.target, args.output, config)
    else:
        success = process_image(processor, args.source, args.target, args.output, config)
    
    elapsed = time.time() - start_time
    logger.info(f"Processing completed in {elapsed:.2f}s")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())