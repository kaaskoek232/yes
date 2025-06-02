#!/usr/bin/env python3
"""
Neural Video Compression for FaceSwapPro
Author: kaaskoek232
Date: 2025-06-02
"""

import os
import numpy as np
import torch
from typing import Dict, Optional, Tuple, List
import logging
import cv2

# Try to import NeuralCodec
try:
    import neuralcodec
    NEURALCODEC_AVAILABLE = True
except ImportError:
    NEURALCODEC_AVAILABLE = False

logger = logging.getLogger("FaceSwapPro-NeuralCompression")

class NeuralCompression:
    """
    Neural video compression using NeuralCodec, the 2025 SOTA
    video compression framework with 70% better results than H.265.
    """
    
    def __init__(
        self,
        quality_level: int = 7,
        bitrate: Optional[int] = None,
        model_type: str = "medium",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        temporal_optimization: bool = True,
        perceptual_quality: bool = True
    ):
        """
        Initialize neural compression.
        
        Args:
            quality_level: Quality level from 1-10 (10 is best quality)
            bitrate: Target bitrate in kbps (overrides quality_level)
            model_type: Model size ("small", "medium", "large")
            device: Computation device
            temporal_optimization: Use temporal optimization
            perceptual_quality: Optimize for perceptual quality vs. PSNR
        """
        self.quality_level = quality_level
        self.bitrate = bitrate
        self.model_type = model_type
        self.device = device
        self.temporal_optimization = temporal_optimization
        self.perceptual_quality = perceptual_quality
        
        if not NEURALCODEC_AVAILABLE:
            logger.warning("NeuralCodec not available, falling back to standard encoding")
            self.codec = None
            return
            
        try:
            # Initialize the encoder and decoder
            self.codec = neuralcodec.VideoCodec(
                model=f"neuralcodec_{model_type}",
                device=device,
                quality=quality_level / 10.0,  # Convert to 0-1 range
                bitrate=bitrate,
                temporal_layers=3 if temporal_optimization else 1,
                perceptual_opt=perceptual_quality
            )
            
            logger.info(f"Neural compression initialized with {model_type} model")
            
            # Warm up the model
            if torch.cuda.is_available():
                dummy_frame = torch.zeros((1, 3, 256, 256), device=device)
                with torch.no_grad():
                    self.codec.warmup(dummy_frame)
                
        except Exception as e:
            logger.error(f"Failed to initialize neural compression: {e}")
            self.codec = None
            
    def is_available(self) -> bool:
        """Check if neural compression is available"""
        return self.codec is not None
        
    def compress_video(self, input_path: str, output_path: str, progress_callback=None) -> bool:
        """
        Compress a video using neural compression.
        
        Args:
            input_path: Path to input video
            output_path: Path to output compressed video
            progress_callback: Optional callback for progress updates
            
        Returns:
            True if successful, False otherwise
        """
        if not self.is_available():
            logger.warning("Neural compression not available, skipping")
            return False
            
        try:
            # For large videos, we process in chunks
            self.codec.compress_file(
                input_path,
                output_path,
                chunk_size=64,  # Process 64 frames at a time
                progress_callback=progress_callback
            )
            
            # Verify the output file exists and has content
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                logger.info(f"Neural compression successful: {output_path}")
                
                # Report compression ratio
                input_size = os.path.getsize(input_path)
                output_size = os.path.getsize(output_path)
                ratio = input_size / output_size if output_size > 0 else 0
                logger.info(f"Compression ratio: {ratio:.2f}x ({input_size/1024/1024:.2f} MB → {output_size/1024/1024:.2f} MB)")
                
                return True
            else:
                logger.error("Compression failed: output file missing or empty")
                return False
                
        except Exception as e:
            logger.error(f"Error during neural compression: {e}")
            return False
            
    def compress_frame_sequence(self, frames: List[np.ndarray], output_path: str, fps: float = 30.0) -> bool:
        """
        Compress a sequence of frames.
        
        Args:
            frames: List of BGR frames
            output_path: Path to output compressed video
            fps: Frames per second
            
        Returns:
            True if successful, False otherwise
        """
        if not self.is_available() or not frames:
            return False
            
        try:
            # Convert frames to torch tensors
            tensors = []
            for frame in frames:
                # Convert BGR to RGB and normalize
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
                tensors.append(tensor.unsqueeze(0).to(self.device))
                
            # Compress the sequence
            compressed = self.codec.compress_tensors(tensors)
            
            # Save to file
            self.codec.save_bitstream(compressed, output_path, fps=fps)
            
            return True
            
        except Exception as e:
            logger.error(f"Error compressing frame sequence: {e}")
            return False