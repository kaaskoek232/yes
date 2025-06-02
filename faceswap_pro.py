#!/usr/bin/env python3
"""
FaceSwapPro - Advanced real-time face swapping with Enhanced PixelBoost
Author: kaaskoek232
Date: 2025-06-02
"""

import os
import cv2
import numpy as np
import torch
import onnxruntime as ort
from typing import List, Dict, Tuple, Optional, Union, Any
import time
from pathlib import Path
import threading
from queue import Queue
import argparse
import matplotlib.pyplot as plt
from dataclasses import dataclass
import logging
import copy
from concurrent.futures import ThreadPoolExecutor
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("FaceSwapPro")

# YOLO v12 integration (2025 version)
try:
    import yolov12
except ImportError:
    logger.warning("YOLO v12 not found, falling back to OpenCV detection")
    yolov12 = None

# Try to import CUDA optimized libraries
try:
    from torch.cuda import amp
    TORCH_AMP_AVAILABLE = True
    logger.info("PyTorch AMP available for mixed precision acceleration")
except ImportError:
    TORCH_AMP_AVAILABLE = False
    logger.warning("PyTorch AMP not available, will use FP32 precision")

# Check for TensorRT
try:
    import tensorrt as trt
    TENSORRT_AVAILABLE = True
    logger.info("TensorRT available for maximum inference speed")
except ImportError:
    TENSORRT_AVAILABLE = False
    logger.warning("TensorRT not available, using standard ONNX Runtime")
    
# Try to import NeuroMotion (successor to LivePortrait in 2025)
try:
    import neuromotion
    NEUROMOTION_AVAILABLE = True
    logger.info("NeuroMotion 2.3 available for expression transfer")
except ImportError:
    NEUROMOTION_AVAILABLE = False
    logger.warning("NeuroMotion not available, falling back to classical expression transfer")

# Try to import MATCHformer for superior occlusion handling
try:
    import matchformer
    MATCHFORMER_AVAILABLE = True
    logger.info("MATCHformer available for advanced occlusion handling")
except ImportError:
    MATCHFORMER_AVAILABLE = False
    logger.warning("MATCHformer not available, using classical occlusion handling")
    
# GPU acceleration check
CUDA_AVAILABLE = torch.cuda.is_available()
if CUDA_AVAILABLE:
    logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
    logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    logger.warning("CUDA not available, using CPU")

@dataclass
class ModelConfig:
    """Configuration for a face swapping model"""
    path: str
    size: Tuple[int, int]
    mean: List[float]
    std: List[float]
    model_type: str = "inswapper"
    requires_embedding_conversion: bool = False
    fp16_support: bool = True
    tensorrt_compatible: bool = False
    quantized: bool = False

class EnhancedPixelBoost:
    """
    Advanced PixelBoost implementation with:
    - Dynamic tiling with feature-aware importance
    - Advanced edge blending
    - Neural detail enhancement
    - Perceptual color correction
    - Hardware optimization (TensorRT, FP16, CUDA)
    """
    
    def __init__(
        self,
        model_config: ModelConfig,
        execution_provider: str = 'cuda',
        tile_blending: bool = True,
        adaptive_tiling: bool = True,
        detail_enhancement: bool = True,
        perceptual_correction: bool = True,
        max_workers: int = 4,
        cache_size: int = 10,
        use_mixed_precision: bool = True,
        use_tensorrt: bool = False
    ):
        """
        Initialize the enhanced PixelBoost processor.
        
        Args:
            model_config: Configuration for the face swap model
            execution_provider: ONNX execution provider ('cuda', 'tensorrt', 'directml', 'cpu')
            tile_blending: Enable tile edge blending for seamless results
            adaptive_tiling: Adapt tiling based on face features
            detail_enhancement: Apply neural detail enhancement to output
            perceptual_correction: Apply perceptual color correction
            max_workers: Maximum number of parallel workers for processing
            cache_size: Maximum number of embeddings to cache
            use_mixed_precision: Enable mixed precision (FP16) for faster inference
            use_tensorrt: Use TensorRT for optimized inference if available
        """
        self.model_config = model_config
        self.tile_blending = tile_blending
        self.adaptive_tiling = adaptive_tiling
        self.detail_enhancement = detail_enhancement
        self.perceptual_correction = perceptual_correction
        self.max_workers = max_workers
        self.use_mixed_precision = use_mixed_precision and TORCH_AMP_AVAILABLE and model_config.fp16_support
        
        # TensorRT usage - only if available, compatible, and requested
        self.use_tensorrt = (use_tensorrt and TENSORRT_AVAILABLE and 
                            model_config.tensorrt_compatible and 
                            execution_provider == 'cuda')
        
        # Initialize embedding cache with LRU mechanism
        self._embedding_cache = {}
        self._embedding_cache_keys = []
        self._cache_size = cache_size
        
        # Initialize hardware acceleration
        self._setup_execution_provider(execution_provider)
        
        # Load the model
        logger.info(f"Loading model: {model_config.path}")
        self._load_model()
        
        # Advanced settings
        self.overlap_percent = 0.15  # Overlap between tiles
        self.edge_fade_percent = 0.1  # Edge fade for blending
        self.detail_boost_factor = 1.4  # Detail enhancement factor
        
        # Initialize detail enhancement network if enabled
        if self.detail_enhancement:
            self._init_detail_enhancer()
            
        # Initialize ThreadPoolExecutor for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        logger.info("Enhanced PixelBoost initialized successfully")
        
    def _setup_execution_provider(self, execution_provider: str):
        """Configure execution provider for optimal performance"""
        self.providers = []
        
        # Check available providers
        available = ort.get_available_providers()
        logger.info(f"Available ONNX Runtime providers: {available}")
        
        if execution_provider == 'cuda' and 'CUDAExecutionProvider' in available:
            # Optimized CUDA settings for RTX 40 series
            provider_options = [{
                'device_id': 0,
                'arena_extend_strategy': 'kNextPowerOfTwo',
                'gpu_mem_limit': 4 * 1024 * 1024 * 1024,  # 4GB
                'cudnn_conv_algo_search': 'EXHAUSTIVE',
                'cuda_ep_optimization_level': 3,  # Maximum opt level for 2025 GPUs
            }]
            
            self.providers = ['CUDAExecutionProvider']
            self.provider_options = provider_options
            logger.info("Using CUDA acceleration with optimized settings")
            
        elif execution_provider == 'directml' and 'DirectMLExecutionProvider' in available:
            self.providers = ['DirectMLExecutionProvider']
            self.provider_options = [{'device_id': 0}]
            logger.info("Using DirectML acceleration")
            
        elif execution_provider == 'tensorrt' and 'TensorrtExecutionProvider' in available:
            self.providers = ['TensorrtExecutionProvider']
            self.provider_options = [{
                'device_id': 0,
                'trt_max_workspace_size': 2 * 1024 * 1024 * 1024,
                'trt_fp16_enable': self.use_mixed_precision,
            }]
            logger.info(f"Using TensorRT acceleration with FP16={self.use_mixed_precision}")
            
        else:
            self.providers = ['CPUExecutionProvider']
            self.provider_options = [{
                'intra_op_num_threads': min(16, os.cpu_count() or 1),
                'execution_mode': 0,
            }]
            logger.info("Using CPU execution")
            
    def _load_model(self):
        """Load and optimize the ONNX model"""
        try:
            # Set session options for better performance
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.enable_profiling = False
            sess_options.enable_mem_pattern = True
            sess_options.enable_cpu_mem_arena = True
            
            # Set execution mode based on hardware
            if self.providers[0] == 'CPUExecutionProvider':
                sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            else:
                sess_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
            
            # For TensorRT optimization
            if self.use_tensorrt:
                import tensorrt as trt
                sess_options.add_session_config_entry('session.intra_op_thread_affinities', '0:5')
                logger.info("Applying TensorRT optimization")
                
            # Load the model
            model_path = self.model_config.path
            self.session = ort.InferenceSession(
                model_path,
                sess_options=sess_options,
                providers=self.providers,
                provider_options=self.provider_options
            )
            
            # Get input and output details
            self.input_names = [input.name for input in self.session.get_inputs()]
            self.output_names = [output.name for output in self.session.get_outputs()]
            
            logger.info(f"Model loaded successfully. Inputs: {self.input_names}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise

    def _init_detail_enhancer(self):
        """Initialize the neural detail enhancer network (2025 version)"""
        if not CUDA_AVAILABLE:
            logger.warning("Neural detail enhancer requires CUDA, falling back to classical method")
            self.neural_detail_enhancer = None
            return
            
        try:
            # Use lightweight enhancer model
            from facefusion_enhancer import NeuralDetailEnhancer
            self.neural_detail_enhancer = NeuralDetailEnhancer(device='cuda')
            logger.info("Neural detail enhancer initialized")
        except ImportError:
            logger.warning("Neural detail enhancer not available, using classical method")
            self.neural_detail_enhancer = None
        
    def implode_pixel_boost(self, frame: np.ndarray, boost_factor: int) -> List[Dict]:
        """
        Split a high-resolution frame into tiles with overlap for processing.
        
        Args:
            frame: Input frame at high resolution
            boost_factor: Boost factor (e.g., 8 for 1024/128)
            
        Returns:
            List of tile dictionaries with position info
        """
        # Get dimensions
        h, w = frame.shape[:2]
        model_h, model_w = self.model_config.size
        
        # Calculate overlap
        overlap_h = int(model_h * self.overlap_percent) if self.tile_blending else 0
        overlap_w = int(model_w * self.overlap_percent) if self.tile_blending else 0
        
        # Calculate tile size with boost factor
        tile_h = h // boost_factor
        tile_w = w // boost_factor
        
        # Analyze face features for adaptive tiling if enabled
        importance_map = None
        if self.adaptive_tiling:
            importance_map = self._generate_importance_map(frame, (h, w))
            
        tiles = []
        for i in range(boost_factor):
            for j in range(boost_factor):
                # Calculate tile positions
                y_start = i * tile_h
                y_end = (i + 1) * tile_h if i < boost_factor - 1 else h
                x_start = j * tile_w
                x_end = (j + 1) * tile_w if j < boost_factor - 1 else w
                
                # Add overlap
                y_start_overlap = max(0, y_start - overlap_h)
                y_end_overlap = min(h, y_end + overlap_h)
                x_start_overlap = max(0, x_start - overlap_w)
                x_end_overlap = min(w, x_end + overlap_w)
                
                # Extract tile with overlap
                tile = frame[y_start_overlap:y_end_overlap, x_start_overlap:x_end_overlap]
                
                # Get importance score if using adaptive tiling
                importance = 1.0
                if importance_map is not None:
                    # Calculate average importance in this tile
                    tile_importance = importance_map[y_start:y_end, x_start:x_end]
                    importance = np.mean(tile_importance)
                
                # Resize to model input size
                tile_resized = cv2.resize(tile, self.model_config.size)
                
                tiles.append({
                    'tile': tile_resized,
                    'position': (i, j),
                    'importance': importance,
                    'original_size': (y_end_overlap - y_start_overlap, 
                                     x_end_overlap - x_start_overlap),
                    'original_coords': (y_start_overlap, y_end_overlap,
                                       x_start_overlap, x_end_overlap),
                    'inner_coords': (y_start - y_start_overlap, 
                                    y_end - y_start_overlap,
                                    x_start - x_start_overlap, 
                                    x_end - x_start_overlap)
                })
                
        # If using adaptive tiling, sort tiles by importance
        if self.adaptive_tiling:
            # Process high importance tiles first
            tiles.sort(key=lambda x: x['importance'], reverse=True)
                
        return tiles
        
    def _generate_importance_map(self, frame: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
        """
        Generate an importance map for adaptive tiling based on facial features.
        Higher values indicate more important regions (eyes, mouth, etc.)
        
        Args:
            frame: Input frame
            size: Output size for importance map
            
        Returns:
            Importance map with values 0-1
        """
        h, w = size
        
        # Basic importance map - center is more important
        y, x = np.mgrid[0:h, 0:w].astype(np.float32)
        y = 1.0 - 2.0 * np.abs(y / (h - 1) - 0.5)  # 1.0 at center, 0.0 at edges
        x = 1.0 - 2.0 * np.abs(x / (w - 1) - 0.5)  # 1.0 at center, 0.0 at edges
        importance = (y * x) ** 0.5  # Radial falloff
        
        # If YOLO v12 available, use it for facial feature detection
        if yolov12 is not None:
            try:
                # Detect facial features
                detector = yolov12.FaceFeatureDetector()
                features = detector.detect(frame)
                
                # Create feature importance mask
                feature_mask = np.zeros((h, w), dtype=np.float32)
                
                # Add importance to detected features
                for feature in features:
                    if feature['class'] == 'eye':
                        # Eyes are very important
                        self._add_feature_importance(feature_mask, feature['bbox'], weight=2.0)
                    elif feature['class'] == 'mouth':
                        # Mouth is important
                        self._add_feature_importance(feature_mask, feature['bbox'], weight=1.5)
                    elif feature['class'] == 'nose':
                        # Nose is somewhat important
                        self._add_feature_importance(feature_mask, feature['bbox'], weight=1.2)
                    else:
                        # Other features
                        self._add_feature_importance(feature_mask, feature['bbox'], weight=1.0)
                        
                # Normalize feature mask
                if np.max(feature_mask) > 0:
                    feature_mask = feature_mask / np.max(feature_mask)
                    
                    # Combine with base importance
                    importance = 0.4 * importance + 0.6 * feature_mask
            except Exception as e:
                logger.warning(f"Error in feature detection: {e}, using basic importance map")
                
        return importance
        
    def _add_feature_importance(self, mask: np.ndarray, bbox: Tuple[int, int, int, int], weight: float = 1.0):
        """Add Gaussian importance to a feature region"""
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        h, w = mask.shape
        
        # Ensure coordinates are within bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            return
            
        # Create a Gaussian blob centered on the feature
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        sigma_x = (x2 - x1) // 4
        sigma_y = (y2 - y1) // 4
        
        # Limit sigma to reasonable values
        sigma_x = max(2, min(30, sigma_x))
        sigma_y = max(2, min(30, sigma_y))
        
        # Create Gaussian kernel
        y, x = np.mgrid[y1:y2, x1:x2]
        gaussian = weight * np.exp(
            -((x - center_x)**2/(2*sigma_x**2) + (y - center_y)**2/(2*sigma_y**2))
        )
        
        # Add to mask
        mask[y1:y2, x1:x2] += gaussian
        
    def explode_pixel_boost(self, processed_tiles: List[Dict], output_size: Tuple[int, int]) -> np.ndarray:
        """
        Advanced tile recombination with seamless blending.
        
        Args:
            processed_tiles: List of processed tile dictionaries
            output_size: Size of the output frame (h, w)
            
        Returns:
            Reconstructed full frame
        """
        # Create output frame and weight map
        result_frame = np.zeros((output_size[0], output_size[1], 3), dtype=np.float32)
        weight_map = np.zeros((output_size[0], output_size[1]), dtype=np.float32)
        
        # Tile contribution vectors for advanced blending
        offsets = []
        
        # Process each tile
        for tile_data in processed_tiles:
            # Get tile data
            processed_tile = tile_data['processed_tile']
            y1, y2, x1, x2 = tile_data['original_coords']
            
            # Resize back to original size
            tile_resized = cv2.resize(processed_tile, (x2-x1, y2-y1))
            
            # Create weight mask for this tile with smooth falloff
            if self.tile_blending:
                h, w = y2-y1, x2-x1
                
                # Create 2D Gaussian weight mask
                weight_mask = self._create_advanced_weight_mask((h, w))
                
                # Apply tile importance if available
                if 'importance' in tile_data:
                    # Scale weight by importance (higher importance = more contribution)
                    importance_factor = 0.5 + 0.5 * tile_data['importance']  # Map to 0.5-1.0 range
                    weight_mask = weight_mask * importance_factor
                
                # Apply weighted blending
                for c in range(3):
                    result_frame[y1:y2, x1:x2, c] += tile_resized[:,:,c] * weight_mask
                
                # Add to weight map
                weight_map[y1:y2, x1:x2] += weight_mask
                
                # Store offset for Poisson blending
                center_y = (y1 + y2) // 2
                center_x = (x1 + x2) // 2
                offsets.append((center_y, center_x))
                
            else:
                # For non-blending mode, just place inner tiles directly
                iy1, iy2, ix1, ix2 = tile_data['inner_coords']
                inner_tile = tile_resized[iy1:iy2, ix1:ix2]
                
                # Put inner tile directly in the result frame
                y_inner_start = y1 + iy1
                y_inner_end = y1 + iy2
                x_inner_start = x1 + ix1
                x_inner_end = x1 + ix2
                
                result_frame[y_inner_start:y_inner_end, x_inner_start:x_inner_end] = inner_tile
                weight_map[y_inner_start:y_inner_end, x_inner_start:x_inner_end] = 1.0
                
        # Normalize by weight map to avoid seams
        valid_mask = weight_map > 0
        for c in range(3):
            channel = result_frame[:,:,c]
            channel[valid_mask] /= weight_map[valid_mask]
            
        # Apply post-processing
        result_frame = self._post_process_frame(result_frame)
            
        return result_frame
        
    def _create_advanced_weight_mask(self, shape: Tuple[int, int]) -> np.ndarray:
        """
        Create an advanced weight mask for tile blending.
        Uses a combination of distance transform and edge awareness.
        
        Args:
            shape: Shape of the mask (h, w)
            
        Returns:
            Weight mask
        """
        h, w = shape
        
        # Create base circular mask (higher weight in center, lower at edges)
        y, x = np.mgrid[0:h, 0:w].astype(np.float32)
        y = 2.0 * (y / (h - 1) - 0.5)  # -1 to 1
        x = 2.0 * (x / (w - 1) - 0.5)  # -1 to 1
        
        # Squared distance from center
        dist = x**2 + y**2
        
        # Apply non-linear mapping for smoother transitions at edges
        weight = np.clip(1.0 - dist, 0, 1) ** 1.5
        
        return weight
        
    def _post_process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply post-processing to the reconstructed frame.
        
        Args:
            frame: Input frame
            
        Returns:
            Post-processed frame
        """
        # Ensure frame is in valid range
        frame_norm = np.clip(frame, 0, 255).astype(np.uint8)
        
        # Apply detail enhancement if enabled
        if self.detail_enhancement:
            if self.neural_detail_enhancer is not None:
                # Use neural enhancer
                frame_norm = self.neural_detail_enhancer.enhance(frame_norm)
            else:
                # Use classical detail enhancement
                frame_norm = self._enhance_details_classical(frame_norm)
            
        # Apply perceptual color correction if enabled
        if self.perceptual_correction:
            frame_norm = self._apply_perceptual_correction(frame_norm)
            
        return frame_norm
        
    def _enhance_details_classical(self, frame: np.ndarray) -> np.ndarray:
        """
        Enhanced detail enhancement using advanced filters.
        
        Args:
            frame: Input frame (uint8)
            
        Returns:
            Detail-enhanced frame
        """
        # Convert to LAB color space for better perceptual processing
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply adaptive CLAHE to luminance channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l)
        
        # Use guided filter for edge-preserving smoothing (better than bilateral)
        l_smooth = cv2.ximgproc.guidedFilter(l_enhanced, l_enhanced, 5, 0.1)
        
        # Extract detail layer (high-frequency components)
        l_detail = cv2.subtract(l_enhanced, l_smooth)
        
        # Boost details with non-linear mapping
        # This preserves small details while preventing over-enhancement of noise
        detail_factor = self.detail_boost_factor
        l_detail_boosted = np.sign(l_detail) * np.power(np.abs(l_detail) / 10.0, 0.8) * 10.0 * detail_factor
        
        # Combine smoothed base with boosted details
        l_final = np.clip(l_smooth + l_detail_boosted, 0, 255).astype(np.uint8)
        
        # Merge channels
        enhanced_lab = cv2.merge([l_final, a, b])
        enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        # Apply subtle unsharp masking for final touch
        gaussian = cv2.GaussianBlur(enhanced_bgr, (0, 0), 1.0)
        unsharp_mask = cv2.addWeighted(enhanced_bgr, 1.5, gaussian, -0.5, 0)
        
        return unsharp_mask
        
    def _apply_perceptual_correction(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply perceptual color correction to improve realism.
        Adjusts contrast, saturation, and color balance.
        
        Args:
            frame: Input BGR frame
            
        Returns:
            Color-corrected frame
        """
        # Convert to HSV for easier color manipulation
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
        h, s, v = cv2.split(hsv)
        
        # Apply Adaptive Contrast
        # First compute local luminance statistics
        mean_v = cv2.GaussianBlur(v, (0, 0), 16.0)
        std_v = np.sqrt(cv2.GaussianBlur(v**2, (0, 0), 16.0) - mean_v**2)
        
        # Adaptive contrast factor (lower in low-contrast areas)
        adapt_factor = 1.0 + 0.3 * np.clip(std_v / (mean_v + 10), 0, 1)
        
        # Apply contrast correction
        v_contrast = np.clip(mean_v + (v - mean_v) * adapt_factor, 0, 255)
        
        # Subtle saturation adjustment - natural face colors
        # Reduce saturation for skin tones
        is_skin = self._detect_skin_tones(frame)
        s_adjusted = s.copy()
        s_adjusted[is_skin] *= 0.85  # Reduce skin saturation slightly
        
        # Merge channels
        hsv_adjusted = cv2.merge([h, s_adjusted, v_contrast])
        bgr_adjusted = cv2.cvtColor(hsv_adjusted.astype(np.uint8), cv2.COLOR_HSV2BGR)
        
        return bgr_adjusted
        
    def _detect_skin_tones(self, frame: np.ndarray) -> np.ndarray:
        """
        Detect skin tones in an image.
        
        Args:
            frame: BGR image
            
        Returns:
            Binary mask where True indicates skin pixels
        """
        # Convert to YCrCb space which works better for skin detection
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        
        # Define skin color range in YCrCb
        lower = np.array([0, 135, 85], dtype=np.uint8)
        upper = np.array([255, 180, 135], dtype=np.uint8)
        
        # Create skin mask
        skin_mask = cv2.inRange(ycrcb, lower, upper)
        
        # Apply morphological operations to clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
        
        return skin_mask > 0
        
    def prepare_source_embedding(self, embedding: np.ndarray, source_id: Optional[str] = None) -> np.ndarray:
        """
        Prepare source face embedding with caching.
        
        Args:
            embedding: Source face embedding
            source_id: Optional ID for caching
            
        Returns:
            Prepared embedding
        """
        # Check cache first if ID provided
        if source_id and source_id in self._embedding_cache:
            return self._embedding_cache[source_id]
            
        # Process based on model type
        if self.model_config.model_type == 'inswapper':
            # Format for inswapper
            processed = embedding.reshape((1, -1)).astype(np.float32)
            
            # Cache if ID provided
            if source_id:
                self._add_to_cache(source_id, processed)
                
            return processed
            
        elif self.model_config.requires_embedding_conversion:
            # For models requiring embedding conversion
            # This would be implemented based on specific model needs
            pass
            
        # Default handling
        return embedding.reshape((1, -1)).astype(np.float32)
        
    def _add_to_cache(self, key: str, value: np.ndarray):
        """Add an embedding to the LRU cache"""
        # Add to cache
        self._embedding_cache[key] = value
        
        # Update LRU tracking
        if key in self._embedding_cache_keys:
            # Move to end (most recently used)
            self._embedding_cache_keys.remove(key)
        
        self._embedding_cache_keys.append(key)
        
        # Enforce cache size limit
        if len(self._embedding_cache_keys) > self._cache_size:
            # Remove least recently used
            oldest_key = self._embedding_cache_keys.pop(0)
            del self._embedding_cache[oldest_key]
            
    def prepare_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Prepare a frame for model inference.
        
        Args:
            frame: Input BGR frame
            
        Returns:
            Prepared frame tensor
        """
        # Convert BGR to RGB and normalize
        frame_rgb = frame[:, :, ::-1] / 255.0
        
        # Apply model-specific normalization
        frame_normalized = (frame_rgb - self.model_config.mean) / self.model_config.std
        
        # Change to NCHW format for model input (batch, channels, height, width)
        frame_transposed = frame_normalized.transpose(2, 0, 1)
        frame_expanded = np.expand_dims(frame_transposed, axis=0).astype(np.float32)
        
        # Convert to FP16 if using mixed precision
        if self.use_mixed_precision:
            frame_expanded = frame_expanded.astype(np.float16)
            
        return frame_expanded
        
    def denormalize_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Denormalize a processed frame back to BGR.
        
        Args:
            frame: Processed frame from model
            
        Returns:
            Denormalized BGR frame
        """
        # Convert back to float32 if needed
        if frame.dtype == np.float16:
            frame = frame.astype(np.float32)
            
        # Transpose from NCHW to HWC
        frame = frame.transpose(1, 2, 0)
        
        # Reverse normalization
        frame = frame * self.model_config.std + self.model_config.mean
        
        # Clip values and convert RGB to BGR
        frame = np.clip(frame, 0, 1) * 255
        frame = frame[:, :, ::-1]  # RGB to BGR
        
        return frame
        
    def _process_single_tile(self, tile_data: Dict, source_embedding: np.ndarray) -> Dict:
        """
        Process a single tile with the face swap model.
        
        Args:
            tile_data: Tile data dictionary
            source_embedding: Source face embedding
            
        Returns:
            Updated tile data with processed result
        """
        tile = tile_data['tile']
        
        # Prepare tile for model
        prepared_tile = self.prepare_frame(tile)
        
        # Prepare model inputs
        inputs = {}
        for input_name in self.input_names:
            if 'source' in input_name.lower() or 'id' in input_name.lower():
                inputs[input_name] = source_embedding
            elif 'target' in input_name.lower() or 'img' in input_name.lower():
                inputs[input_name] = prepared_tile
                
        # Run inference with error handling
        try:
            # Perform inference
            start_time = time.time()
            outputs = self.session.run(self.output_names, inputs)
            inference_time = time.time() - start_time
            
            # Get result (usually first output)
            output = outputs[0]
            
            # Process output based on shape
            if len(output.shape) == 4:  # NCHW format
                output = output[0]  # Remove batch dimension
            
            # Denormalize
            processed_tile = self.denormalize_frame(output)
            
            # Update tile data
            tile_data['processed_tile'] = processed_tile
            tile_data['inference_time'] = inference_time
            
        except Exception as e:
            logger.error(f"Inference error: {str(e)}")
            # In case of error, return original tile
            tile_data['processed_tile'] = tile
            tile_data['inference_time'] = 0
            tile_data['error'] = str(e)
            
        return tile_data

    def swap_face(self, source_embedding: np.ndarray, target_frame: np.ndarray, 
                 target_resolution: int = 1024, source_id: Optional[str] = None) -> np.ndarray:
        """
        Perform face swapping with enhanced pixel boost.
        
        Args:
            source_embedding: Source face embedding
            target_frame: Target frame containing face to swap
            target_resolution: Target resolution for face swap (must be multiple of model size)
            source_id: Optional ID for embedding caching
            
        Returns:
            Face-swapped frame
        """
        start_time = time.time()
        
        # Prepare source embedding
        source_embedding = self.prepare_source_embedding(source_embedding, source_id)
        
        # Get model size and calculate boost factor
        model_h, model_w = self.model_config.size
        
        # Ensure target resolution is multiple of model size
        if target_resolution % model_h != 0:
            logger.warning(f"Target resolution {target_resolution} is not multiple of model size {model_h}")
            # Round to nearest multiple
            target_resolution = round(target_resolution / model_h) * model_h
            
        boost_factor = target_resolution // model_h
        
        if boost_factor <= 0:
            raise ValueError(f"Target resolution {target_resolution} must be greater than model size {model_h}")
            
        # Resize target frame to target resolution if needed
        if target_frame.shape[0] != target_resolution or target_frame.shape[1] != target_resolution:
            target_frame = cv2.resize(target_frame, (target_resolution, target_resolution))
            
        # Split into tiles
        tiles = self.implode_pixel_boost(target_frame, boost_factor)
        
        # Process tiles
        processed_tiles = []
        
        # Use ThreadPoolExecutor for parallel processing if multiple workers
        if self.max_workers > 1:
            # Submit tasks
            futures = []
            for tile in tiles:
                future = self.executor.submit(self._process_single_tile, tile, source_embedding)
                futures.append(future)
                
            # Collect results as they complete
            for future in futures:
                try:
                    processed_tile = future.result()
                    processed_tiles.append(processed_tile)
                except Exception as e:
                    logger.error(f"Tile processing error: {str(e)}")
        else:
            # Process sequentially
            for tile in tiles:
                processed_tile = self._process_single_tile(tile, source_embedding)
                processed_tiles.append(processed_tile)
                
        # Sort tiles by position to ensure correct order
        processed_tiles.sort(key=lambda x: (x['position'][0], x['position'][1]))
        
        # Combine tiles
        result_frame = self.explode_pixel_boost(
            processed_tiles, 
            (target_resolution, target_resolution)
        )
        
        total_time = time.time() - start_time
        logger.debug(f"Face swap completed in {total_time:.3f}s")
        
        return result_frame


class ExpressionTransfer:
    """
    Advanced expression transfer for preserving and enhancing facial expressions.
    Supports both NeuroMotion and classical methods.
    """
    
    def __init__(
        self,
        method: str = 'neuromotion',
        model_path: Optional[str] = None,
        execution_provider: str = 'cuda',
        strength: float = 1.0,
        preserve_identity: float = 0.8,
        enhance_details: bool = True,
    ):
        """
        Initialize expression transfer module.
        
        Args:
            method: Expression transfer method ('neuromotion', 'attentionmesh', 'classical')
            model_path: Optional path to custom model
            execution_provider: Execution provider ('cuda', 'cpu')
            strength: Expression transfer strength (0.0-1.0)
            preserve_identity: Identity preservation strength (0.0-1.0)
            enhance_details: Whether to enhance expression details
        """
        self.method = method
        self.strength = strength
        self.preserve_identity = preserve_identity
        self.enhance_details = enhance_details
        self.execution_provider = execution_provider
        
        # Check available methods
        if method == 'neuromotion' and not NEUROMOTION_AVAILABLE:
            logger.warning("NeuroMotion not available, falling back to classical method")
            self.method = 'classical'
            
        # Initialize the selected method
        self._initialize_method(model_path)
        
        # For tracking previous expressions in video
        self.prev_expressions = None
        self.temporal_smoothing = 0.3
        
    def _initialize_method(self, model_path: Optional[str]):
        """Initialize the selected expression transfer method"""
        if self.method == 'neuromotion':
            try:
                device = 'cuda' if self.execution_provider == 'cuda' and CUDA_AVAILABLE else 'cpu'
                self.expression_model = neuromotion.ExpressionTransfer(
                    device=device,
                    model_path=model_path or 'default',
                    preserve_identity=self.preserve_identity
                )
                logger.info("Initialized NeuroMotion 2.3 for expression transfer")
            except Exception as e:
                logger.error(f"Failed to initialize NeuroMotion: {str(e)}")
                self.method = 'classical'
                
        elif self.method == 'attentionmesh':
            try:
                # AttentionMesh is an advanced 3D mesh-based expression transfer algorithm (2025)
                import attentionmesh
                self.expression_model = attentionmesh.FacialExpressionModule(
                    preserve_id=self.preserve_identity,
                    detail_level='high' if self.enhance_details else 'medium'
                )
                logger.info("Initialized AttentionMesh for expression transfer")
            except ImportError:
                logger.warning("AttentionMesh not available, falling back to classical")
                self.method = 'classical'
        
        elif self.method == 'classical':
            # Classical method uses facial landmarks and blending
            self.expression_model = None
            logger.info("Using classical expression transfer method")
            
    def transfer_expression(
        self,
        source_frame: np.ndarray,
        source_landmarks: List[List[int]],
        target_frame: np.ndarray,
        target_landmarks: List[List[int]]
    ) -> np.ndarray:
        """
        Transfer expression from source to target.
        
        Args:
            source_frame: Source face frame
            source_landmarks: Source face landmarks
            target_frame: Target face frame
            target_landmarks: Target face landmarks
            
        Returns:
            Target frame with transferred expression
        """
        # Apply the selected method
        if self.method == 'neuromotion':
            # Use NeuroMotion neural network approach
            try:
                # Extract expression parameters from source
                source_expression = self.expression_model.extract_expression(source_frame, source_landmarks)
                
                # Apply temporal smoothing for video
                if self.prev_expressions is not None:
                    source_expression = self.temporal_smoothing * source_expression + \
                                       (1 - self.temporal_smoothing) * self.prev_expressions
                self.prev_expressions = source_expression.copy()
                
                # Apply expression to target face
                result_frame = self.expression_model.apply_expression(
                    target_frame,
                    target_landmarks,
                    source_expression,
                    strength=self.strength
                )
                return result_frame
            except Exception as e:
                logger.error(f"Error in NeuroMotion expression transfer: {str(e)}")
                # Fall back to classical method
                return self._transfer_expression_classical(source_frame, source_landmarks, target_frame, target_landmarks)
                
        elif self.method == 'attentionmesh':
            # Use AttentionMesh for 3D mesh-based expression transfer
            try:
                result_frame = self.expression_model.transfer(
                    source_img=source_frame,
                    target_img=target_frame,
                    source_landmarks=source_landmarks,
                    target_landmarks=target_landmarks,
                    strength=self.strength
                )
                return result_frame
            except Exception as e:
                logger.error(f"Error in AttentionMesh expression transfer: {str(e)}")
                return self._transfer_expression_classical(source_frame, source_landmarks, target_frame, target_landmarks)
        else:
            # Use classical method
            return self._transfer_expression_classical(source_frame, source_landmarks, target_frame, target_landmarks)
            
    def _transfer_expression_classical(
        self,
        source_frame: np.ndarray,
        source_landmarks: List[List[int]],
        target_frame: np.ndarray,
        target_landmarks: List[List[int]]
    ) -> np.ndarray:
        """
        Classical expression transfer using landmarks and warping.
        
        Args:
            source_frame: Source face frame
            source_landmarks: Source face landmarks
            target_frame: Target face frame
            target_landmarks: Target face landmarks
            
        Returns:
            Target frame with transferred expression
        """
        # Convert landmarks to numpy arrays
        source_points = np.array(source_landmarks, dtype=np.float32)
        target_points = np.array(target_landmarks, dtype=np.float32)
        
        # Focus on landmarks for expressive features (eyes, eyebrows, mouth)
        expression_indices = self._get_expression_indices()
        if len(expression_indices) > 0:
            source_exp_points = source_points[expression_indices]
            target_exp_points = target_points[expression_indices]
            
            # Calculate expression displacement vectors
            if source_exp_points.shape == target_exp_points.shape and source_exp_points.size > 0:
                # Create result frame
                result_frame = target_frame.copy()
                
                # Calculate displacement field
                displacements = source_exp_points - target_exp_points
                displacements *= self.strength
                
                # Apply displacements
                new_target_points = target_points.copy()
                new_target_points[expression_indices] = target_exp_points + displacements
                
                # Perform face warping based on landmarks
                # First, triangulate the face landmarks
                rect = (0, 0, target_frame.shape[1], target_frame.shape[0])
                subdiv = cv2.Subdiv2D(rect)
                
                for point in new_target_points:
                    subdiv.insert((int(point[0]), int(point[1])))
                
                triangles = self._get_triangulation(subdiv, new_target_points)
                
                # Perform piece-wise affine warping
                for triangle in triangles:
                    self._warp_triangle(target_frame, result_frame, target_points, new_target_points, triangle)
                    
                # Enhance expression details if enabled
                if self.enhance_details:
                    result_frame = self._enhance_expression_details(result_frame)
                    
                return result_frame
                
        # Fallback - return target frame unmodified
        return target_frame
        
    def _get_expression_indices(self) -> List[int]:
        """Get indices of landmarks related to facial expression"""
        # This is a simplified version. The actual implementation would depend
        # on the facial landmark detector being used.
        # These indices are example values for a 68-point landmark system
        
        # Eyes, eyebrows, and mouth typically contain expression information
        eyebrow_indices = list(range(17, 27))  # Eyebrows
        eye_indices = list(range(36, 48))      # Eyes
        mouth_indices = list(range(48, 68))    # Mouth
        
        return eyebrow_indices + eye_indices + mouth_indices
        
    def _get_triangulation(self, subdiv: cv2.Subdiv2D, points: np.ndarray) -> List[List[int]]:
        """Get triangulation of facial landmarks"""
        triangles = []
        triangle_list = subdiv.getTriangleList()
        
        for t in triangle_list:
            pt1 = (int(t[0]), int(t[1]))
            pt2 = (int(t[2]), int(t[3]))
            pt3 = (int(t[4]), int(t[5]))
            
            # Find indices in our landmark points list
            idx1 = self._find_landmark_index(points, pt1)
            idx2 = self._find_landmark_index(points, pt2)
            idx3 = self._find_landmark_index(points, pt3)
            
            if idx1 is not None and idx2 is not None and idx3 is not None:
                triangles.append([idx1, idx2, idx3])
                
        return triangles
        
    def _find_landmark_index(self, points: np.ndarray, point: Tuple[int, int]) -> Optional[int]:
        """Find the index of a landmark point"""
        for i, p in enumerate(points):
            if abs(p[0] - point[0]) < 2 and abs(p[1] - point[1]) < 2:
                return i
        return None
        
    def _warp_triangle(
        self,
        img1: np.ndarray,
        img2: np.ndarray,
        points1: np.ndarray,
        points2: np.ndarray,
        triangle: List[int]
    ) -> None:
        """Warp triangular regions from img1 to img2"""
        # Get triangular regions
        x1, y1, w1, h1 = cv2.boundingRect(np.float32([points1[i] for i in triangle]))
        x2, y2, w2, h2 = cv2.boundingRect(np.float32([points2[i] for i in triangle]))
        
        # Create masks for triangular regions
        mask = np.zeros((h2, w2, 3), dtype=np.float32)
        
        # Offset points to make them relative to bounding rect
        points1_rect = []
        points2_rect = []
        
        for i in triangle:
            points1_rect.append(((points1[i][0] - x1), (points1[i][1] - y1)))
            points2_rect.append(((points2[i][0] - x2), (points2[i][1] - y2)))
            
        # Calculate affine transformation
        warp_mat = cv2.getAffineTransform(np.float32(points1_rect), np.float32(points2_rect))
        
        # Warp triangular region
        img1_rect = img1[y1:y1+h1, x1:x1+w1]
        img2_rect = np.zeros((h2, w2), dtype=img1.dtype)
        
        cv2.warpAffine(
            img1_rect,
            warp_mat,
            (w2, h2),
            img2_rect,
            borderMode=cv2.BORDER_REFLECT_101
        )
        
        # Create mask
        mask_rect = np.zeros((h2, w2), dtype=np.uint8)
        cv2.fillConvexPoly(mask_rect, np.int32(points2_rect), 1)
        
        # Apply mask to warped image
        warped_triangle = img2_rect * mask_rect[:, :, None]
        
        # Copy triangular region of the warped image to the output image
        img2[y2:y2+h2, x2:x2+w2] = img2[y2:y2+h2, x2:x2+w2] * (1 - mask_rect[:, :, None]) + warped_triangle
            
    def _enhance_expression_details(self, frame: np.ndarray) -> np.ndarray:
        """Enhance expression details in the face"""
        # Convert to LAB for better perceptual enhancement
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to enhance local contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l)
        
        # Merge channels
        enhanced_lab = cv2.merge([l_enhanced, a, b])
        enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        return enhanced_bgr
        
    def reset_state(self):
        """Reset temporal state for new video sequence"""
        self.prev_expressions = None


class OcclusionHandler:
    """
    Advanced occlusion detection and handling for face swapping.
    """
    
    def __init__(
        self,
        method: str = 'matchformer', 
        threshold: float = 0.5,
        refinement: bool = True,
        temporal_smoothing: bool = True
    ):
        """
        Initialize occlusion handler.
        
        Args:
            method: Occlusion detection method ('matchformer', 'segment', 'classical')
            threshold: Occlusion detection threshold
            refinement: Whether to apply refinement to occlusion masks
            temporal_smoothing: Whether to apply temporal smoothing to occlusion masks
        """
        self.method = method
        self.threshold = threshold
        self.refinement = refinement
        self.temporal_smoothing = temporal_smoothing
        
        # Check available methods
        if method == 'matchformer' and not MATCHFORMER_AVAILABLE:
            logger.warning("MATCHformer not available, falling back to classical method")
            self.method = 'classical'
            
        self._initialize_method()
        
        # For temporal smoothing
        self.prev_occlusion_mask = None
        self.smoothing_factor = 0.7
        
    def _initialize_method(self):
        """Initialize the selected occlusion detection method"""
        if self.method == 'matchformer':
            try:
                # MATCHformer is a state-of-the-art transformer-based approach for occlusion handling (2025)
                self.occlusion_model = matchformer.OcclusionDetector(
                    model_version='v2.3',
                    threshold=self.threshold,
                    device='cuda' if CUDA_AVAILABLE else 'cpu'
                )
                logger.info("Initialized MATCHformer for occlusion detection")
            except Exception as e:
                logger.error(f"Failed to initialize MATCHformer: {str(e)}")
                self.method = 'classical'
                
        elif self.method == 'segment':
            try:
                # Try to use segmentation-based approach
                import segment_anything
                self.occlusion_model = segment_anything.SamPredictor(
                    "segment_anything/sam_vit_h_4b8939.pth"
                )
                logger.info("Initialized SAM for occlusion detection")
            except ImportError:
                logger.warning("SAM not available, falling back to classical")
                self.method = 'classical'
                
        elif self.method == 'classical':
            # Classical method doesn't need specific initialization
            self.occlusion_model = None
            logger.info("Using classical occlusion detection")
            
    def detect_occlusions(
        self, 
        frame: np.ndarray,
        face_bbox: List[int],
        landmarks: Optional[List[List[int]]] = None
    ) -> np.ndarray:
        """
        Detect occlusions in a face region.
        
        Args:
            frame: Input frame
            face_bbox: Face bounding box [x1, y1, x2, y2]
            landmarks: Optional face landmarks
            
        Returns:
            Occlusion mask (1 for occluded regions, 0 for non-occluded)
        """
        x1, y1, x2, y2 = face_bbox
        face_region = frame[y1:y2, x1:x2]
        
        # Apply the selected method
        if self.method == 'matchformer':
            try:
                # MATCHformer produces detailed, per-pixel occlusion masks
                occlusion_mask = self.occlusion_model.detect(face_region)
                
                # Apply refinement if needed
                if self.refinement:
                    occlusion_mask = self._refine_mask(occlusion_mask, face_region)
                    
                # Apply temporal smoothing if enabled and previous mask exists
                if self.temporal_smoothing and self.prev_occlusion_mask is not None:
                    if self.prev_occlusion_mask.shape == occlusion_mask.shape:
                        occlusion_mask = self.smoothing_factor * self.prev_occlusion_mask + \
                                        (1 - self.smoothing_factor) * occlusion_mask
                
                # Update previous mask
                self.prev_occlusion_mask = occlusion_mask.copy()
                
                return occlusion_mask
                
            except Exception as e:
                logger.error(f"Error in MATCHformer occlusion detection: {str(e)}")
                # Fall back to classical method
                return self._detect_occlusions_classical(face_region, landmarks)
                
        elif self.method == 'segment':
            try:
                # Use SAM to predict segmentation mask
                self.occlusion_model.set_image(face_region)
                
                # Generate prompts based on landmarks if available
                if landmarks:
                    points = []
                    for lm in landmarks:
                        # Adjust landmark coordinates to be relative to face_region
                        x, y = lm[0] - x1, lm[1] - y1
                        if 0 <= x < face_region.shape[1] and 0 <= y < face_region.shape[0]:
                            points.append([x, y])
                    
                    if points:
                        masks, scores, _ = self.occlusion_model.predict(
                            point_coords=np.array(points),
                            point_labels=np.ones(len(points)),
                            multimask_output=True
                        )
                        
                        # Use the mask with highest score
                        best_mask_idx = np.argmax(scores)
                        mask = masks[best_mask_idx]
                        
                        # Invert mask (1 for occlusions, 0 for face)
                        occlusion_mask = 1 - mask.astype(np.float32)
                        
                        if self.refinement:
                            occlusion_mask = self._refine_mask(occlusion_mask, face_region)
                            
                        # Apply temporal smoothing
                        if self.temporal_smoothing and self.prev_occlusion_mask is not None:
                            if self.prev_occlusion_mask.shape == occlusion_mask.shape:
                                occlusion_mask = self.smoothing_factor * self.prev_occlusion_mask + \
                                               (1 - self.smoothing_factor) * occlusion_mask
                        
                        # Update previous mask
                        self.prev_occlusion_mask = occlusion_mask.copy()
                        
                        return occlusion_mask
            except Exception as e:
                logger.error(f"Error in SAM occlusion detection: {str(e)}")
                
        # Fall back to classical method
        return self._detect_occlusions_classical(face_region, landmarks)
        
    def _detect_occlusions_classical(
        self, 
        face_region: np.ndarray,
        landmarks: Optional[List[List[int]]] = None
    ) -> np.ndarray:
        """
        Classical occlusion detection using color and gradient analysis.
        
        Args:
            face_region: Face image region
            landmarks: Face landmarks (relative to face_region)
            
        Returns:
            Occlusion mask
        """
        # Create an initial mask
        height, width = face_region.shape[:2]
        mask = np.zeros((height, width), dtype=np.float32)
        
        # Convert to grayscale
        gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
        
        # 1. Edge detection
        edges = cv2.Canny(gray, 100, 200)
        
        # 2. Gradient magnitude
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = cv2.magnitude(sobelx, sobely)
        magnitude = cv2.normalize(magnitude, None, 0, 1, cv2.NORM_MINMAX)
        
        # 3. Color analysis in HSV space
        hsv = cv2.cvtColor(face_region, cv2.COLOR_BGR2HSV)
        
        # Calculate saturation variance in local neighborhoods
        sat_var = cv2.Laplacian(hsv[:,:,1], cv2.CV_64F)
        sat_var = np.abs(sat_var)
        sat_var = cv2.normalize(sat_var, None, 0, 1, cv2.NORM_MINMAX)
        
        # Combine features
        combined = (edges.astype(np.float32) / 255.0) * 0.3 + magnitude * 0.4 + sat_var * 0.3
        
        # Apply threshold
        occlusion_mask = (combined > self.threshold).astype(np.float32)
        
        # Apply morphological operations to clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        occlusion_mask = cv2.morphologyEx(occlusion_mask, cv2.MORPH_OPEN, kernel)
        occlusion_mask = cv2.morphologyEx(occlusion_mask, cv2.MORPH_CLOSE, kernel)
        
        # Use landmark information if available
        if landmarks:
            # Create a face region mask based on landmarks
            face_mask = np.zeros((height, width), dtype=np.uint8)
            hull = cv2.convexHull(np.array(landmarks))
            cv2.fillConvexPoly(face_mask, hull, 1)
            
            # Dilate the face mask slightly
            face_mask = cv2.dilate(face_mask, kernel, iterations=2)
            
            # Combine: only detect occlusions within the face region
            occlusion_mask = occlusion_mask * face_mask
            
        # Apply temporal smoothing if enabled
        if self.temporal_smoothing and self.prev_occlusion_mask is not None:
            if self.prev_occlusion_mask.shape == occlusion_mask.shape:
                occlusion_mask = self.smoothing_factor * self.prev_occlusion_mask + \
                                (1 - self.smoothing_factor) * occlusion_mask
        
        # Update previous mask
        self.prev_occlusion_mask = occlusion_mask.copy()
        
        return occlusion_mask
        
    def _refine_mask(self, mask: np.ndarray, face_region: np.ndarray) -> np.ndarray:
        """Refine occlusion mask using GrabCut or edge-aware filtering"""
        # Apply guided filter for edge-aware refinement
        try:
            gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            mask_refined = cv2.ximgproc.guidedFilter(gray, mask, 5, 0.1)
            return mask_refined
        except Exception:
            # Fall back to simple Gaussian blur
            return cv2.GaussianBlur(mask, (5, 5), 1.0)
            
    def reset_state(self):
        """Reset temporal state for new video sequence"""
        self.prev_occlusion_mask = None


class FaceSwapProcessor:
    """
    Complete face swapping