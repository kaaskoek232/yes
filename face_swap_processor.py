#!/usr/bin/env python3
"""
Face Swap Processor for FaceSwapPro
Author: kaaskoek232  
Date: 2025-06-02
"""

import os
import sys
import cv2
import numpy as np
import torch
import onnxruntime
from typing import Dict, List, Optional, Union, Tuple, Callable
import logging
from pathlib import Path
import time

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from src.utils.model_downloader import ensure_model_available, get_model_path
from src.core.face_processor import FaceProcessor
from src.core.pixel_boost import PixelBoost

logger = logging.getLogger("FaceSwapPro-Processor")

class FaceSwapProcessor:
    """
    Main face swapping processor integrating detection, swapping, and enhancement.
    """
    
    def __init__(
        self,
        face_swap_model: str = "inswapper_128.onnx",
        detection_model: str = "yolov12m_face.pt",
        recognition_model: str = "neoarcface_512.onnx",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        enhance_output: bool = True,
        detection_threshold: float = 0.5,
        recognition_threshold: float = 0.6
    ):
        """
        Initialize face swap processor.
        
        Args:
            face_swap_model: Face swap model name
            detection_model: Face detection model name
            recognition_model: Face recognition model name
            device: Computation device
            enhance_output: Apply PixelBoost enhancement
            detection_threshold: Face detection confidence threshold
            recognition_threshold: Face recognition similarity threshold
        """
        self.device = device
        self.enhance_output = enhance_output
        self.detection_threshold = detection_threshold
        self.recognition_threshold = recognition_threshold
        self.face_swap_model = None
        self.face_database = {}  # id -> embedding
        
        # Download models if needed
        models_to_download = [face_swap_model, detection_model, recognition_model]
        for model_name in models_to_download:
            if not ensure_model_available(model_name):
                logger.error(f"Failed to download model: {model_name}")
                
        # Initialize face processor
        self.face_processor = FaceProcessor(
            detection_model=detection_model,
            recognition_model=recognition_model,
            detection_threshold=detection_threshold,
            device=device
        )
        
        # Initialize PixelBoost if enhancement requested
        if enhance_output:
            self.pixel_boost = PixelBoost(device=device)
        else:
            self.pixel_boost = None
        
        # Initialize face swap model
        self._init_face_swap(face_swap_model)
        
    def _init_face_swap(self, model_name: str):
        """Initialize face swap model"""
        try:
            model_path = get_model_path(model_name)
            
            # Create ONNX session
            providers = ['CUDAExecutionProvider' if self.device == 'cuda' else 'CPUExecutionProvider']
            self.face_swap_model = onnxruntime.InferenceSession(model_path, providers=providers)
            
            logger.info(f"Face swap model initialized with {model_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize face swap model: {e}")
            
    def is_available(self) -> bool:
        """Check if face swapping is available"""
        return self.face_swap_model is not None and self.face_processor.is_available()
        
    def detect_faces(self, image: np.ndarray) -> List[Dict]:
        """
        Detect faces in an image.
        
        Args:
            image: Input BGR image
            
        Returns:
            List of face dictionaries
        """
        return self.face_processor.detect_faces(image)
        
    def add_face_to_database(self, face_id: str, face_image: np.ndarray) -> bool:
        """
        Add a face to the database.
        
        Args:
            face_id: Unique ID for the face
            face_image: Face image
            
        Returns:
            True if successful, False otherwise
        """
        return self.face_processor.add_face_to_database(face_id, face_image)
        
    def swap_face(
        self,
        source_embedding: np.ndarray,
        target_face: np.ndarray,
        target_landmarks: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Swap a single face.
        
        Args:
            source_embedding: Source face embedding
            target_face: Target face image
            target_landmarks: Target face landmarks
            
        Returns:
            Swapped face image
        """
        if not self.is_available():
            return target_face
            
        try:
            # Resize target face to expected input size
            target_face_resized = cv2.resize(target_face, (192, 192))
            
            # Preprocess image
            target_face_rgb = cv2.cvtColor(target_face_resized, cv2.COLOR_BGR2RGB)
            target_face_tensor = target_face_rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
            target_face_tensor = np.expand_dims(target_face_tensor, axis=0)
            
            # Prepare inputs
            inputs = {
                self.face_swap_model.get_inputs()[0].name: source_embedding.reshape(1, -1),
                self.face_swap_model.get_inputs()[1].name: target_face_tensor
            }
            
            # Run inference
            outputs = self.face_swap_model.run(None, inputs)
            
            # Process output
            swapped_face = outputs[0][0]
            swapped_face = swapped_face.transpose(1, 2, 0)
            swapped_face = (swapped_face * 255).astype(np.uint8)
            swapped_face = cv2.cvtColor(swapped_face, cv2.COLOR_RGB2BGR)
            
            # Resize back to original size
            h, w = target_face.shape[:2]
            swapped_face = cv2.resize(swapped_face, (w, h))
            
            # Enhance with PixelBoost if enabled
            if self.pixel_boost is not None:
                swapped_face = self.pixel_boost.process_face(swapped_face)
                
            return swapped_face
            
        except Exception as e:
            logger.error(f"Error swapping face: {e}")
            return target_face
            
    def process_frame(
        self,
        frame: np.ndarray,
        source_embedding: np.ndarray,
        target_faces: Optional[List[Dict]] = None,
        swap_all: bool = True,
        target_ids: Optional[List[int]] = None,
        resolution: int = 512
    ) -> np.ndarray:
        """
        Process a single frame.
        
        Args:
            frame: Input BGR frame
            source_embedding: Source face embedding
            target_faces: Optional pre-detected target faces
            swap_all: Swap all detected faces
            target_ids: IDs of specific faces to swap
            resolution: Processing resolution
            
        Returns:
            Processed frame with swapped faces
        """
        if not self.is_available():
            return frame
            
        try:
            # Detect faces if not provided
            if target_faces is None:
                target_faces = self.detect_faces(frame)
                
            if not target_faces:
                return frame
                
            # Create result image
            result = frame.copy()
            
            # Process each face
            for face in target_faces:
                # Skip if not in target IDs
                if not swap_all and (target_ids is not None) and (face['id'] not in target_ids):
                    continue
                    
                # Extract face
                bbox = face['bbox']
                face_img = self.face_processor.extract_face(frame, bbox, scale=1.2)
                
                # Get landmarks
                landmarks = face.get('landmarks') or None
                
                # Swap face
                swapped_face = self.swap_face(source_embedding, face_img, landmarks)
                
                # Blend back into result image
                x1, y1, x2, y2 = [int(coord) for coord in bbox]
                
                # Create mask for seamless blending
                mask = np.zeros_like(frame)
                cv2.ellipse(
                    mask,
                    center=(int((x1+x2)/2), int((y1+y2)/2)),
                    axes=(int((x2-x1)/2), int((y2