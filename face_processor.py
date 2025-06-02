#!/usr/bin/env python3
"""
Face Processor for FaceSwapPro
Author: kaaskoek232
Date: 2025-06-02
"""

import os
import sys
import cv2
import numpy as np
import torch
import onnxruntime
from typing import Dict, List, Optional, Union, Tuple
import logging
from pathlib import Path

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from src.utils.model_downloader import ensure_model_available, get_model_path

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("FaceSwapPro-FaceProcessor")

class FaceProcessor:
    """Face detection, recognition, and alignment using SOTA 2025 models"""
    
    def __init__(
        self,
        detection_model: str = "yolov12m_face.pt",
        recognition_model: str = "neoarcface_512.onnx",
        landmark_model: str = "landmark_detector_v2.onnx",
        detection_threshold: float = 0.5,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize face processor with SOTA models.
        
        Args:
            detection_model: Face detection model name (YOLO v12)
            recognition_model: Face recognition model name (NeoArcFace)
            landmark_model: Face landmark detection model name
            detection_threshold: Face detection confidence threshold
            device: Computation device
        """
        self.detection_threshold = detection_threshold
        self.device = device
        self.face_detector = None
        self.face_recognizer = None
        self.landmark_detector = None
        self.face_database = {}  # id -> embedding
        
        # Download models if needed
        models_to_download = [detection_model, recognition_model, landmark_model]
        for model_name in models_to_download:
            if not ensure_model_available(model_name):
                logger.error(f"Failed to download model: {model_name}")
                
        # Initialize face detector
        self._init_face_detector(detection_model)
        
        # Initialize face recognizer
        self._init_face_recognizer(recognition_model)
        
        # Initialize landmark detector
        self._init_landmark_detector(landmark_model)
        
    def _init_face_detector(self, model_name: str):
        """Initialize YOLO v12 face detector"""
        try:
            # Import YOLO
            from ultralytics import YOLO
            
            # Load YOLO model
            model_path = get_model_path(model_name)
            self.face_detector = YOLO(model_path)
            
            logger.info(f"Face detector initialized successfully with {model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize face detector: {e}")
            
    def _init_face_recognizer(self, model_name: str):
        """Initialize NeoArcFace face recognizer"""
        try:
            model_path = get_model_path(model_name)
            
            # Create ONNX session
            providers = ['CUDAExecutionProvider' if self.device == 'cuda' else 'CPUExecutionProvider']
            self.face_recognizer = onnxruntime.InferenceSession(model_path, providers=providers)
            
            logger.info(f"Face recognizer initialized successfully with {model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize face recognizer: {e}")
            
    def _init_landmark_detector(self, model_name: str):
        """Initialize face landmark detector"""
        try:
            model_path = get_model_path(model_name)
            
            # Create ONNX session
            providers = ['CUDAExecutionProvider' if self.device == 'cuda' else 'CPUExecutionProvider']
            self.landmark_detector = onnxruntime.InferenceSession(model_path, providers=providers)
            
            logger.info(f"Landmark detector initialized successfully with {model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize landmark detector: {e}")
            
    def is_available(self) -> bool:
        """Check if face processor is available"""
        return self.face_detector is not None
        
    def detect_faces(self, image: np.ndarray) -> List[Dict]:
        """
        Detect faces in an image using YOLO v12.
        
        Args:
            image: Input BGR image
            
        Returns:
            List of face dictionaries
        """
        if not self.is_available():
            logger.error("Face detector not available")
            return []
            
        try:
            # Process with YOLO
            results = self.face_detector(image, verbose=False)
            
            # Convert to our format
            faces = []
            for i, det in enumerate(results[0].boxes.data.cpu().numpy()):
                x1, y1, x2, y2, conf, _ = det
                
                if conf < self.detection_threshold:
                    continue
                    
                # Extract face
                face_img = self.extract_face(image, [x1, y1, x2, y2], scale=1.2)
                
                # Get landmarks
                landmarks = self._get_landmarks(face_img)
                
                # Get embedding
                embedding = self._get_embedding(face_img)
                
                faces.append({
                    'id': i,
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'score': float(conf),
                    'landmarks': landmarks.tolist() if landmarks is not None else None,
                    'embedding': embedding.tolist() if embedding is not None else None,
                    'face_image': face_img
                })
                
            return faces
            
        except Exception as e:
            logger.error(f"Error detecting faces: {e}")
            return []
            
    def extract_face(self, image: np.ndarray, bbox: List[float], scale: float = 1.0) -> np.ndarray:
        """
        Extract face region from image.
        
        Args:
            image: Input BGR image
            bbox: Face bounding box [x1, y1, x2, y2]
            scale: Scale factor for padding
            
        Returns:
            Extracted face image
        """
        h, w = image.shape[:2]
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        
        # Apply scale factor
        face_width = x2 - x1
        face_height = y2 - y1
        
        # Calculate padding
        padding_x = int((scale - 1.0) * face_width / 2)
        padding_y = int((scale - 1.0) * face_height / 2)
        
        # Apply padding with bounds checking
        x1 = max(0, x1 - padding_x)
        y1 = max(0, y1 - padding_y)
        x2 = min(w, x2 + padding_x)
        y2 = min(h, y2 + padding_y)
        
        # Extract face region
        face = image[y1:y2, x1:x2]
        return face
        
    def _get_landmarks(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """Get facial landmarks for a face image"""
        if self.landmark_detector is None:
            return None
            
        try:
            # Preprocess image
            face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            face_resized = cv2.resize(face_rgb, (192, 192))
            face_tensor = face_resized.transpose(2, 0, 1).astype(np.float32) / 255.0
            face_tensor = np.expand_dims(face_tensor, axis=0)
            
            # Run inference
            inputs = {self.landmark_detector.get_inputs()[0].name: face_tensor}
            outputs = self.landmark_detector.run(None, inputs)
            
            # Process landmarks
            landmarks = outputs[0][0].reshape(-1, 2)
            
            # Scale landmarks to original image
            h, w = face_image.shape[:2]
            landmarks[:, 0] *= w / 192
            landmarks[:, 1] *= h / 192
            
            return landmarks
            
        except Exception as e:
            logger.error(f"Error getting landmarks: {e}")
            return None
            
    def _get_embedding(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """Get face embedding for a face image"""
        if self.face_recognizer is None:
            return None
            
        try:
            # Preprocess image
            face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            face_resized = cv2.resize(face_rgb, (512, 512))
            face_tensor = face_resized.transpose(2, 0, 1).astype(np.float32) / 255.0
            face_tensor = np.expand_dims(face_tensor, axis=0)
            
            # Mean and std normalization
            mean = np.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1))
            std = np.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1))
            face_tensor = (face_tensor - mean) / std
            
            # Run inference
            inputs = {self.face_recognizer.get_inputs()[0].name: face_tensor}
            outputs = self.face_recognizer.run(None, inputs)
            
            # Normalize embedding
            embedding = outputs[0][0]
            embedding = embedding / np.linalg.norm(embedding)
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            return None
            
    def compare_faces(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compare two face embeddings and return similarity score (higher is more similar)"""
        similarity = np.dot(embedding1, embedding2)
        return float(similarity)
        
    def add_face_to_database(self, face_id: str, face_image: np.ndarray) -> bool:
        """
        Add a face to the database.
        
        Args:
            face_id: Unique ID for the face
            face_image: Face image
            
        Returns:
            True if successful, False otherwise
        """
        if not self.is_available():
            return False
            
        try:
            # Detect faces
            faces = self.detect_faces(face_image)
            
            if not faces:
                logger.warning(f"No face detected for ID {face_id}")
                return False
                
            # Use the first detected face
            embedding = np.array(faces[0]['embedding'])
            
            # Add to database
            self.face_database[face_id] = embedding
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding face to database: {e}")
            return False
            
    def find_face_in_database(self, face_image: np.ndarray, threshold: float = 0.6) -> Optional[str]:
        """
        Find a face in the database.
        
        Args:
            face_image: Face image
            threshold: Similarity threshold
            
        Returns:
            Face ID if found, None otherwise
        """
        if not self.face_database:
            return None
            
        try:
            # Get embedding
            embedding = self._get_embedding(face_image)
            
            if embedding is None:
                return None
                
            # Find best match
            best_match = None
            best_similarity = -1.0
            
            for face_id, db_embedding in self.face_database.items():
                similarity = self.compare_faces(embedding, db_embedding)
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = face_id
                    
            # Check threshold
            if best_similarity < threshold:
                return None
                
            return best_match
            
        except Exception as e:
            logger.error(f"Error finding face in database: {e}")
            return None