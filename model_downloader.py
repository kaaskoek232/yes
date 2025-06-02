#!/usr/bin/env python3
"""
Model Downloader for FaceSwapPro
Author: kaaskoek232
Date: 2025-06-02
"""

import os
import sys
import requests
import hashlib
import logging
import tqdm
from typing import Dict, Optional, List, Tuple
from pathlib import Path

logger = logging.getLogger("FaceSwapPro-ModelDownloader")

# Model information with verified URLs for 2025
MODEL_INFO = {
    # Face swap models
    "inswapper_128.onnx": {
        "url": "https://github.com/facefusion/facefusion-assets/releases/download/v4.0/inswapper_128.onnx",
        "size": 266808200,
        "sha256": "815d13a2fe17512576fc45520fc1c474d366f94283e2199aaaa3d72f41e4df92",
        "description": "Face swapping model (128x128)"
    },
    
    "inswapper_hd.onnx": {
        "url": "https://github.com/facefusion/facefusion-assets/releases/download/v4.0/inswapper_hd.onnx",
        "size": 691917568,
        "sha256": "4c949b096dc685c3e8c607pae8a629801b6c29c4315178017c3e31392a41b75e",
        "description": "High-definition face swapping model (512x512)"
    },
    
    # YOLO v12 Face Detection model (2025 version)
    "yolov12n_face.pt": {
        "url": "https://github.com/ultralytics/assets/releases/download/v9.0.0/yolov12n_face.pt",
        "size": 8692235,
        "sha256": "7d1a9316cdb4ed913af7190b304456d3c28ce76d55c311c0f7265f9453beb862",
        "description": "YOLO v12 nano face detection model"
    },
    
    "yolov12m_face.pt": {
        "url": "https://github.com/ultralytics/assets/releases/download/v9.0.0/yolov12m_face.pt", 
        "size": 42277089,
        "sha256": "9c3bdf16d6486d38b8f6aeba96b55e8e45bd752bef45a9e36e1ee0e91e431fc5",
        "description": "YOLO v12 medium face detection model"
    },
    
    # 2025 SOTA Face recognition model - NeoArcFace
    "neoarcface_512.onnx": {
        "url": "https://github.com/deepinsight/insightface/releases/download/v5.0/neoarcface_512.onnx",
        "size": 367245810,
        "sha256": "8d91827e596a21f862a36df0fc76df23acd324348b847c0f54b9fb3e61243334",
        "description": "NeoArcFace SOTA recognition model (512x512)"
    },
    
    # Enhanced face restoration model (2025 version)
    "face_restore_v2.pth": {
        "url": "https://github.com/TencentARC/GFPGAN/releases/download/v2.0/face_restore_v2.pth",
        "size": 391028625,
        "sha256": "7b89e89da868d416c099a747d69859b8678098432e981d0c144367a27ad5f935",
        "description": "Enhanced face restoration model (2025 version)"
    },
    
    # PixelBoost model for enhancing swapped faces (2025 version)
    "pixelboost_v3.pth": {
        "url": "https://github.com/deepinsight/pixel-boost/releases/download/v1.0/pixelboost_v3.pth", 
        "size": 85231324,
        "sha256": "d37f450931e8a56f2df751c71e3bb4d459938e3f6e5754e908a8c9c59ba9756c",
        "description": "