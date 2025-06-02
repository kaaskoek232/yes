#!/usr/bin/env python3
"""
AI Scene Composition for FaceSwapPro
Author: kaaskoek232
Date: 2025-06-02
"""

import os
import numpy as np
import torch
from typing import Dict, Optional, Tuple, List, Union
import logging
import cv2

# Try to import Gen-2 successor (ControlNet++/2025 SOTA)
try:
    import controlnetplus as cnp
    from diffusers import StableDiffusionControlNetPipeline
    CONTROLNET_AVAILABLE = True
except ImportError:
    CONTROLNET_AVAILABLE = False

try:
    import segment_anything_v2 as sam_v2
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False

logger = logging.getLogger("FaceSwapPro-SceneComposer")

class SceneComposer:
    """
    Advanced scene composition with AI-generated backgrounds
    and intelligent blending using ControlNet++ and SAM v2.
    """
    
    def __init__(
        self,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        controlnet_model: str = "controlnet_sd3",
        sam_model: str = "sam_v2_h",
        quality_level: int = 1,  # 0=fast, 1=standard, 2=high
        seed: Optional[int] = None
    ):
        """
        Initialize scene composer.
        
        Args:
            device: Computation device
            controlnet_model: ControlNet model name
            sam_model: SAM model name
            quality_level: Quality level (0=fast, 1=standard, 2=high)
            seed: Random seed for generation
        """
        self.device = device
        self.quality_level = quality_level
        self.seed = seed
        
        # Initialize segmentation
        self.sam_available = False
        if SAM_AVAILABLE:
            try:
                self.sam = sam_v2.SamAutomaticMaskGenerator.from_pretrained(
                    f"models/{sam_model}",
                    device=device
                )
                self.sam_available = True
                logger.info(f"SAM initialized with {sam_model}")
            except Exception as e:
                logger.error(f"Failed to initialize SAM: {e}")
                
        # Initialize ControlNet
        self.controlnet_available = False
        if CONTROLNET_AVAILABLE:
            try:
                # Advanced settings based on quality level
                inference_steps = [20, 30, 50][quality_level]
                guidance_scale = [5.0, 7.5, 9.0][quality_level]
                
                self.controlnet = StableDiffusionControlNetPipeline.from_pretrained(
                    f"models/{controlnet_model}",
                    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                    use_safetensors=True
                ).to(device)
                
                # Set generation parameters
                self.controlnet.scheduler = cnp.EulerDiscreteScheduler.from_config(
                    self.controlnet.scheduler.config
                )
                
                # Default generation settings
                self.inference_steps = inference_steps
                self.guidance_scale = guidance_scale
                
                self.controlnet_available = True
                logger.info(f"ControlNet initialized with {controlnet_model}")
            except Exception as e:
                logger.error(f"Failed to initialize ControlNet: {e}")
                
    def is_available(self) -> bool:
        """Check if scene composition is available"""
        return self.sam_available and self.controlnet_available
        
    def segment_person(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Segment person from image using SAM v2.
        
        Args:
            image: Input BGR image
            
        Returns:
            Tuple of (person mask, segmented person)
        """
        if not self.sam_available:
            # Simple fallback using grabcut
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            rect = (10, 10, image.shape[1]-20, image.shape[0]-20)
            bgd_model = np.zeros((1, 65), dtype=np.float64)
            fgd_model = np.zeros((1, 65), dtype=np.float64)
            
            cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
            mask = np.where((mask==2) | (mask==0), 0, 1).astype('uint8')
            
            segmented = image * mask[:, :, np.newaxis]
            return mask, segmented
            
        # Use SAM for precise segmentation
        try:
            # Convert to RGB for SAM
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Generate masks
            masks = self.sam.generate(rgb_image)
            
            # Find the largest mask near the center (likely the person)
            h, w = image.shape[:2]
            center_y, center_x = h // 2, w // 2
            
            best_mask = None
            best_score = -float('inf')
            
            for mask_data in masks:
                mask = mask_data["segmentation"].astype(np.uint8)
                area = mask_data["area"]
                
                # Calculate distance from center to mask
                coords = np.column_stack(np.where(mask > 0))
                if len(coords) == 0:
                    continue
                    
                y_mean, x_mean = coords.mean(axis=0)
                center_dist = np.sqrt((y_mean - center_y)**2 + (x_mean - center_x)**2)
                
                # Score combines area and proximity to center
                score = area - center_dist * 10
                
                if score > best_score:
                    best_score = score
                    best_mask = mask
                    
            if best_mask is None:
                # Fallback to grabcut
                return self.segment_person(image)
                
            # Apply mask to image
            segmented = image * best_mask[:, :, np.newaxis]
            
            return best_mask, segmented
            
        except Exception as e:
            logger.error(f"Error in person segmentation: {e}")
            # Fallback
            return self.segment_person(image)
            
    def generate_background(
        self,
        prompt: str,
        control_image: Optional[np.ndarray] = None,
        width: int = 1024,
        height: int = 1024
    ) -> Optional[np.ndarray]:
        """
        Generate background based on prompt and optional control image.
        
        Args:
            prompt: Text prompt describing the desired background
            control_image: Optional control image for guided generation
            width: Output width
            height: Output height
            
        Returns:
            Generated BGR image or None if failed
        """
        if not self.controlnet_available:
            logger.warning("ControlNet not available")
            return None
            
        try:
            # Set random seed if specified
            generator = None
            if self.seed is not None:
                generator = torch.Generator(device=self.device).manual_seed(self.seed)
                
            # Prepare control image if provided
            if control_image is not None:
                # Resize and convert to RGB
                control_image = cv2.resize(control_image, (width, height))
                control_image = cv2.cvtColor(control_image, cv2.COLOR_BGR2RGB)
                # Normalize
                control_image = control_image / 255.0
                control_image = torch.from_numpy(control_image).permute(2, 0, 1).float().to(self.device)
                
            # Add negative prompt for better quality
            negative_prompt = "low quality, blurry, distorted, deformed, disfigured, bad anatomy, watermark"
            
            # Generate image
            image = self.controlnet(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=control_image,
                num_inference_steps=self.inference_steps,
                guidance_scale=self.guidance_scale,
                generator=generator
            ).images[0]
            
            # Convert to BGR for OpenCV
            image_np = np.array(image)
            image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            
            return image_bgr
            
        except Exception as e:
            logger.error(f"Error generating background: {e}")
            return None
            
    def compose_scene(
        self,
        person_image: np.ndarray,
        background_prompt: str,
        maintain_lighting: bool = True,
        maintain_position: bool = True,
        maintain_scale: bool = True,
        blend_edges: bool = True
    ) -> Optional[np.ndarray]:
        """
        Compose a scene by placing a person on an AI-generated background.
        
        Args:
            person_image: Image containing a person
            background_prompt: Prompt describing the desired background
            maintain_lighting: Attempt to maintain lighting consistency
            maintain_position: Keep person in similar position
            maintain_scale: Maintain person's scale
            blend_edges: Apply edge blending for better integration
            
        Returns:
            Composed scene or None if failed
        """
        if not self.is_available():
            logger.warning("Scene composition not available")
            return None
            
        try:
            # 1. Segment the person
            person_mask, segmented_person = self.segment_person(person_image)
            
            # 2. Extract scene structure for consistent background generation
            scene_sketch = self._extract_scene_structure(person_image, person_mask)
            
            # 3. Generate background
            h, w = person_image.shape[:2]
            background = self.generate_background(
                prompt=background_prompt,
                control_image=scene_sketch if maintain_position else None,
                width=w,
                height=h
            )
            
            if background is None:
                return None
                
            # 4. Apply lighting transfer if requested
            if maintain_lighting:
                background = self._transfer_lighting(person_image, background, person_mask)
                
            # 5. Compose person onto background
            result = background.copy()
            
            # If blending edges
            if blend_edges:
                # Create feathered mask
                kernel_size = max(int(min(w, h) * 0.03), 5)  # 3% of image size, at least 5px
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
                feathered_mask = cv2.dilate(person_mask, kernel)
                feathered_mask = cv2.GaussianBlur(feathered_mask, (kernel_size, kernel_size), 0)
                feathered_mask = feathered_mask / 255.0
                
                # Expand mask to 3 channels
                feathered_mask_3ch = np.repeat(feathered_mask[:, :, np.newaxis], 3, axis=2)
                
                # Blend
                result = background * (1 - feathered_mask_3ch) + segmented_person * feathered_mask_3ch
            else:
                # Simple mask-based composition
                mask_3ch = np.repeat(person_mask[:, :, np.newaxis], 3, axis=2)
                result = background * (1 - mask_3ch) + segmented_person * mask_3ch
                
            return result.astype(np.uint8)
            
        except Exception as e:
            logger.error(f"Error in scene composition: {e}")
            return None
            
    def _extract_scene_structure(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Extract scene structure for consistent background generation"""
        # Create a simple sketch with edges and person silhouette
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Create sketch image
        sketch = np.zeros_like(image)
        sketch[:,:,0] = edges  # Blue channel
        sketch[:,:,1] = mask * 255  # Green channel - person silhouette
        
        return sketch
        
    def _transfer_lighting(self, source: np.ndarray, target: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Transfer lighting characteristics from source to target"""
        try:
            # Convert to LAB color space
            source_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB)
            target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB)
            
            # Calculate lighting statistics from source (non-person areas)
            inv_mask = 1 - mask
            if np.sum(inv_mask) > 0:
                source_l = source_lab[:,:,0]
                source_mean_l = np.sum(source_l * inv_mask) / np.sum(inv_mask)
                source_std_l = np.sqrt(np.sum(((source_l - source_mean_l) * inv_mask) ** 2) / np.sum(inv_mask))
            else:
                # Fallback if mask covers entire image
                source_l = source_lab[:,:,0]
                source_mean_l = np.mean(source_l)
                source_std_l = np.std(source_l)
                
            # Calculate target statistics
            target_l = target_lab[:,:,0]
            target_mean_l = np.mean(target_l)
            target_std_l = np.std(target_l)
            
            # Adjust target lighting
            target_l = ((target_l - target_mean_l) / (target_std_l + 1e-6) * source_std_l + source_mean_l)
            target_l = np.clip(target_l, 0, 255).astype(np.uint8)
            
            # Replace L channel
            target_lab[:,:,0] = target_l
            
            # Convert back to BGR
            adjusted_target = cv2.cvtColor(target_lab, cv2.COLOR_LAB2BGR)
            
            return adjusted_target
            
        except Exception as e:
            logger.warning(f"Error in lighting transfer: {e}")
            return target