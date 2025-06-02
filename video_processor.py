import cv2
import numpy as np
import os
import time
from typing import List, Dict, Tuple, Optional, Callable, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import threading

class FrameBuffer:
    """Thread-safe frame buffer for video processing"""
    
    def __init__(self, max_size: int = 30):
        self.buffer = []
        self.max_size = max_size
        self.lock = threading.Lock()
        
    def put(self, frame: np.ndarray) -> None:
        """Add a frame to the buffer"""
        with self.lock:
            self.buffer.append(frame)
            # Keep buffer size under limit
            while len(self.buffer) > self.max_size:
                self.buffer.pop(0)
                
    def get(self) -> Optional[np.ndarray]:
        """Get the oldest frame from buffer"""
        with self.lock:
            if not self.buffer:
                return None
            return self.buffer.pop(0)
            
    def peek(self) -> Optional[np.ndarray]:
        """View the oldest frame without removing"""
        with self.lock:
            if not self.buffer:
                return None
            return self.buffer[0]
            
    def size(self) -> int:
        """Get current buffer size"""
        with self.lock:
            return len(self.buffer)
            
    def clear(self) -> None:
        """Clear the buffer"""
        with self.lock:
            self.buffer.clear()


class VideoProcessor:
    """
    Advanced video processing with face swapping and object detection.
    """
    
    def __init__(
        self,
        face_processor,
        pixel_boost,
        object_detector=None,
        max_workers: int = 4,
        use_threading: bool = True
    ):
        """
        Initialize the video processor.
        
        Args:
            face_processor: FaceProcessor instance
            pixel_boost: EnhancedPixelBoost instance
            object_detector: RF-DETR instance (optional)
            max_workers: Maximum number of worker threads
            use_threading: Whether to use threaded processing
        """
        self.face_processor = face_processor
        self.pixel_boost = pixel_boost
        self.object_detector = object_detector
        self.max_workers = max_workers
        self.use_threading = use_threading
        
        # Initialize frame buffers
        self.input_buffer = FrameBuffer(max_size=30)
        self.output_buffer = FrameBuffer(max_size=30)
        
        # Processing flags
        self.is_processing = False
        self.processing_thread = None
        
        # Statistics
        self.stats = {
            'fps': 0,
            'processing_time': 0,
            'frames_processed': 0,
            'faces_detected': 0,
            'objects_detected': 0
        }
        
    def start_processing(self) -> None:
        """Start background processing thread"""
        if self.is_processing:
            return
            
        self.is_processing = True
        self.processing_thread = threading.Thread(target=self._process_frames)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
    def stop_processing(self) -> None:
        """Stop background processing"""
        self.is_processing = False
        if self.processing_thread:
            self.processing_thread.join(timeout=1.0)
            self.processing_thread = None
            
    def _process_frames(self) -> None:
        """Background frame processing loop"""
        while self.is_processing:
            # Check if we have frames to process
            if self.input_buffer.size() == 0:
                time.sleep(0.01)  # Small sleep to prevent busy waiting
                continue
                
            # Get a frame to process
            frame = self.input_buffer.get()
            if frame is None:
                continue
                
            # Process the frame
            start_time = time.time()
            processed_frame = self._process_single_frame(frame)
            self.stats['processing_time'] = time.time() - start_time
            
            # Add to output buffer
            self.output_buffer.put(processed_frame)
            self.stats['frames_processed'] += 1
            self.stats['fps'] = 1.0 / max(self.stats['processing_time'], 0.001)
            
    def _process_single_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process a single frame with face swapping and object detection.
        
        Args:
            frame: Input BGR frame
            
        Returns:
            Processed BGR frame
        """
        result_frame = frame.copy()
        
        # Detect faces
        faces = self.face_processor.detect_faces(frame)
        self.stats['faces_detected'] = len(faces)
        
        # Apply face swapping to each detected face if source embedding exists
        source_embedding = getattr(self, 'source_embedding', None)
        if source_embedding is not None:
            for face in faces:
                # Extract face area with padding
                bbox = face['bbox']
                x1, y1, x2, y2 = [int(coord) for coord in bbox]
                
                # Add padding
                padding_factor = 0.2
                face_width = x2 - x1
                face_height = y2 - y1
                
                # Ensure square crop with padding
                size = max(face_width, face_height)
                size_with_padding = int(size * (1 + padding_factor))
                
                # Calculate center and new coordinates
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                
                new_x1 = max(0, center_x - size_with_padding // 2)
                new_y1 = max(0, center_y - size_with_padding // 2)
                new_x2 = min(frame.shape[1], center_x + size_with_padding // 2)
                new_y2 = min(frame.shape[0], center_y + size_with_padding // 2)
                
                # Extract face area
                face_area = frame[new_y1:new_y2, new_x1:new_x2]
                
                # Skip if face area is invalid
                if face_area.size == 0 or face_area.shape[0] == 0 or face_area.shape[1] == 0:
                    continue
                    
                # Determine target resolution based on face size
                # Use higher resolution for larger faces
                face_size = max(new_x2 - new_x1, new_y2 - new_y1)
                if face_size >= 400:
                    target_resolution = 1024
                elif face_size >= 200:
                    target_resolution = 512
                else:
                    target_resolution = 256
                
                # Resize to target resolution
                face_resized = cv2.resize(face_area, (target_resolution, target_resolution))
                
                # Save original for later refinement
                original_face = face_resized.copy()
                
                # Apply face swap with pixel boost
                try:
                    swapped_face = self.pixel_boost.swap_face(
                        source_embedding=source_embedding,
                        target_frame=face_resized,
                        target_resolution=target_resolution,
                        source_id="main_source_face"
                    )
                    
                    # Apply refinements
                    refined_face = self.pixel_boost.add_face_refinements(swapped_face, original_face)
                    
                    # Resize back to original face size
                    final_face = cv2.resize(refined_face, (new_x2 - new_x1, new_y2 - new_y1))
                    
                    # Create a mask for smooth blending
                    mask = np.ones((new_y2 - new_y1, new_x2 - new_x1), dtype=np.float32)
                    
                    # Feather the edges
                    feather_amount = int((new_x2 - new_x1) * 0.05)  # 5% of face width
                    if feather_amount > 0:
                        mask = cv2.GaussianBlur(mask, (feather_amount * 2 + 1, feather_amount * 2 + 1), 0)
                        
                    # Expand mask to 3 channels
                    mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
                    
                    # Apply alpha blending
                    result_frame[new_y1:new_y2, new_x1:new_x2] = final_face * mask + result_frame[new_y1:new_y2, new_x1:new_x2] * (1 - mask)
                    
                except Exception as e:
                    print(f"Error in face swapping: {e}")
        
        # Perform object detection if available
        if self.object_detector is not None:
            try:
                detections = self.object_detector.detect(frame)
                self.stats['objects_detected'] = len(detections)
                
                # Draw detections on the result frame
                result_frame = self.object_detector.draw_detections(result_frame, detections)
            except Exception as e:
                print(f"Error in object detection: {e}")
                
        # Draw faces for visualization
        result_frame = self.face_processor.draw_faces(result_frame, faces)
        
        # Add stats
        self._add_stats_overlay(result_frame)
        
        return result_frame
    
    def _add_stats_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Add performance stats overlay to frame"""
        # Add semi-transparent overlay
        h, w = frame.shape[:2]
        overlay = frame.copy()
        
        # Create stats text
        stats_text = [
            f"FPS: {self.stats['fps']:.1f}",
            f"Processing: {self.stats['processing_time']*1000:.1f}ms",
            f"Faces: {self.stats['faces_detected']}",
            f"Objects: {self.stats.get('objects_detected', 0)}"
        ]
        
        # Draw background
        cv2.rectangle(overlay, (10, 10), (200, 20 + 20 * len(stats_text)), (0, 0, 0), -1)
        
        # Add text
        for i, text in enumerate(stats_text):
            cv2.putText(
                overlay, text, (15, 30 + 20 * i),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
            )
            
        # Add overlay to original frame
        alpha = 0.7
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        return frame
        
    def set_source_face(self, source_image_path: str) -> bool:
        """
        Set source face for swapping.
        
        Args:
            source_image_path: Path to source face image
            
        Returns:
            True if successful, False otherwise
        """
        if not os.path.exists(source_image_path):
            return False
            
        # Load the image
        source_img = cv2.imread(source_image_path)
        if source_img is None:
            return False
            
        # Detect faces
        faces = self.face_processor.detect_faces(source_img)
        if not faces:
            return False
            
        # Use the first detected face
        self.source_embedding = faces[0]['embedding']
        self.source_face_image = self.face_processor.extract_aligned_face(source_img, faces[0])
        return True
        
    def process_video(self, 
                     input_path: str, 
                     output_path: str, 
                     progress_callback: Optional[Callable[[float], None]] = None) -> bool:
        """
        Process a video file.
        
        Args:
            input_path: Path to input video
            output_path: Path to output video
            progress_callback: Optional callback function for progress updates
            
        Returns:
            True if successful, False otherwise
        """
        if not os.path.exists(input_path):
            return False
            
        # Open the video
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            return False
            
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Process frames
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process the frame
            processed = self._process_single_frame(frame)
            
            # Write to output
            out.write(processed)
            
            # Update progress
            frame_count += 1
            if progress_callback:
                progress = frame_count / total_frames
                progress_callback(progress)
                
        # Release resources
        cap.release()
        out.release()
        
        return True
        
    def process_webcam(self, 
                      camera_id: int = 0,
                      display: bool = True,
                      output_path: Optional[str] = None) -> None:
        """
        Process webcam stream.
        
        Args:
            camera_id: Camera device ID
            display: Whether to display output window
            output_path: Optional path to save video
        """
        # Open the webcam
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"Failed to open camera {camera_id}")
            return
            
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Create video writer if needed
        video_writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
        # Start processing in background thread
        self.start_processing()
        
        try:
            while True:
                # Read a frame
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Add to input buffer
                self.input_buffer.put(frame)
                
                # Try to get a processed frame
                processed_frame = self.output_buffer.get()
                
                if processed_frame is not None:
                    # Write to file if needed
                    if video_writer:
                        video_writer.write(processed_frame)
                        
                    # Display if needed
                    if display:
                        cv2.imshow('FaceSwapPro', processed_frame)
                        
                # Check for exit key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            # Clean up
            self.stop_processing()
            cap.release()
            if video_writer:
                video_writer.release()
            if display:
                cv2.destroyAllWindows()