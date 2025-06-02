#!/usr/bin/env python3
"""
Batch Processing System for FaceSwapPro
Author: kaaskoek232
Date: 2025-06-02
"""

import os
import time
import json
import threading
import queue
import logging
import uuid
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, asdict
from enum import Enum
import cv2
import numpy as np

logger = logging.getLogger("FaceSwapPro-BatchProcessor")

class JobStatus(Enum):
    """Job status enum"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"

@dataclass
class Job:
    """Batch processing job"""
    id: str
    name: str
    type: str  # "image" or "video"
    source_path: str
    target_path: str
    output_path: str
    settings: Dict[str, Any]
    status: JobStatus = JobStatus.PENDING
    progress: float = 0.0
    created_at: float = 0.0
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    error_message: Optional[str] = None
    preview_path: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        result = asdict(self)
        result["status"] = self.status.value
        return result
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Job':
        """Create job from dictionary"""
        data = data.copy()
        data["status"] = JobStatus(data["status"])
        return cls(**data)

class BatchProcessor:
    """
    Advanced batch processing system for handling multiple face swapping tasks.
    Manages job queue, persistence, and parallel processing.
    """
    
    def __init__(
        self,
        processor,
        max_parallel_jobs: int = 1,
        queue_file: str = "jobs.json",
        temp_dir: str = "temp",
        auto_start: bool = True
    ):
        """
        Initialize batch processor.
        
        Args:
            processor: FaceSwapProcessor instance
            max_parallel_jobs: Maximum number of parallel jobs
            queue_file: Path to job queue file
            temp_dir: Path to temp directory
            auto_start: Automatically start processing
        """
        self.processor = processor
        self.max_parallel_jobs = max_parallel_jobs
        self.queue_file = queue_file
        self.temp_dir = temp_dir
        
        # Create temp directory if it doesn't exist
        os.makedirs(temp_dir, exist_ok=True)
        
        # Job queue and processing state
        self.job_queue = queue.Queue()
        self.jobs = {}  # id -> Job
        self.active_jobs = {}  # id -> thread
        self.running = False
        self.lock = threading.RLock()
        
        # Load existing jobs
        self._load_jobs()
        
        # Start processing if auto_start
        if auto_start:
            self.start()
            
    def _load_jobs(self):
        """Load jobs from queue file"""
        if not os.path.exists(self.queue_file):
            return
            
        try:
            with open(self.queue_file, 'r') as f:
                jobs_data = json.load(f)
                
            for job_data in jobs_data:
                job = Job.from_dict(job_data)
                self.jobs[job.id] = job
                
                # Re-queue pending and processing jobs
                if job.status in [JobStatus.PENDING, JobStatus.PROCESSING]:
                    job.status = JobStatus.PENDING
                    job.progress = 0.0
                    job.started_at = None
                    self.job_queue.put(job.id)
                    
            logger.info(f"Loaded {len(self.jobs)} jobs from queue file")
            
        except Exception as e:
            logger.error(f"Error loading jobs: {e}")
            
    def _save_jobs(self):
        """Save jobs to queue file"""
        try:
            with self.lock:
                jobs_data = [job.to_dict() for job in self.jobs.values()]
                
            with open(self.queue_file, 'w') as f:
                json.dump(jobs_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving jobs: {e}")
            
    def add_job(
        self,
        name: str,
        type: str,
        source_path: str,
        target_path: str,
        output_path: Optional[str] = None,
        settings: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Add a job to the queue.
        
        Args:
            name: Job name
            type: Job type ("image" or "video")
            source_path: Path to source face image
            target_path: Path to target image or video
            output_path: Path to output file (or auto-generated if None)
            settings: Job settings
            
        Returns:
            Job ID
        """
        if type not in ["image", "video"]:
            raise ValueError(f"Invalid job type: {type}")
            
        # Generate output path if not provided
        if not output_path:
            ext = os.path.splitext(target_path)[1]
            output_path = os.path.join(self.temp_dir, f"output_{uuid.uuid4()}{ext}")
            
        # Create job
        job_id = str(uuid.uuid4())
        job = Job(
            id=job_id,
            name=name,
            type=type,
            source_path=source_path,
            target_path=target_path,
            output_path=output_path,
            settings=settings or {},
            created_at=time.time()
        )
        
        # Add job to queue
        with self.lock:
            self.jobs[job_id] = job
            self.job_queue.put(job_id)
            self._save_jobs()
            
        logger.info(f"Added job: {name} (ID: {job_id})")
        return job_id
        
    def start(self):
        """Start batch processing"""
        with self.lock:
            if self.running:
                return
                
            self.running = True
            
        # Start worker thread
        worker_thread = threading.Thread(target=self._process_queue)
        worker_thread.daemon = True
        worker_thread.start()
        
        logger.info("Batch processor started")
        
    def stop(self):
        """Stop batch processing"""
        with self.lock:
            self.running = False
            
        logger.info("Batch processor stopped")
        
    def get_job(self, job_id: str) -> Optional[Job]:
        """Get job by ID"""
        with self.lock:
            return self.jobs.get(job_id)
            
    def get_jobs(self, status: Optional[JobStatus] = None) -> List[Job]:
        """Get all jobs, optionally filtered by status"""
        with self.lock:
            if status:
                return [job for job in self.jobs.values() if job.status == status]
            return list(self.jobs.values())
            
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a job"""
        with self.lock:
            job = self.jobs.get(job_id)
            if not job:
                return False
                
            # If job is active, interrupt the thread
            if job_id in self.active_jobs:
                # We can't really stop the thread, but we can mark it as canceled
                # and it will check this status periodically
                job.status = JobStatus.CANCELED
                return True
                
            # If job is pending, remove from queue
            if job.status == JobStatus.PENDING:
                # We can't remove from Queue directly, so mark as canceled
                job.status = JobStatus.CANCELED
                self._save_jobs()
                return True
                
            return False
            
    def _process_queue(self):
        """Process jobs in the queue"""
        while True:
            # Check if processing is stopped
            if not self.running:
                break
                
            # Check if we can process more jobs
            with self.lock:
                active_count = len(self.active_jobs)
                if active_count >= self.max_parallel_jobs:
                    time.sleep(0.5)
                    continue
                    
            try:
                # Get next job ID (non-blocking)
                try:
                    job_id = self.job_queue.get(block=False)
                except queue.Empty:
                    time.sleep(0.5)
                    continue
                    
                # Get job
                with self.lock:
                    job = self.jobs.get(job_id)
                    
                # Skip canceled or completed jobs
                if not job or job.status in [JobStatus.CANCELED, JobStatus.COMPLETED, JobStatus.FAILED]:
                    self.job_queue.task_done()
                    continue
                    
                # Start job processing thread
                job_thread = threading.Thread(target=self._process_job, args=(job,))
                job_thread.daemon = True
                
                # Update job status and save
                with self.lock:
                    job.status = JobStatus.PROCESSING
                    job.started_at = time.time()
                    self.active_jobs[job_id] = job_thread
                    self._save_jobs()
                    
                # Start job thread
                job_thread.start()
                
            except Exception as e:
                logger.error(f"Error in job processor: {e}")
                time.sleep(1)
                
    def _process_job(self, job: Job):
        """Process a single job"""
        logger.info(f"Processing job: {job.name} (ID: {job.id})")
        
        try:
            # Define progress callback
            def progress_callback(current, total):
                progress = current / total if total > 0 else 0
                job.progress = progress
                
            # Check source and target exist
            if not os.path.exists(job.source_path):
                raise FileNotFoundError(f"Source file not found: {job.source_path}")
                
            if not os.path.exists(job.target_path):
                raise FileNotFoundError(f"Target file not found: {job.target_path}")
                
            # Load source image
            source_image = cv2.imread(job.source_path)
            if source_image is None:
                raise ValueError(f"Failed to load source image: {job.source_path}")
                
            # Add source face to processor
            self.processor.add_face_to_database('source', source_image)
            source_embedding = self.processor.face_database.get('source')
            
            # Process job based on type
            if job.type == "image":
                # Load target image
                target_image = cv2.imread(job.target_path)
                if target_image is None:
                    raise ValueError(f"Failed to load target image: {job.target_path}")
                    
                # Detect faces
                target_faces = self.processor.detect_faces(target_image)
                
                # Process image
                result = self.processor.process_frame(
                    frame=target_image,
                    source_embedding=source_embedding,
                    target_faces=target_faces,
                    **job.settings
                )
                
                # Save result
                cv2.imwrite(job.output_path, result)
                
                # Generate preview if it doesn't exist
                if not job.preview_path:
                    preview_path = os.path.join(self.temp_dir, f"preview_{job.id}.jpg")
                    preview = cv2.resize(result, (640, 480))
                    cv2.imwrite(preview_path, preview)
                    job.preview_path = preview_path
                
            elif job.type == "video":
                # Process video
                self.processor.process_video(
                    source_face=source_image,
                    target_video=job.target_path,
                    output_path=job.output_path,
                    progress_callback=progress_callback,
                    **job.settings
                )
                
                # Generate preview if it doesn't exist
                if not job.preview_path:
                    preview_path = os.path.join(self.temp_dir, f"preview_{job.id}.jpg")
                    
                    # Extract a frame from the output video
                    cap = cv2.VideoCapture(job.output_path)
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    cap.set(cv2.CAP_PROP_POS_FRAMES, min(100, frame_count // 4))
                    ret, frame = cap.read()
                    cap.release()
                    
                    if ret:
                        preview = cv2.resize(frame, (640, 480))
                        cv2.imwrite(preview_path, preview)
                        job.preview_path = preview_path
                        
            # Update job status
            with self.lock:
                job.status = JobStatus.COMPLETED
                job.completed_at = time.time()
                job.progress = 1.0
                
                # Remove from active jobs
                self.active_jobs.pop(job.id, None)
                self._save_jobs()
                
            logger.info(f"Job completed: {job.name} (ID: {job.id})")
            
        except Exception as e:
            logger.error(f"Error processing job: {job.name} (ID: {job.id}) - {str(e)}")
            
            # Update job status
            with self.lock:
                job.status = JobStatus.FAILED
                job.error_message = str(e)
                job.completed_at = time.time()
                
                # Remove from active jobs
                self.active_jobs.pop(job.id, None)
                self._save_jobs()
                
        finally:
            # Mark job as done in queue
            self.job_queue.task_done()