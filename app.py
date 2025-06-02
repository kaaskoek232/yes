#!/usr/bin/env python3
"""
FaceSwapPro - Advanced GUI for next-generation face swapping
Author: kaaskoek232
Date: 2025-06-02 19:08:43
"""

# ... (previous code remains the same) ...

# In the FaceSwapProUI class, continuing with the save_config method:

    def save_config(self):
        """Save configuration to file"""
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), CONFIG_FILE)
        
        try:
            with open(config_path, 'w') as f:
                json.dump(self.config, f, indent=4)
        except Exception as e:
            logger.error(f"Error saving configuration: {str(e)}")
            
    def reset_settings(self):
        """Reset settings to defaults"""
        # Default configuration
        self.config = {
            "theme": DEFAULT_THEME,
            "resolution": 512,
            "swap_all": True,
            "output_dir": "",
            "use_gpu": CUDA_AVAILABLE,
            "face_swap_model": "inswapper_128",
            "tile_blending": True,
            "adaptive_tiling": True,
            "detail_enhancement": True,
            "perceptual_correction": True,
            "overlap_percent": 0.15,
            "detail_factor": 0.5,
            "expression_method": "NeuroMotion",
            "enable_expression": True,
            "expression_strength": 0.7,
            "preserve_identity": 0.8,
            "temporal_smoothing": True,
            "occlusion_method": "MATCHformer",
            "enable_occlusion": True,
            "occlusion_threshold": 0.5,
            "occlusion_refinement": True,
            "enable_motion_tracking": True,
            "enable_temporal_smoothing": True,
            "processing_threads": 4,
            "output_format": "MP4 (H.264)",
            "output_quality": 80
        }
        
        # Apply default theme
        self.apply_theme(DEFAULT_THEME)
        
        # Notify user
        QMessageBox.information(self, "Settings Reset", "All settings have been reset to defaults.")
        
    def init_processor(self):
        """Initialize the face swapping processor"""
        try:
            # Use GPU if available and enabled
            execution_provider = 'cuda' if CUDA_AVAILABLE and self.config.get("use_gpu", True) else 'cpu'
            
            # Get face swap model path
            model_name = self.config.get("face_swap_model", "inswapper_128")
            model_path = os.path.join(MODELS_DIR, f"{model_name}.onnx")
            
            # Check if model exists
            if not os.path.exists(model_path):
                logger.warning(f"Model not found: {model_path}, will attempt to download")
                # In a real implementation, you'd add download logic here
                
            # Configure the model
            if 'inswapper' in model_name.lower():
                model_config = ModelConfig(
                    path=model_path,
                    size=(128, 128),
                    mean=[0.0, 0.0, 0.0],
                    std=[1.0, 1.0, 1.0],
                    model_type='inswapper',
                    fp16_support='fp16' in model_name.lower()
                )
            elif 'simswap' in model_name.lower():
                model_config = ModelConfig(
                    path=model_path,
                    size=(256, 256) if '256' in model_name else (512, 512),
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
            if self.config.get("enable_expression", True):
                self.expression_transfer = ExpressionTransfer(
                    method=self.config.get("expression_method", "neuromotion").lower(),
                    execution_provider=execution_provider,
                    strength=self.config.get("expression_strength", 0.7),
                    preserve_identity=self.config.get("preserve_identity", 0.8),
                    enhance_details=self.config.get("detail_enhancement", True)
                )
            else:
                self.expression_transfer = None
                
            # Initialize occlusion handler if enabled
            if self.config.get("enable_occlusion", True):
                self.occlusion_handler = OcclusionHandler(
                    method=self.config.get("occlusion_method", "matchformer").lower(),
                    threshold=self.config.get("occlusion_threshold", 0.5),
                    refinement=self.config.get("occlusion_refinement", True),
                    temporal_smoothing=self.config.get("temporal_smoothing", True)
                )
            else:
                self.occlusion_handler = None
                
            # Initialize the processor
            self.processor = FaceSwapProcessor(
                face_detector_model='yolov12-face',
                face_recognizer_model='arcface_2025',
                face_swap_model=model_path,
                execution_provider=execution_provider,
                gpu_id=0,
                detection_threshold=0.5,
                recognition_threshold=0.7,
                target_resolution=int(self.config.get("resolution", 512)),
                enable_motion_tracking=self.config.get("enable_motion_tracking", True),
                enable_temporal_smoothing=self.config.get("enable_temporal_smoothing", True)
            )
            
            logger.info("Processor initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing processor: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to initialize processor: {str(e)}")
            
    def load_source_image(self):
        """Load a source image file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Source Image", 
            str(Path.home()), 
            "Images (*.png *.jpg *.jpeg *.bmp)"
        )
        
        if not file_path:
            return
            
        try:
            # Load the image using OpenCV
            image = cv2.imread(file_path)
            
            if image is None:
                QMessageBox.critical(self, "Error", "Failed to load image")
                return
                
            # Store the source image
            self.source_image = image
            
            # Display the source image in the preview
            self.display_source_preview(image)
            
            # Update status
            self.statusBar.showMessage(f"Source image loaded: {os.path.basename(file_path)}")
            
            # Auto-detect faces
            self.detect_source_faces()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error loading source image: {str(e)}")
            
    def display_source_preview(self, image):
        """Display the source image preview"""
        try:
            # Convert to RGB for Qt
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            
            # Scale image to fit preview while maintaining aspect ratio
            preview_width = self.source_preview.width()
            preview_height = self.source_preview.height()
            
            # Calculate scaling factor
            scale = min(preview_width / w, preview_height / h)
            new_width = int(w * scale)
            new_height = int(h * scale)
            
            # Resize the image
            resized = cv2.resize(rgb_image, (new_width, new_height))
            
            # Convert to QImage and then to QPixmap
            bytes_per_line = ch * new_width
            q_img = QImage(resized.data, new_width, new_height, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)
            
            # Display in the preview label
            self.source_preview.setPixmap(pixmap)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error displaying preview: {str(e)}")
            
    def detect_source_faces(self):
        """Detect faces in the source image"""
        if self.source_image is None:
            QMessageBox.warning(self, "Warning", "No source image loaded")
            return
            
        try:
            # Use the processor to detect faces
            if not self.processor:
                self.init_processor()
                
            self.statusBar.showMessage("Detecting faces in source image...")
            
            # Detect faces
            faces = self.processor.detect_faces(self.source_image)
            
            if not faces:
                QMessageBox.warning(self, "Warning", "No faces detected in the source image")
                return
                
            # Store detected faces
            self.detected_source_faces = faces
            
            # Display image with detected faces
            image_with_faces = self.source_image.copy()
            
            for i, face in enumerate(faces):
                bbox = face.get('bbox', [0, 0, 0, 0])
                x1, y1, x2, y2 = [int(coord) for coord in bbox]
                
                # Draw face rectangle
                cv2.rectangle(image_with_faces, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image_with_faces, f"Face {i+1}", (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
            # Update the preview
            self.display_source_preview(image_with_faces)
            
            # Update status
            self.statusBar.showMessage(f"Detected {len(faces)} faces in source image")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error detecting faces: {str(e)}")
            
    def load_target_media(self, is_video=False):
        """Load a target image or video file"""
        if is_video:
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Open Target Video", 
                str(Path.home()), 
                "Videos (*.mp4 *.avi *.mov *.mkv)"
            )
        else:
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Open Target Image", 
                str(Path.home()), 
                "Images (*.png *.jpg *.jpeg *.bmp)"
            )
            
        if not file_path:
            return
            
        try:
            if is_video:
                # Load the video
                self.target_video = file_path
                self.target_image = None
                
                # Show video in media player
                self.media_player.load_video(file_path)
                self.right_panel.setCurrentWidget(self.media_player)
                
                # Update status
                self.statusBar.showMessage(f"Target video loaded: {os.path.basename(file_path)}")
                
            else:
                # Load the image using OpenCV
                image = cv2.imread(file_path)
                
                if image is None:
                    QMessageBox.critical(self, "Error", "Failed to load image")
                    return
                    
                # Store the target image
                self.target_image = image
                self.target_video = None
                
                # Display the target image
                self.image_view.set_image(image)
                self.right_panel.setCurrentWidget(self.image_view)
                
                # Update status
                self.statusBar.showMessage(f"Target image loaded: {os.path.basename(file_path)}")
                
                # Auto-detect faces
                self.detect_target_faces()
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error loading target media: {str(e)}")
            
    def detect_target_faces(self):
        """Detect faces in the target image/video frame"""
        if self.target_image is None and self.target_video is None:
            QMessageBox.warning(self, "Warning", "No target media loaded")
            return
            
        try:
            # Use the processor to detect faces
            if not self.processor:
                self.init_processor()
                
            self.statusBar.showMessage("Detecting faces in target media...")
            
            if self.target_image is not None:
                # Detect faces in image
                target_image = self.target_image
            else:
                # Get current frame from video
                cap = cv2.VideoCapture(self.target_video)
                
                # If media player is showing a specific frame, use that
                current_frame = self.media_player.get_current_frame()
                if current_frame > 0:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
                    
                ret, target_image = cap.read()
                cap.release()
                
                if not ret:
                    QMessageBox.warning(self, "Warning", "Failed to extract frame from video")
                    return
            
            # Detect faces
            faces = self.processor.detect_faces(target_image)
            
            if not faces:
                QMessageBox.warning(self, "Warning", "No faces detected in the target media")
                return
                
            # Store detected faces
            self.detected_target_faces = faces
            
            # Display in the face detection widget
            self.face_detection.set_image(target_image, faces)
            self.right_panel.setCurrentWidget(self.face_detection)
            
            # Update status
            self.statusBar.showMessage(f"Detected {len(faces)} faces in target media")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error detecting faces: {str(e)}")
            
    def start_processing(self):
        """Start processing the source and target media"""
        # Check if source and target are selected
        if self.source_image is None:
            QMessageBox.warning(self, "Warning", "Please select a source image first")
            return
            
        if self.target_image is None and self.target_video is None:
            QMessageBox.warning(self, "Warning", "Please select a target image or video first")
            return
            
        # Check if processor is initialized
        if not self.processor:
            self.init_processor()
            
        # Prepare processing settings
        settings = {
            'swap_all': self.face_selection_combo.currentText() == "All Faces",
            'resolution': self.resolution_combo.currentText(),
        }
        
        # If specific face is selected, add its ID
        if not settings['swap_all'] and self.face_detection.selected_face >= 0:
            selected_face = self.face_detection.get_selected_face()
            if selected_face:
                settings['target_ids'] = [selected_face.get('track_id', self.face_detection.selected_face)]
                
        # Determine output path
        if self.target_video:
            # For video, create output path based on target
            filename, ext = os.path.splitext(self.target_video)
            output_format = self.config.get("output_format", "MP4 (H.264)")
            
            if output_format == "MP4 (H.264)":
                ext = ".mp4"
            elif output_format == "AVI":
                ext = ".avi"
            elif output_format == "MKV":
                ext = ".mkv"
                
            self.output_path = f"{filename}_swapped{ext}"
            
            # Use custom output directory if specified
            output_dir = self.config.get("output_dir", "")
            if output_dir and os.path.isdir(output_dir):
                self.output_path = os.path.join(output_dir, os.path.basename(self.output_path))
                
        else:
            # For image, let user choose output path
            default_dir = self.config.get("output_dir", "") or os.path.dirname(self.target_image)
            default_name = os.path.splitext(os.path.basename(self.target_image))[0] + "_swapped.png"
            default_path = os.path.join(default_dir, default_name)
            
            self.output_path, _ = QFileDialog.getSaveFileName(
                self, "Save Output Image", 
                default_path, 
                "Images (*.png *.jpg *.jpeg)"
            )
            
            if not self.output_path:
                return  # User canceled
                
        # Update UI
        self.process_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        
        # Add output path to settings
        settings['output_path'] = self.output_path
        
        # Start processing
        if self.target_video:
            self.process_video(settings)
        else:
            self.process_image(settings)
            
    def process_image(self, settings):
        """Process a single image"""
        try:
            # Create and start processing thread
            self.processing_thread = ImageProcessingThread(
                processor=self.processor,
                source_image=self.source_image,
                target_image=self.target_image,
                settings=settings
            )
            
            # Connect signals
            self.processing_thread.update_result.connect(self.on_image_processed)
            self.processing_thread.process_finished.connect(self.on_processing_finished)
            
            # Start processing
            self.statusBar.showMessage("Processing image...")
            self.processing_thread.start()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error processing image: {str(e)}")
            self.on_processing_finished(False, str(e))
            
    def process_video(self, settings):
        """Process a video"""
        try:
            # Create and start processing thread
            self.processing_thread = VideoThread(
                processor=self.processor,
                source_image=self.source_image,
                video_path=self.target_video,
                output_path=self.output_path,
                settings=settings
            )
            
            # Connect signals
            self.processing_thread.update_progress.connect(self.on_progress_update)
            self.processing_thread.update_frame.connect(self.on_frame_update)
            self.processing_thread.process_finished.connect(self.on_processing_finished)
            
            # Start processing
            self.statusBar.showMessage("Processing video...")
            self.processing_thread.start()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error processing video: {str(e)}")
            self.on_processing_finished(False, str(e))
            
    def on_progress_update(self, current, total):
        """Update progress bar during video processing"""
        progress = int(100 * current / total) if total > 0 else 0
        self.progress_bar.setValue(progress)
        self.statusBar.showMessage(f"Processing frame {current} of {total}...")
        
    def on_frame_update(self, frame):
        """Display a preview frame during video processing"""
        # Convert to RGB for Qt
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        
        # Create QImage and QPixmap
        q_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        
        # Display in the media player
        self.media_player.display.setPixmap(pixmap.scaled(
            self.media_player.display.width(),
            self.media_player.display.height(),
            Qt.AspectRatioMode.KeepAspectRatio
        ))
        
    def on_image_processed(self, result_image):
        """Display processed image result"""
        # Display the result
        self.image_view.set_image(result_image)
        self.right_panel.setCurrentWidget(self.image_view)
        
    def on_processing_finished(self, success, result):
        """Handle completion of processing"""
        # Reset UI
        self.process_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.progress_bar.setVisible(False)
        
        if success:
            self.statusBar.showMessage(f"Processing completed successfully. Saved to: {os.path.basename(result)}")
            
            # Ask if user wants to open the result
            if QMessageBox.question(
                self, "Processing Complete", 
                "Processing completed successfully.\nDo you want to open the result?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            ) == QMessageBox.StandardButton.Yes:
                try:
                    # Use system default application to open the result
                    if sys.platform == 'win32':
                        os.startfile(result)
                    elif sys.platform == 'darwin':  # macOS
                        os.system(f'open "{result}"')
                    else:  # Linux
                        os.system(f'xdg-open "{result}"')
                except Exception as e:
                    logger.error(f"Error opening result file: {str(e)}")
        else:
            self.statusBar.showMessage(f"Processing failed: {result}")
            
    def cancel_processing(self):
        """Cancel ongoing processing"""
        if self.processing_thread and self.processing_thread.isRunning():
            # Stop the thread
            if hasattr(self.processing_thread, 'stop'):
                self.processing_thread.stop()
                
            self.processing_thread.wait(1000)  # Wait for 1 second
            
            if self.processing_thread.isRunning():
                self.processing_thread.terminate()
                
            self.processing_thread = None
            
            # Reset UI
            self.process_btn.setEnabled(True)
            self.cancel_btn.setEnabled(False)
            self.progress_bar.setVisible(False)
            self.statusBar.showMessage("Processing canceled")
    
    def save_result(self):
        """Save the current result to a file"""
        # Not implemented in this version
        QMessageBox.information(self, "Not Implemented", "This feature is not implemented in this version.")
    
    def show_tutorial(self):
        """Show tutorial dialog"""
        dialog = QDialog(self)
        dialog.setWindowTitle("FaceSwapPro Tutorial")
        dialog.resize(800, 600)
        
        # Dialog layout
        layout = QVBoxLayout(dialog)
        
        # Create tutorial content
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        
        tutorial_widget = QWidget()
        tutorial_layout = QVBoxLayout(tutorial_widget)
        
        # Add tutorial sections
        tutorial_layout.addWidget(self._create_tutorial_section(
            "Getting Started",
            "Welcome to FaceSwapPro, the most advanced face swapping application. "
            "This tutorial will guide you through the basic workflow."
        ))
        
        tutorial_layout.addWidget(self._create_tutorial_section(
            "Step 1: Load Source Image",
            "Start by clicking 'Load Source' to select an image containing the face "
            "you want to use as the source. The source face will be detected automatically."
        ))
        
        tutorial_layout.addWidget(self._create_tutorial_section(
            "Step 2: Load Target Media",
            "Click 'Load Image' or 'Load Video' to select your target media. "
            "This is where the source face will be applied."
        ))
        
        tutorial_layout.addWidget(self._create_tutorial_section(
            "Step 3: Detect Faces",
            "Use the 'Detect Faces' button to find faces in your target media. "
            "You can then select which face(s) to swap."
        ))
        
        tutorial_layout.addWidget(self._create_tutorial_section(
            "Step 4: Adjust Settings",
            "Set the desired resolution and choose whether to swap all detected faces "
            "or just the selected one. Click 'Advanced Settings' for more options."
        ))
        
        tutorial_layout.addWidget(self._create_tutorial_section(
            "Step 5: Process",
            "Click the 'Process' button to start face swapping. For videos, you'll see "
            "the progress and a preview during processing."
        ))
        
        tutorial_layout.addWidget(self._create_tutorial_section(
            "Advanced Features",
            "FaceSwapPro includes state-of-the-art features like expression transfer, "
            "occlusion handling, and enhanced PixelBoost technology for high-resolution results."
        ))
        
        # Add to scroll area
        scroll_area.setWidget(tutorial_widget)
        layout.addWidget(scroll_area)
        
        # Close button
        close_button = QPushButton("Close")
        close_button.clicked.connect(dialog.accept)
        layout.addWidget(close_button, alignment=Qt.AlignmentFlag.AlignRight)
        
        # Show dialog
        dialog.exec()
        
    def _create_tutorial_section(self, title, content):
        """Create a section for the tutorial dialog"""
        section = QGroupBox(title)
        section.setStyleSheet("QGroupBox { font-weight: bold; }")
        
        layout = QVBoxLayout(section)
        
        content_label = QLabel(content)
        content_label.setWordWrap(True)
        content_label.setTextFormat(Qt.TextFormat.RichText)
        
        layout.addWidget(content_label)
        
        return section
    
    def show_about(self):
        """Show about dialog"""
        QMessageBox.about(
            self,
            f"About {APP_NAME}",
            f"<h2>{APP_NAME} v{APP_VERSION}</h2>"
            f"<p>Author: {APP_AUTHOR}</p>"
            f"<p>Date: 2025-06-02 19:08:43</p>"
            "<p>Advanced face swapping application with state-of-the-art features:</p>"
            "<ul>"
            "<li>Enhanced PixelBoost technology</li>"
            "<li>Expression transfer with NeuroMotion</li>"
            "<li>Occlusion handling with MATCHformer</li>"
            "<li>YOLO v12 face detection</li>"
            "</ul>"
            "<p>Copyright © 2025 All rights reserved.</p>"
        )
    
    def closeEvent(self, event):
        """Handle application close event"""
        # Save configuration
        self.save_config()
        
        # Clean up resources
        if self.media_player:
            self.media_player.release()
            
        if self.processing_thread and self.processing_thread.isRunning():
            self.processing_thread.stop()
            self.processing_thread.wait(1000)
            
        event.accept()


def main():
    """Main application entry point"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="FaceSwapPro - Advanced face swapping application")
    parser.add_argument("--source", help="Source face image path")
    parser.add_argument("--target", help="Target image or video path")
    parser.add_argument("--output", help="Output path")
    parser.add_argument("--resolution", type=int, help="Processing resolution")
    parser.add_argument("--cpu", action="store_true", help="Force CPU processing")
    args = parser.parse_args()
    
    # Create application
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setApplicationVersion(APP_VERSION)
    
    # Create main window
    main_window = FaceSwapProUI(vars(args))
    main_window.show()
    
    # Run application
    sys.exit(app.exec())


if __name__ == "__main__":
    main()