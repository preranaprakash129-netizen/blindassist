import torch
import cv2
import numpy as np
from ultralytics import YOLO
import os
from scene_classifier import SceneClassifier

class DepthYOLOPipeline:
    def __init__(self, yolo_model_path="../models/yolov8.pt", use_gpu=True):
        """
        Initialize the pipeline with YOLO + Depth estimation
        
        Args:
            yolo_model_path: Path to YOLO model (relative to src/ folder)
            use_gpu: Use GPU if available
        """
        # Get absolute path to model
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        full_model_path = os.path.join(project_root, yolo_model_path.lstrip('../'))
        
        # Initialize YOLOv8
        print(f"Loading YOLO from: {full_model_path}")
        self.yolo_model = YOLO(full_model_path)
        
        
        # Initialize MiDaS depth estimation
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        print(f"Loading MiDaS depth model on {self.device}...")
        
        self.midas = torch.hub.load("intel-isl/MiDaS", "DPT_Large")
        self.midas.to(self.device)
        self.midas.eval()
        
        # Load transforms
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        self.transform = midas_transforms.dpt_transform
        # Initialize Scene Classifier
        print("Loading scene classification model...")
        self.scene_classifier = SceneClassifier(use_gpu=use_gpu)
        
        print("✓ All models loaded successfully!")
    
    def get_depth_at_bbox(self, depth_map, bbox):
        """Extract median depth within bounding box"""
        x1, y1, x2, y2 = map(int, bbox)
        
        h, w = depth_map.shape
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)
        
        roi_depth = depth_map[y1:y2, x1:x2]
        
        if roi_depth.size == 0:
            return None
        
        return np.median(roi_depth)
    
    def depth_to_distance(self, depth_value, depth_map):
        """
        Convert relative depth to approximate distance in meters
        CALIBRATE THIS for your specific camera!
        """
        min_depth = depth_map.min()
        max_depth = depth_map.max()
        
        if max_depth == min_depth:
            return None
        
        normalized_depth = (depth_value - min_depth) / (max_depth - min_depth)
        
        # TODO: Calibrate these values for your camera!
        min_distance = 0.5  # meters
        max_distance = 10.0  # meters
        
        distance = max_distance - (normalized_depth * (max_distance - min_distance))
        
        return distance
    
    def process_frame(self, frame):
        """
        Process single frame with YOLO + Depth
        
        Returns:
            detections: List of dicts with detection info
            depth_map: Raw depth map
        """
        # Run YOLO object detection
        yolo_results = self.yolo_model(frame, verbose=False)[0]
        
        # Run depth estimation
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_batch = self.transform(img_rgb).to(self.device)
        
        with torch.no_grad():
            depth_prediction = self.midas(input_batch)
            depth_prediction = torch.nn.functional.interpolate(
                depth_prediction.unsqueeze(1),
                size=frame.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        
        depth_map = depth_prediction.cpu().numpy()
        
        # Combine YOLO detections with depth information
        detections = []
        
        for box in yolo_results.boxes:
            bbox = box.xyxy[0].cpu().numpy()
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = yolo_results.names[class_id]
            
            depth_value = self.get_depth_at_bbox(depth_map, bbox)
            
            if depth_value is not None:
                distance = self.depth_to_distance(depth_value, depth_map)
            else:
                distance = None
            
            detections.append({
                'class': class_name,
                'confidence': confidence,
                'bbox': bbox,
                'depth_value': depth_value,
                'distance_m': distance,
                'class_id': class_id
            })
        
     # Classify scene
        scene_type, scene_confidence = self.scene_classifier.classify_scene(frame)
        scene_context = self.scene_classifier.get_scene_context(scene_type)
        
        return detections, depth_map, scene_type, scene_context   
    
    def visualize_results(self, frame, detections, depth_map):
        """Draw detections with depth info on frame"""
        # Create colorful depth visualization
        depth_colormap = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        depth_colormap = cv2.applyColorMap(depth_colormap, cv2.COLORMAP_MAGMA)
        
        annotated_frame = frame.copy()
        
        for det in detections:
            x1, y1, x2, y2 = map(int, det['bbox'])
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Create label with distance
            label = f"{det['class']}"
            if det['distance_m'] is not None:
                label += f" {det['distance_m']:.1f}m"
            label += f" {det['confidence']:.2f}"
            
            # Draw label background
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(annotated_frame, (x1, y1 - 20), (x1 + w, y1), (0, 255, 0), -1)
            cv2.putText(annotated_frame, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        # Combine camera view and depth map side by side
        combined = np.hstack([annotated_frame, depth_colormap])
        
        return combined