from depth_yolo_pipeline import DepthYOLOPipeline
from audio_feedback import AudioFeedbackSystem
import cv2
import time

def main():
    print("=" * 50)
    print("BLIND ASSIST - NAVIGATION SYSTEM")
    print("=" * 50)
    
    # Initialize systems
    print("\n[1/3] Loading AI models...")
    pipeline = DepthYOLOPipeline(yolo_model_path="../models/yolov8.pt")
    
    print("\n[2/3] Initializing audio system...")
    audio_system = AudioFeedbackSystem()
    
    print("\n[3/3] Starting camera...")
    cap = cv2.VideoCapture(0)  # Change to 1 or 2 if camera 0 doesn't work
    
    if not cap.isOpened():
        print("ERROR: Could not open camera!")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("\n✓ System ready! Press 'q' to quit.\n")
    
    frame_count = 0
    process_every_n_frames = 3  # Process every 3rd frame for speed
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        frame_count += 1
        
        if frame_count % process_every_n_frames == 0:
            start_time = time.time()
            
            # Process frame with YOLO + Depth
            detections, depth_map, scene_type, scene_context = pipeline.process_frame(frame)
            
            # Voice feedback
            audio_system.process_detections(detections, scene_context)
            
            # Calculate FPS
            inference_time = time.time() - start_time
            fps = 1.0 / inference_time if inference_time > 0 else 0
            
            # Visualization (optional - can disable for production)
            result_frame = pipeline.visualize_results(frame, detections, depth_map)
            cv2.putText(result_frame, f"FPS: {fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(result_frame, f"Scene: {scene_type}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            cv2.imshow('Blind Assist Navigation', result_frame)
            
            # Console output
            if detections:
                print(f"\rDetections: {len(detections)} | FPS: {fps:.1f}  ", end='')
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("\n\nShutting down...")

if __name__ == "__main__":
    main()