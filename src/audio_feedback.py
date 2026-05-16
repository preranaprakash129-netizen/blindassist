import pyttsx3
import threading
from collections import deque
import time

class AudioFeedbackSystem:
    def __init__(self):
        """Initialize text-to-speech engine for voice navigation"""
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 180)  # Speaking speed
        self.engine.setProperty('volume', 1.0)  # Volume (0.0 to 1.0)
        
        # Track announcements to avoid repetition
        self.announcement_queue = deque(maxlen=5)
        self.last_announcement_time = {}
        self.cooldown_seconds = 3  # Don't repeat same object within 3 seconds
        
        # Distance thresholds for priority
        self.CRITICAL_DISTANCE = 1.0  # meters - very close
        self.WARNING_DISTANCE = 2.5   # meters - close
        self.INFO_DISTANCE = 5.0      # meters - far but notable
        
        print("✓ Audio feedback system initialized")
    
    def get_priority(self, detection):
        """
        Calculate priority score for a detection
        Higher score = more important to announce
        """
        distance = detection.get('distance_m')
        class_name = detection['class']
        
        if distance is None:
            return 0
        
        priority = 0
        
        # Priority based on distance (closer = higher priority)
        if distance < self.CRITICAL_DISTANCE:
            priority += 100
        elif distance < self.WARNING_DISTANCE:
            priority += 50
        elif distance < self.INFO_DISTANCE:
            priority += 20
        
        # Priority based on object type
        critical_objects = ['person', 'car', 'truck', 'bicycle', 'motorcycle', 'bus']
        important_objects = ['chair', 'door', 'bench', 'potted plant', 'stairs', 'step']
        
        if class_name in critical_objects:
            priority += 30
        elif class_name in important_objects:
            priority += 10
        
        # Add scene-based priority boost
        scene_boost = detection.get('scene_boost', 0)
        priority += scene_boost
        
        
        
        return priority
    
    def should_announce(self, detection):
        """
        Check if this detection should be announced
        Prevents spam by using cooldown timer
        """
        class_name = detection['class']
        current_time = time.time()
        
        # Check if we announced this object recently
        if class_name in self.last_announcement_time:
            time_since_last = current_time - self.last_announcement_time[class_name]
            if time_since_last < self.cooldown_seconds:
                return False  # Too soon, don't announce
        
        # Check if priority is high enough
        priority = self.get_priority(detection)
        if priority < 20:  # Minimum priority threshold
            return False
        
        return True
    
    def generate_message(self, detection):
        """
        Generate natural language announcement for the detection
        """
        class_name = detection['class']
        distance = detection['distance_m']
        
        if distance is None:
            return f"{class_name} detected"
        
        # Choose urgency and description based on distance
        if distance < self.CRITICAL_DISTANCE:
            distance_desc = "very close"
            urgency = "Warning! "
        elif distance < self.WARNING_DISTANCE:
            distance_desc = f"{distance:.1f} meters ahead"
            urgency = ""
        else:
            distance_desc = f"far ahead at {distance:.0f} meters"
            urgency = ""
        
        # Clean up class name (remove underscores, capitalize)
        clean_name = class_name.replace('_', ' ').title()
        
        message = f"{urgency}{clean_name} {distance_desc}"
        
        return message
    
    def announce(self, message):
        """
        Speak message out loud (non-blocking)
        Uses threading so it doesn't freeze the camera
        """
        def speak():
            try:
                self.engine.say(message)
                self.engine.runAndWait()
            except Exception as e:
                print(f"Audio error: {e}")
        
        # Run in separate thread to avoid blocking
        thread = threading.Thread(target=speak)
        thread.daemon = True
        thread.start()
    
    def process_detections(self, detections, scene_context=None):
        """
        Process all detections and announce the most important ones
        """
        if not detections:
            return

        # Apply scene context priority boost
        if scene_context:
            for det in detections:
                det['scene_boost'] = scene_context.get('priority_boost', 0)

        # Sort by priority (highest first)
        prioritized = sorted(detections, key=self.get_priority, reverse=True)

        # Announce top priority items (don't overwhelm user)
        announced_count = 0
        max_announcements = 2  # Only announce top 2 per frame

        for det in prioritized:
            if announced_count >= max_announcements:
                break

            if self.should_announce(det):
                message = self.generate_message(det)
                print(f"🔊 Speaking: {message}")
                self.announce(message)

                # Update last announcement time
                self.last_announcement_time[det['class']] = time.time()
                announced_count += 1