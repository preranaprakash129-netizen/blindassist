import torch
import torchvision.models as models
import torchvision.transforms as transforms
import cv2
import numpy as np
from collections import Counter

class SceneClassifier:
    def __init__(self, use_gpu=True):
        """
        Scene classification using pre-trained MobileNetV2
        Classifies environment type for navigation context
        """
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        
        print("Loading scene classification model...")
        # Use lightweight MobileNetV2 (only ~14MB)
        self.model = models.mobilenet_v2(pretrained=True)
        self.model.to(self.device)
        self.model.eval()
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225]),
        ])
        
        # Scene category mapping (based on ImageNet classes)
        self.scene_mapping = {
            'indoor': [526, 610, 762, 831, 857, 871, 765, 920],  # classroom, desk, monitor, space_heater, table, television, microwave, wall_clock
            'outdoor': [400, 895, 980, 717, 970, 487],  # ambulance, traffic_light, alp, park, street, castle
            'staircase': [850],  # staircase class
            'corridor': [762, 920, 526],  # monitor, wall_clock, classroom (corridor-like)
            'office': [610, 831, 762, 526],  # desk, space_heater, monitor, classroom
            'crowded': [444, 671, 837],  # bicycle, microphone, streetcar (proxy for crowds)
        }
        
        # Scene history for smoothing
        self.scene_history = []
        self.history_size = 5
        
        print("✓ Scene classifier ready!")
    
    def classify_scene(self, frame):
        """
        Classify the scene type from a video frame
        
        Returns:
            scene_type: str (indoor/outdoor/staircase/corridor/office/crowded)
            confidence: float
        """
        # Preprocess image
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_tensor = self.transform(img_rgb)
        input_batch = input_tensor.unsqueeze(0).to(self.device)
        
        # Get predictions
        with torch.no_grad():
            output = self.model(input_batch)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
        
        # Get top 10 predictions
        top10_prob, top10_catid = torch.topk(probabilities, 10)
        top10_catid = top10_catid.cpu().numpy()
        top10_prob = top10_prob.cpu().numpy()
        
        # Map to scene categories
        scene_scores = {
            'indoor': 0,
            'outdoor': 0,
            'staircase': 0,
            'corridor': 0,
            'office': 0,
            'crowded': 0
        }
        
        for cat_id, prob in zip(top10_catid, top10_prob):
            for scene, categories in self.scene_mapping.items():
                if cat_id in categories:
                    scene_scores[scene] += prob
        
        # Get best scene
        best_scene = max(scene_scores, key=scene_scores.get)
        confidence = scene_scores[best_scene]
        
        # Add to history for smoothing
        self.scene_history.append(best_scene)
        if len(self.scene_history) > self.history_size:
            self.scene_history.pop(0)
        
        # Get most common scene from recent history
        if len(self.scene_history) >= 3:
            scene_counts = Counter(self.scene_history)
            smoothed_scene = scene_counts.most_common(1)[0][0]
        else:
            smoothed_scene = best_scene
        
        return smoothed_scene, float(confidence)
    
    def get_scene_context(self, scene_type):
        """
        Get navigation context based on scene type
        """
        contexts = {
            'staircase': {
                'warning': 'Staircase environment detected',
                'priority_boost': 50,
                'caution_level': 'high'
            },
            'corridor': {
                'warning': 'Narrow corridor ahead',
                'priority_boost': 20,
                'caution_level': 'medium'
            },
            'crowded': {
                'warning': 'Crowded area detected',
                'priority_boost': 30,
                'caution_level': 'high'
            },
            'outdoor': {
                'warning': 'Outdoor environment',
                'priority_boost': 10,
                'caution_level': 'low'
            },
            'indoor': {
                'warning': 'Indoor environment',
                'priority_boost': 0,
                'caution_level': 'low'
            },
            'office': {
                'warning': 'Office environment',
                'priority_boost': 0,
                'caution_level': 'low'
            }
        }
        
        return contexts.get(scene_type, contexts['indoor'])