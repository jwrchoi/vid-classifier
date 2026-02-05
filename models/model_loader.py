"""
Model loading utilities for the Running Shoe Video Classifier.

This module handles loading the trained ResNet-50 models and running inference.
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from pathlib import Path
import numpy as np
from typing import Dict, List, Optional, Tuple
import warnings

# Suppress some torch warnings
warnings.filterwarnings('ignore', category=UserWarning)


class ModelLoader:
    """
    Loads and manages multiple trained classification models.
    
    This class handles:
    - Loading model weights from .pth files
    - Running inference on frames
    - Aggregating frame-level predictions to video-level
    """
    
    def __init__(self, models_dir: Path, model_configs: Dict, device: torch.device):
        """
        Initialize the model loader.
        
        Args:
            models_dir: Directory containing .pth model files
            model_configs: Dictionary of model configurations (from config.py)
            device: PyTorch device (cpu, cuda, or mps)
        """
        self.models_dir = Path(models_dir)
        self.model_configs = model_configs
        self.device = device
        self.loaded_models: Dict[str, nn.Module] = {}
        
        # Image preprocessing transform (must match training)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def load_model(self, model_name: str) -> Optional[nn.Module]:
        """
        Load a single model by name.
        
        Args:
            model_name: Key from model_configs (e.g., 'pov_binary')
            
        Returns:
            Loaded model or None if loading fails
        """
        if model_name in self.loaded_models:
            return self.loaded_models[model_name]
        
        if model_name not in self.model_configs:
            print(f"[ERROR] Unknown model: {model_name}")
            return None
        
        config = self.model_configs[model_name]
        model_path = self.models_dir / config['model_file']
        
        if not model_path.exists():
            print(f"[WARNING] Model file not found: {model_path}")
            return None
        
        try:
            # Create ResNet-50 architecture
            model = models.resnet50(weights=None)  # Don't load ImageNet weights
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, config['num_classes'])
            
            # Load trained weights
            state_dict = torch.load(model_path, map_location=self.device)
            model.load_state_dict(state_dict)
            
            # Move to device and set to evaluation mode
            model = model.to(self.device)
            model.eval()
            
            self.loaded_models[model_name] = model
            print(f"[OK] Loaded model: {model_name} ({config['display_name']})")
            return model
            
        except Exception as e:
            print(f"[ERROR] Failed to load {model_name}: {e}")
            return None
    
    def load_all_models(self, model_names: List[str]) -> Dict[str, nn.Module]:
        """
        Load multiple models.
        
        Args:
            model_names: List of model names to load
            
        Returns:
            Dictionary of successfully loaded models
        """
        for name in model_names:
            self.load_model(name)
        return self.loaded_models
    
    def predict_frame(self, frame: np.ndarray, model_name: str) -> Tuple[str, float, int]:
        """
        Run inference on a single frame.
        
        Args:
            frame: RGB image as numpy array (H, W, 3)
            model_name: Which model to use
            
        Returns:
            Tuple of (predicted_label, confidence, class_index)
        """
        if model_name not in self.loaded_models:
            self.load_model(model_name)
        
        model = self.loaded_models.get(model_name)
        if model is None:
            return ("unknown", 0.0, -1)
        
        config = self.model_configs[model_name]
        
        # Preprocess frame
        input_tensor = self.transform(frame).unsqueeze(0).to(self.device)
        
        # Run inference
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1)
            confidence, pred_idx = torch.max(probs, dim=1)
            
            pred_idx = pred_idx.item()
            confidence = confidence.item()
            pred_label = config['classes'][pred_idx]
        
        return (pred_label, confidence, pred_idx)
    
    def predict_frames_batch(
        self, 
        frames: List[np.ndarray], 
        model_name: str,
        batch_size: int = 32
    ) -> List[Tuple[str, float, int]]:
        """
        Run inference on multiple frames efficiently.
        
        Args:
            frames: List of RGB images as numpy arrays
            model_name: Which model to use
            batch_size: Batch size for inference
            
        Returns:
            List of (predicted_label, confidence, class_index) tuples
        """
        if model_name not in self.loaded_models:
            self.load_model(model_name)
        
        model = self.loaded_models.get(model_name)
        if model is None:
            return [("unknown", 0.0, -1)] * len(frames)
        
        config = self.model_configs[model_name]
        results = []
        
        # Process in batches
        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i:i + batch_size]
            
            # Preprocess batch
            batch_tensors = torch.stack([
                self.transform(frame) for frame in batch_frames
            ]).to(self.device)
            
            # Run inference
            with torch.no_grad():
                outputs = model(batch_tensors)
                probs = torch.softmax(outputs, dim=1)
                confidences, pred_indices = torch.max(probs, dim=1)
                
                for conf, idx in zip(confidences, pred_indices):
                    pred_label = config['classes'][idx.item()]
                    results.append((pred_label, conf.item(), idx.item()))
        
        return results
    
    def aggregate_predictions(
        self, 
        frame_predictions: List[Tuple[str, float, int]],
        method: str = 'majority'
    ) -> Tuple[str, float, Dict]:
        """
        Aggregate frame-level predictions to video-level.
        
        Args:
            frame_predictions: List of (label, confidence, index) tuples
            method: Aggregation method ('majority' or 'mean_confidence')
            
        Returns:
            Tuple of (aggregated_label, aggregated_confidence, details_dict)
        """
        if not frame_predictions:
            return ("unknown", 0.0, {})
        
        labels = [p[0] for p in frame_predictions]
        confidences = [p[1] for p in frame_predictions]
        
        if method == 'majority':
            # Count votes for each label
            from collections import Counter
            label_counts = Counter(labels)
            winner_label, winner_count = label_counts.most_common(1)[0]
            
            # Average confidence for winning label
            winner_confidences = [
                conf for label, conf, _ in frame_predictions 
                if label == winner_label
            ]
            avg_confidence = np.mean(winner_confidences)
            
            details = {
                'label_counts': dict(label_counts),
                'total_frames': len(frame_predictions),
                'winner_ratio': winner_count / len(frame_predictions),
                'all_confidences': confidences
            }
            
        elif method == 'mean_confidence':
            # Weight by confidence
            from collections import defaultdict
            label_conf_sums = defaultdict(float)
            label_counts = defaultdict(int)
            
            for label, conf, _ in frame_predictions:
                label_conf_sums[label] += conf
                label_counts[label] += 1
            
            # Find label with highest total confidence
            winner_label = max(label_conf_sums, key=label_conf_sums.get)
            avg_confidence = label_conf_sums[winner_label] / label_counts[winner_label]
            
            details = {
                'label_conf_sums': dict(label_conf_sums),
                'total_frames': len(frame_predictions)
            }
        else:
            raise ValueError(f"Unknown aggregation method: {method}")
        
        return (winner_label, avg_confidence, details)
    
    def predict_video(
        self, 
        frames: List[np.ndarray], 
        model_name: str,
        aggregation: str = 'majority'
    ) -> Dict:
        """
        Full pipeline: predict on frames and aggregate.
        
        Args:
            frames: List of sampled video frames
            model_name: Which model to use
            aggregation: Aggregation method
            
        Returns:
            Dictionary with prediction results
        """
        config = self.model_configs.get(model_name, {})
        
        # Get frame-level predictions
        frame_preds = self.predict_frames_batch(frames, model_name)
        
        # Aggregate to video level
        agg_label, agg_conf, details = self.aggregate_predictions(
            frame_preds, method=aggregation
        )
        
        return {
            'model_name': model_name,
            'display_name': config.get('display_name', model_name),
            'prediction': agg_label,
            'confidence': round(agg_conf, 4),
            'num_frames': len(frames),
            'frame_predictions': frame_preds,
            'details': details
        }
    
    def predict_all_models(
        self, 
        frames: List[np.ndarray],
        model_names: Optional[List[str]] = None
    ) -> Dict[str, Dict]:
        """
        Run all loaded models on the same frames.
        
        Args:
            frames: List of sampled video frames
            model_names: Specific models to run (None = all loaded)
            
        Returns:
            Dictionary mapping model_name -> prediction results
        """
        if model_names is None:
            model_names = list(self.loaded_models.keys())
        
        results = {}
        for model_name in model_names:
            if model_name in self.loaded_models or model_name in self.model_configs:
                results[model_name] = self.predict_video(frames, model_name)
        
        return results


def get_confidence_level(confidence: float) -> str:
    """
    Convert confidence score to a categorical level.
    
    Args:
        confidence: Confidence score (0-1)
        
    Returns:
        'high', 'medium', or 'low'
    """
    if confidence >= 0.85:
        return 'high'
    elif confidence >= 0.65:
        return 'medium'
    else:
        return 'low'


def format_prediction_display(prediction_result: Dict) -> str:
    """
    Format a prediction result for display.
    
    Args:
        prediction_result: Output from predict_video()
        
    Returns:
        Formatted string for display
    """
    label = prediction_result['prediction']
    conf = prediction_result['confidence']
    level = get_confidence_level(conf)
    
    emoji = {'high': '🟢', 'medium': '🟡', 'low': '🔴'}[level]
    
    return f"{emoji} {label} ({conf*100:.1f}%)"
