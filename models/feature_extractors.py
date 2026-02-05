"""
Feature extractors for non-model-based features.

This module provides computational extraction of:
- Editing pace (scene cuts)
- Visual density
- Gesture/hand detection (using MediaPipe)
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from scipy.signal import find_peaks

# Try to import MediaPipe (optional dependency)
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("[WARNING] MediaPipe not installed. Gesture detection will be limited.")


class EditingPaceAnalyzer:
    """
    Analyzes video editing pace by detecting scene cuts.
    
    Uses histogram comparison between consecutive frames to detect
    significant visual changes (cuts).
    """
    
    def __init__(
        self, 
        cut_threshold: float = 0.5,
        min_cut_distance_frames: int = 5
    ):
        """
        Initialize the editing pace analyzer.
        
        Args:
            cut_threshold: Correlation threshold below which a cut is detected
            min_cut_distance_frames: Minimum frames between detected cuts
        """
        self.cut_threshold = cut_threshold
        self.min_cut_distance_frames = min_cut_distance_frames
    
    def detect_cuts(self, frames: List[np.ndarray]) -> List[int]:
        """
        Detect scene cuts in a sequence of frames.
        
        Args:
            frames: List of RGB frames
            
        Returns:
            List of frame indices where cuts occur
        """
        if len(frames) < 2:
            return []
        
        # Compute histogram differences
        diffs = []
        prev_hist = None
        
        for frame in frames:
            # Convert to grayscale
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            else:
                gray = frame
            
            # Compute histogram
            hist = cv2.calcHist([gray], [0], None, [64], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            
            if prev_hist is not None:
                # Correlation-based difference (1 = identical, 0 = different)
                corr = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
                diffs.append(1 - corr)  # Convert so higher = more different
            
            prev_hist = hist
        
        if not diffs:
            return []
        
        diffs = np.array(diffs)
        
        # Find peaks (cuts) using adaptive threshold
        threshold = max(
            np.mean(diffs) + 1.5 * np.std(diffs),
            self.cut_threshold
        )
        
        cuts, _ = find_peaks(
            diffs, 
            height=threshold, 
            distance=self.min_cut_distance_frames
        )
        
        return cuts.tolist()
    
    def analyze(
        self, 
        frames: List[np.ndarray], 
        fps: float = 30.0,
        frame_interval: int = 1
    ) -> Dict:
        """
        Full editing pace analysis.
        
        Args:
            frames: List of sampled frames
            fps: Original video FPS
            frame_interval: Interval at which frames were sampled
            
        Returns:
            Dictionary with pace metrics
        """
        cuts = self.detect_cuts(frames)
        
        # Estimate actual video duration
        actual_fps = fps / frame_interval
        duration_sec = len(frames) / actual_fps if actual_fps > 0 else 0
        
        # Calculate metrics
        num_cuts = len(cuts)
        cuts_per_second = num_cuts / duration_sec if duration_sec > 0 else 0
        
        # Shot durations
        if num_cuts > 0:
            cut_frames = np.concatenate([[0], cuts, [len(frames) - 1]])
            shot_durations = np.diff(cut_frames) / actual_fps
            avg_shot_duration = np.mean(shot_durations)
            min_shot_duration = np.min(shot_durations)
        else:
            avg_shot_duration = duration_sec
            min_shot_duration = duration_sec
            shot_durations = [duration_sec]
        
        # Categorize pace
        if cuts_per_second > 0.5:
            pace_category = 'fast'
        elif cuts_per_second > 0.2:
            pace_category = 'moderate'
        else:
            pace_category = 'slow'
        
        return {
            'num_cuts': num_cuts,
            'cuts_per_second': round(cuts_per_second, 3),
            'avg_shot_duration_sec': round(avg_shot_duration, 2),
            'min_shot_duration_sec': round(min_shot_duration, 2),
            'duration_sec': round(duration_sec, 2),
            'pace_category': pace_category,
            'cut_frames': cuts
        }


class VisualDensityAnalyzer:
    """
    Analyzes visual density (how "busy" the frame is).
    
    Components:
    - Edge density (shape complexity)
    - Color diversity
    - High-frequency content (detail level)
    """
    
    def analyze_frame(self, frame: np.ndarray) -> Dict:
        """
        Analyze visual density of a single frame.
        
        Args:
            frame: RGB image as numpy array
            
        Returns:
            Dictionary with density metrics
        """
        h, w = frame.shape[:2]
        frame_area = h * w
        
        # Convert to grayscale
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        else:
            gray = frame
        
        # 1. Edge density (Canny edges)
        edges = cv2.Canny(gray, 100, 200)
        edge_density = np.sum(edges > 0) / frame_area
        
        # 2. Color diversity (unique colors in quantized space)
        if len(frame.shape) == 3:
            quantized = (frame // 32) * 32  # Quantize to 8 levels per channel
            pixels = quantized.reshape(-1, 3)
            unique_colors = len(np.unique(pixels, axis=0))
            color_diversity = min(unique_colors / 500, 1.0)  # Normalize
        else:
            color_diversity = 0.0
        
        # 3. High-frequency content (Laplacian variance)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian_var = laplacian.var() / 10000  # Normalize
        high_freq_content = min(laplacian_var, 1.0)
        
        # 4. Contrast (standard deviation of intensity)
        contrast = gray.std() / 128  # Normalize to 0-1 range
        
        # Composite score (weighted average)
        composite = (
            0.30 * edge_density +
            0.25 * color_diversity +
            0.25 * high_freq_content +
            0.20 * contrast
        )
        
        return {
            'edge_density': round(edge_density, 4),
            'color_diversity': round(color_diversity, 4),
            'high_freq_content': round(high_freq_content, 4),
            'contrast': round(contrast, 4),
            'composite_score': round(composite, 4)
        }
    
    def analyze_video(self, frames: List[np.ndarray]) -> Dict:
        """
        Analyze visual density across video frames.
        
        Args:
            frames: List of sampled frames
            
        Returns:
            Dictionary with aggregated density metrics
        """
        if not frames:
            return {
                'avg_density': 0.0,
                'density_category': 'unknown',
                'frame_densities': []
            }
        
        # Analyze each frame
        frame_results = [self.analyze_frame(f) for f in frames]
        composite_scores = [r['composite_score'] for r in frame_results]
        
        # Aggregate
        avg_density = np.mean(composite_scores)
        std_density = np.std(composite_scores)
        
        # Categorize
        if avg_density < 0.15:
            category = 'minimal'
        elif avg_density < 0.30:
            category = 'moderate'
        else:
            category = 'high'
        
        return {
            'avg_density': round(avg_density, 4),
            'std_density': round(std_density, 4),
            'min_density': round(min(composite_scores), 4),
            'max_density': round(max(composite_scores), 4),
            'density_category': category,
            'num_frames_analyzed': len(frames)
        }


class GestureAnalyzer:
    """
    Analyzes gestures and hand visibility using MediaPipe.
    
    Note: Requires MediaPipe to be installed.
    """
    
    def __init__(self):
        """Initialize the gesture analyzer."""
        self.available = MEDIAPIPE_AVAILABLE
        
        if self.available:
            self.mp_hands = mp.solutions.hands
            self.mp_pose = mp.solutions.pose
    
    def analyze_frame(self, frame: np.ndarray) -> Dict:
        """
        Analyze gestures in a single frame.
        
        Args:
            frame: RGB image as numpy array
            
        Returns:
            Dictionary with gesture information
        """
        if not self.available:
            return {
                'hands_visible': None,
                'num_hands': None,
                'gesture': 'unknown',
                'error': 'MediaPipe not available'
            }
        
        result = {
            'hands_visible': False,
            'num_hands': 0,
            'gesture': 'none',
            'hand_positions': []
        }
        
        # Detect hands
        with self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.5
        ) as hands:
            
            # MediaPipe expects RGB
            if frame.shape[2] == 3:
                rgb_frame = frame
            else:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            hands_result = hands.process(rgb_frame)
            
            if hands_result.multi_hand_landmarks:
                result['hands_visible'] = True
                result['num_hands'] = len(hands_result.multi_hand_landmarks)
                
                # Analyze hand positions
                h, w = frame.shape[:2]
                for hand_landmarks in hands_result.multi_hand_landmarks:
                    # Get wrist position (landmark 0)
                    wrist = hand_landmarks.landmark[0]
                    
                    # Determine position in frame
                    if wrist.y < 0.33:
                        position = 'upper'
                    elif wrist.y < 0.66:
                        position = 'middle'
                    else:
                        position = 'lower'
                    
                    result['hand_positions'].append(position)
                    
                    # Simple pointing detection
                    # (index finger extended, others curled)
                    index_tip = hand_landmarks.landmark[8]
                    index_mcp = hand_landmarks.landmark[5]
                    
                    if index_tip.y < index_mcp.y - 0.05:
                        result['gesture'] = 'pointing'
        
        return result
    
    def analyze_video(self, frames: List[np.ndarray], sample_rate: int = 5) -> Dict:
        """
        Analyze gestures across video frames.
        
        Args:
            frames: List of sampled frames
            sample_rate: Analyze every Nth frame (for speed)
            
        Returns:
            Dictionary with aggregated gesture information
        """
        if not self.available:
            return {
                'hands_visible_ratio': None,
                'gesture_category': 'unknown',
                'error': 'MediaPipe not available'
            }
        
        # Sample frames for efficiency
        sampled_frames = frames[::sample_rate]
        
        hands_visible_count = 0
        pointing_count = 0
        
        for frame in sampled_frames:
            result = self.analyze_frame(frame)
            if result['hands_visible']:
                hands_visible_count += 1
            if result['gesture'] == 'pointing':
                pointing_count += 1
        
        hands_ratio = hands_visible_count / len(sampled_frames) if sampled_frames else 0
        
        # Categorize
        if pointing_count > len(sampled_frames) * 0.1:
            gesture_category = 'pointing'
        elif hands_ratio > 0.5:
            gesture_category = 'hands_prominent'
        elif hands_ratio > 0.1:
            gesture_category = 'hands_visible'
        else:
            gesture_category = 'no_hands'
        
        return {
            'hands_visible_ratio': round(hands_ratio, 3),
            'pointing_ratio': round(pointing_count / len(sampled_frames) if sampled_frames else 0, 3),
            'gesture_category': gesture_category,
            'frames_analyzed': len(sampled_frames)
        }


class FaceAnalyzer:
    """
    Analyzes face presence and prominence using MediaPipe.
    """
    
    def __init__(self):
        """Initialize the face analyzer."""
        self.available = MEDIAPIPE_AVAILABLE
        
        if self.available:
            self.mp_face = mp.solutions.face_detection
    
    def analyze_frame(self, frame: np.ndarray) -> Dict:
        """
        Analyze faces in a single frame.
        
        Args:
            frame: RGB image as numpy array
            
        Returns:
            Dictionary with face information
        """
        if not self.available:
            return {
                'face_visible': None,
                'num_faces': None,
                'error': 'MediaPipe not available'
            }
        
        with self.mp_face.FaceDetection(
            min_detection_confidence=0.5,
            model_selection=1  # Full-range model
        ) as face_detection:
            
            results = face_detection.process(frame)
            
            if results.detections:
                # Get the largest face
                faces = []
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    area = bbox.width * bbox.height
                    faces.append({
                        'area': area,
                        'confidence': detection.score[0]
                    })
                
                largest_face = max(faces, key=lambda x: x['area'])
                
                return {
                    'face_visible': True,
                    'num_faces': len(faces),
                    'largest_face_area': round(largest_face['area'], 4),
                    'face_confidence': round(largest_face['confidence'], 3)
                }
            else:
                return {
                    'face_visible': False,
                    'num_faces': 0,
                    'largest_face_area': 0,
                    'face_confidence': 0
                }
    
    def analyze_video(self, frames: List[np.ndarray], sample_rate: int = 3) -> Dict:
        """
        Analyze face presence across video frames.
        
        Args:
            frames: List of sampled frames
            sample_rate: Analyze every Nth frame
            
        Returns:
            Dictionary with aggregated face information
        """
        if not self.available:
            return {
                'face_visible_ratio': None,
                'identity_cue': 'unknown',
                'error': 'MediaPipe not available'
            }
        
        sampled_frames = frames[::sample_rate]
        
        face_visible_count = 0
        face_areas = []
        
        for frame in sampled_frames:
            result = self.analyze_frame(frame)
            if result['face_visible']:
                face_visible_count += 1
                face_areas.append(result['largest_face_area'])
        
        face_ratio = face_visible_count / len(sampled_frames) if sampled_frames else 0
        avg_face_area = np.mean(face_areas) if face_areas else 0
        
        # Categorize identity cue strength
        if face_ratio > 0.7 and avg_face_area > 0.1:
            identity_cue = 'strong'  # Creator is prominently visible
        elif face_ratio > 0.3:
            identity_cue = 'moderate'  # Face sometimes visible
        elif face_ratio > 0:
            identity_cue = 'weak'  # Occasional face
        else:
            identity_cue = 'none'  # No face (product-only or POV)
        
        return {
            'face_visible_ratio': round(face_ratio, 3),
            'avg_face_prominence': round(avg_face_area, 4),
            'identity_cue': identity_cue,
            'frames_analyzed': len(sampled_frames)
        }


def extract_all_features(
    frames: List[np.ndarray],
    fps: float = 30.0,
    frame_interval: int = 15
) -> Dict:
    """
    Extract all non-model features from video frames.
    
    Args:
        frames: List of sampled frames
        fps: Original video FPS
        frame_interval: Interval at which frames were sampled
        
    Returns:
        Dictionary with all extracted features
    """
    pace_analyzer = EditingPaceAnalyzer()
    density_analyzer = VisualDensityAnalyzer()
    gesture_analyzer = GestureAnalyzer()
    face_analyzer = FaceAnalyzer()
    
    return {
        'editing_pace': pace_analyzer.analyze(frames, fps, frame_interval),
        'visual_density': density_analyzer.analyze_video(frames),
        'gesture': gesture_analyzer.analyze_video(frames),
        'face_presence': face_analyzer.analyze_video(frames)
    }
