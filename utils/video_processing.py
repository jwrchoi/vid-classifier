"""
Video processing utilities for the Running Shoe Video Classifier.

Handles:
- Loading videos from files
- Sampling frames
- Video metadata extraction
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Generator
import os


def get_video_info(video_path: Path) -> Dict:
    """
    Extract metadata from a video file.
    
    Args:
        video_path: Path to video file
        
    Returns:
        Dictionary with video metadata
    """
    video_path = Path(video_path)
    
    if not video_path.exists():
        return {'error': f'File not found: {video_path}'}
    
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        return {'error': f'Could not open video: {video_path}'}
    
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0
        
        return {
            'path': str(video_path),
            'filename': video_path.name,
            'fps': round(fps, 2),
            'frame_count': frame_count,
            'width': width,
            'height': height,
            'duration_sec': round(duration, 2),
            'size_mb': round(video_path.stat().st_size / (1024 * 1024), 2)
        }
    finally:
        cap.release()


def sample_frames(
    video_path: Path,
    interval: int = 15,
    max_frames: int = 100,
    start_frame: int = 0,
    end_frame: Optional[int] = None
) -> Tuple[List[np.ndarray], Dict]:
    """
    Sample frames from a video at regular intervals.
    
    Args:
        video_path: Path to video file
        interval: Sample every Nth frame
        max_frames: Maximum number of frames to sample
        start_frame: Frame to start sampling from
        end_frame: Frame to stop sampling at (None = end of video)
        
    Returns:
        Tuple of (list of RGB frames, metadata dict)
    """
    video_path = Path(video_path)
    frames = []
    
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        return [], {'error': f'Could not open video: {video_path}'}
    
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if end_frame is None:
            end_frame = total_frames
        
        # Set starting position
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        frame_idx = start_frame
        sampled_indices = []
        
        while frame_idx < end_frame and len(frames) < max_frames:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Sample at interval
            if (frame_idx - start_frame) % interval == 0:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
                sampled_indices.append(frame_idx)
            
            frame_idx += 1
        
        metadata = {
            'video_path': str(video_path),
            'fps': fps,
            'total_frames': total_frames,
            'sampled_count': len(frames),
            'interval': interval,
            'sampled_indices': sampled_indices
        }
        
        return frames, metadata
        
    finally:
        cap.release()


def sample_frames_generator(
    video_path: Path,
    interval: int = 15,
    max_frames: int = 100
) -> Generator[Tuple[int, np.ndarray], None, None]:
    """
    Generator version of frame sampling (memory efficient).
    
    Yields:
        Tuple of (frame_index, RGB frame)
    """
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        return
    
    try:
        frame_idx = 0
        yielded_count = 0
        
        while yielded_count < max_frames:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            if frame_idx % interval == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                yield (frame_idx, frame_rgb)
                yielded_count += 1
            
            frame_idx += 1
            
    finally:
        cap.release()


def get_representative_frames(
    video_path: Path,
    num_frames: int = 5
) -> List[np.ndarray]:
    """
    Get evenly-spaced representative frames from a video.
    
    Useful for displaying thumbnails or quick preview.
    
    Args:
        video_path: Path to video file
        num_frames: Number of frames to extract
        
    Returns:
        List of RGB frames
    """
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        return []
    
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            return []
        
        # Calculate frame indices to sample
        if num_frames >= total_frames:
            indices = list(range(total_frames))
        else:
            step = total_frames / num_frames
            indices = [int(i * step) for i in range(num_frames)]
        
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
        
        return frames
        
    finally:
        cap.release()


def list_videos(
    directory: Path,
    extensions: set = {'.mp4', '.mov', '.avi', '.mkv', '.webm'}
) -> List[Dict]:
    """
    List all video files in a directory.
    
    Args:
        directory: Directory to scan
        extensions: Set of valid video extensions
        
    Returns:
        List of dictionaries with video info
    """
    directory = Path(directory)
    
    if not directory.exists():
        return []
    
    videos = []
    
    for path in sorted(directory.iterdir()):
        # Skip hidden files
        if path.name.startswith('.'):
            continue
        
        # Check extension
        if path.suffix.lower() not in extensions:
            continue
        
        # Skip empty files
        if path.stat().st_size == 0:
            continue
        
        videos.append({
            'path': str(path),
            'filename': path.name,
            'size_mb': round(path.stat().st_size / (1024 * 1024), 2)
        })
    
    return videos


def create_video_thumbnail(
    video_path: Path,
    output_path: Optional[Path] = None,
    frame_position: float = 0.1,
    size: Tuple[int, int] = (320, 180)
) -> Optional[np.ndarray]:
    """
    Create a thumbnail image from a video.
    
    Args:
        video_path: Path to video file
        output_path: If provided, save thumbnail to this path
        frame_position: Position in video to capture (0.0 to 1.0)
        size: Output size (width, height)
        
    Returns:
        Thumbnail as numpy array, or None if failed
    """
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        return None
    
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        target_frame = int(total_frames * frame_position)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ret, frame = cap.read()
        
        if not ret:
            return None
        
        # Resize
        thumbnail = cv2.resize(frame, size)
        
        # Convert to RGB
        thumbnail_rgb = cv2.cvtColor(thumbnail, cv2.COLOR_BGR2RGB)
        
        # Save if output path provided
        if output_path:
            cv2.imwrite(str(output_path), thumbnail)
        
        return thumbnail_rgb
        
    finally:
        cap.release()


def extract_video_id(filename: str) -> str:
    """
    Extract video ID from filename.
    
    Expected formats:
    - video-username-timestamp-videoid.mp4
    - videoid.mp4
    
    Args:
        filename: Video filename
        
    Returns:
        Extracted video ID
    """
    # Remove extension
    name = Path(filename).stem
    
    # Try to extract from pattern: video-user-timestamp-id
    parts = name.split('-')
    if len(parts) >= 4:
        # Last part is likely the video ID
        potential_id = parts[-1]
        if potential_id.isdigit() and len(potential_id) > 10:
            return potential_id
    
    # Otherwise return the whole name
    return name


class VideoProcessor:
    """
    High-level video processor that combines multiple operations.
    """
    
    def __init__(
        self,
        frame_interval: int = 15,
        max_frames: int = 100
    ):
        """
        Initialize the video processor.
        
        Args:
            frame_interval: Interval for frame sampling
            max_frames: Maximum frames to sample
        """
        self.frame_interval = frame_interval
        self.max_frames = max_frames
    
    def process_video(self, video_path: Path) -> Dict:
        """
        Process a single video: extract info and sample frames.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary with video info and sampled frames
        """
        video_path = Path(video_path)
        
        # Get video info
        info = get_video_info(video_path)
        
        if 'error' in info:
            return info
        
        # Sample frames
        frames, sample_meta = sample_frames(
            video_path,
            interval=self.frame_interval,
            max_frames=self.max_frames
        )
        
        # Get representative frames for display
        display_frames = get_representative_frames(video_path, num_frames=5)
        
        return {
            'info': info,
            'frames': frames,
            'display_frames': display_frames,
            'sample_metadata': sample_meta,
            'video_id': extract_video_id(video_path.name)
        }
    
    def get_video_for_annotation(self, video_path: Path) -> Dict:
        """
        Prepare a video for annotation (lighter processing).
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary with video info and frames for inference
        """
        video_path = Path(video_path)
        
        # Get basic info
        info = get_video_info(video_path)
        
        if 'error' in info:
            return {'error': info['error'], 'video_path': str(video_path)}
        
        # Sample frames for inference
        frames, sample_meta = sample_frames(
            video_path,
            interval=self.frame_interval,
            max_frames=self.max_frames
        )
        
        return {
            'video_path': str(video_path),
            'video_id': extract_video_id(video_path.name),
            'filename': video_path.name,
            'info': info,
            'frames': frames,
            'num_frames': len(frames)
        }
