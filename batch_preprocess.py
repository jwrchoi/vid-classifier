#!/usr/bin/env python3
"""
Batch Preprocessing Script for Running Shoe Video Classifier.

Pre-computes model predictions and features for all videos in a directory.
This makes the annotation tool load instantly without per-video processing delays.

Usage:
    python batch_preprocess.py                          # Use VIDEO_DIR from config
    python batch_preprocess.py /path/to/videos          # Specify custom directory
    python batch_preprocess.py /path/to/videos --limit 20   # Process only first 20 videos

Output:
    - data/predictions_cache.csv: Model predictions for each video
    - data/features_cache.csv: Computed features (pace, density, gesture)
"""

import argparse
import sys
import time
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    VIDEO_DIR, OUTPUT_DIR, DEVICE,
    MODEL_CONFIGS, ACTIVE_MODELS,
    FRAME_SAMPLE_INTERVAL, MAX_FRAMES_PER_VIDEO,
    SUPPORTED_VIDEO_EXTENSIONS, find_models_dir, ensure_output_dir
)
from utils.video_processing import list_videos, sample_frames, get_video_info, extract_video_id
from models.model_loader import ModelLoader
from models.feature_extractors import extract_all_features


def print_header():
    """Print a nice header."""
    print("=" * 60)
    print("  Running Shoe Video Classifier - Batch Preprocessing")
    print("=" * 60)
    print()


def print_progress(current: int, total: int, video_name: str, elapsed: float):
    """Print progress bar and status."""
    pct = (current / total) * 100
    bar_len = 30
    filled = int(bar_len * current / total)
    bar = "█" * filled + "░" * (bar_len - filled)
    
    # Estimate remaining time
    if current > 0:
        avg_time = elapsed / current
        remaining = avg_time * (total - current)
        remaining_str = f"{remaining/60:.1f} min" if remaining > 60 else f"{remaining:.0f} sec"
    else:
        remaining_str = "calculating..."
    
    # Truncate video name if too long
    max_name_len = 30
    if len(video_name) > max_name_len:
        video_name = video_name[:max_name_len-3] + "..."
    
    print(f"\r  [{bar}] {current}/{total} ({pct:.0f}%) | {video_name:<{max_name_len}} | ETA: {remaining_str}    ", end="", flush=True)


def load_models_for_batch():
    """Load all models for batch processing."""
    print("Loading models...")
    
    models_dir = find_models_dir()
    
    if not models_dir.exists():
        print(f"  ⚠️  Models directory not found: {models_dir}")
        print("  Continuing without model predictions (only computed features)")
        return None
    
    loader = ModelLoader(models_dir, MODEL_CONFIGS, DEVICE)
    loader.load_all_models(ACTIVE_MODELS)
    
    if not loader.loaded_models:
        print("  ⚠️  No models loaded")
        return None
    
    print(f"  ✅ Loaded {len(loader.loaded_models)} models on {DEVICE}")
    for name in loader.loaded_models:
        print(f"     - {MODEL_CONFIGS[name]['display_name']}")
    
    return loader


def process_single_video(
    video_path: Path,
    model_loader: Optional[ModelLoader]
) -> Dict:
    """
    Process a single video: extract frames, run models, compute features.
    
    Args:
        video_path: Path to video file
        model_loader: Loaded models (can be None)
        
    Returns:
        Dictionary with all predictions and features
    """
    result = {
        'video_id': extract_video_id(video_path.name),
        'filename': video_path.name,
        'path': str(video_path),
        'processed_at': datetime.now().isoformat(),
        'success': False,
        'error': None
    }
    
    try:
        # Get video info
        video_info = get_video_info(video_path)
        if 'error' in video_info:
            result['error'] = video_info['error']
            return result
        
        result['duration_sec'] = video_info.get('duration_sec', 0)
        result['fps'] = video_info.get('fps', 30)
        
        # Sample frames
        frames, meta = sample_frames(
            video_path,
            interval=FRAME_SAMPLE_INTERVAL,
            max_frames=MAX_FRAMES_PER_VIDEO
        )
        
        if not frames:
            result['error'] = "Could not extract frames"
            return result
        
        result['num_frames_sampled'] = len(frames)
        
        # Run model predictions
        if model_loader:
            for model_name in ACTIVE_MODELS:
                if model_name in model_loader.loaded_models:
                    pred_result = model_loader.predict_video(frames, model_name)
                    
                    # Store prediction and confidence
                    result[f'pred_{model_name}'] = pred_result['prediction']
                    result[f'conf_{model_name}'] = pred_result['confidence']
                    
                    # Store full details as JSON for the cache
                    result[f'details_{model_name}'] = json.dumps(pred_result.get('details', {}))
        
        # Compute features
        fps = video_info.get('fps', 30)
        features = extract_all_features(frames, fps, FRAME_SAMPLE_INTERVAL)
        
        # Editing pace
        if 'editing_pace' in features:
            pace = features['editing_pace']
            result['pace_category'] = pace.get('pace_category', '')
            result['cuts_per_second'] = pace.get('cuts_per_second', 0)
            result['avg_shot_duration'] = pace.get('avg_shot_duration_sec', 0)
            result['num_cuts'] = pace.get('num_cuts', 0)
        
        # Visual density
        if 'visual_density' in features:
            density = features['visual_density']
            result['density_category'] = density.get('density_category', '')
            result['density_score'] = density.get('avg_density', 0)
        
        # Gesture
        if 'gesture' in features:
            gesture = features['gesture']
            result['gesture_category'] = gesture.get('gesture_category', '')
            result['hands_visible_ratio'] = gesture.get('hands_visible_ratio', 0)
        
        # Face presence
        if 'face_presence' in features:
            face = features['face_presence']
            result['face_visible_ratio'] = face.get('face_visible_ratio', 0)
            result['identity_cue'] = face.get('identity_cue', '')
        
        result['success'] = True
        
    except Exception as e:
        result['error'] = str(e)
    
    return result


def batch_process(
    video_dir: Path,
    output_dir: Path,
    limit: Optional[int] = None,
    skip_existing: bool = True
) -> pd.DataFrame:
    """
    Process all videos in a directory.
    
    Args:
        video_dir: Directory containing videos
        output_dir: Directory to save results
        limit: Maximum number of videos to process (None = all)
        skip_existing: Skip videos already in the cache
        
    Returns:
        DataFrame with all results
    """
    print(f"\nScanning for videos in: {video_dir}")
    
    # List videos
    videos = list_videos(video_dir, SUPPORTED_VIDEO_EXTENSIONS)
    
    if not videos:
        print("  ❌ No videos found!")
        return pd.DataFrame()
    
    print(f"  Found {len(videos)} videos")
    
    # Apply limit
    if limit:
        videos = videos[:limit]
        print(f"  Processing first {limit} videos (--limit flag)")
    
    # Check for existing cache
    cache_file = output_dir / "predictions_cache.csv"
    existing_ids = set()
    
    if skip_existing and cache_file.exists():
        try:
            existing_df = pd.read_csv(cache_file)
            existing_ids = set(existing_df['video_id'].dropna().unique())
            print(f"  Found {len(existing_ids)} videos already processed")
        except Exception:
            pass
    
    # Filter out already processed videos
    if skip_existing and existing_ids:
        videos_to_process = [
            v for v in videos 
            if extract_video_id(v['filename']) not in existing_ids
        ]
        skipped = len(videos) - len(videos_to_process)
        if skipped > 0:
            print(f"  Skipping {skipped} already processed videos")
        videos = videos_to_process
    
    if not videos:
        print("  ✅ All videos already processed!")
        return pd.read_csv(cache_file) if cache_file.exists() else pd.DataFrame()
    
    # Load models
    print()
    model_loader = load_models_for_batch()
    
    # Process videos
    print(f"\nProcessing {len(videos)} videos...")
    print()
    
    results = []
    start_time = time.time()
    
    for i, video in enumerate(videos):
        video_path = Path(video['path'])
        
        # Process
        result = process_single_video(video_path, model_loader)
        results.append(result)
        
        # Progress
        elapsed = time.time() - start_time
        print_progress(i + 1, len(videos), video['filename'], elapsed)
        
        # Periodic save (every 10 videos)
        if (i + 1) % 10 == 0:
            save_results(results, output_dir, existing_ids, cache_file)
    
    print()  # New line after progress bar
    
    # Final save
    final_df = save_results(results, output_dir, existing_ids, cache_file)
    
    # Summary
    elapsed = time.time() - start_time
    success_count = sum(1 for r in results if r['success'])
    
    print()
    print("=" * 60)
    print("  Processing Complete!")
    print("=" * 60)
    print(f"  Videos processed: {len(results)}")
    print(f"  Successful: {success_count}")
    print(f"  Failed: {len(results) - success_count}")
    print(f"  Total time: {elapsed/60:.1f} minutes ({elapsed/len(results):.1f} sec/video)")
    print(f"  Results saved to: {cache_file}")
    print()
    
    return final_df


def save_results(
    results: List[Dict],
    output_dir: Path,
    existing_ids: set,
    cache_file: Path
) -> pd.DataFrame:
    """Save results to CSV, merging with existing if present."""
    
    new_df = pd.DataFrame(results)
    
    # Merge with existing cache
    if cache_file.exists() and existing_ids:
        try:
            existing_df = pd.read_csv(cache_file)
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            # Remove duplicates (keep latest)
            combined_df = combined_df.drop_duplicates(subset='video_id', keep='last')
        except Exception:
            combined_df = new_df
    else:
        combined_df = new_df
    
    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(cache_file, index=False)
    
    return combined_df


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Batch preprocess videos for the annotation tool."
    )
    parser.add_argument(
        'video_dir',
        nargs='?',
        default=None,
        help="Directory containing videos (default: VIDEO_DIR from config)"
    )
    parser.add_argument(
        '--limit', '-n',
        type=int,
        default=None,
        help="Maximum number of videos to process"
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help="Output directory (default: data/)"
    )
    parser.add_argument(
        '--force', '-f',
        action='store_true',
        help="Reprocess all videos (don't skip existing)"
    )
    
    args = parser.parse_args()
    
    # Determine video directory
    if args.video_dir:
        video_dir = Path(args.video_dir)
    else:
        video_dir = VIDEO_DIR
    
    # Determine output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = ensure_output_dir()
    
    print_header()
    print(f"Video directory: {video_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Device: {DEVICE}")
    
    if not video_dir.exists():
        print(f"\n❌ Video directory not found: {video_dir}")
        print("Please provide a valid path or update VIDEO_DIR in config.py")
        sys.exit(1)
    
    # Run batch processing
    batch_process(
        video_dir=video_dir,
        output_dir=output_dir,
        limit=args.limit,
        skip_existing=not args.force
    )


if __name__ == "__main__":
    main()
