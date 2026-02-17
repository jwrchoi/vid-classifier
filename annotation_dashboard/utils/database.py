"""
Database utilities for storing and retrieving annotations.

Uses CSV files for simplicity (easy to inspect and export).
Can be upgraded to SQLite or PostgreSQL if needed.
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import json
import os
import shutil


class AnnotationDatabase:
    """
    Manages annotation storage using CSV files.
    
    Features:
    - Save and load annotations
    - Track annotation progress
    - Export for model training
    - Backup functionality
    """
    
    def __init__(self, data_dir: Path):
        """
        Initialize the annotation database.
        
        Args:
            data_dir: Directory to store annotation files
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Main annotations file
        self.annotations_file = self.data_dir / "annotations.csv"
        
        # Predictions cache (model outputs)
        self.predictions_file = self.data_dir / "predictions_cache.csv"
        
        # Session log
        self.session_log_file = self.data_dir / "session_log.csv"
        
        # Backup directory
        self.backup_dir = self.data_dir / "backups"
        self.backup_dir.mkdir(exist_ok=True)
        
        # Initialize files if they don't exist
        self._init_files()
    
    def _init_files(self):
        """Initialize CSV files with headers if they don't exist."""
        
        # Annotations schema
        if not self.annotations_file.exists():
            df = pd.DataFrame(columns=[
                'video_id',
                'frame_index',
                'frame_total',
                'filename',
                'annotator',
                'timestamp',
                # Annotation fields
                'no_human_visible',  # New field
                'perspective',
                'distance',
                'gaze',
                'pace',
                'density',
                'gesture',
                # Model predictions (for comparison)
                'model_perspective',
                'model_perspective_conf',
                'model_distance',
                'model_distance_conf',
                'model_gaze',
                'model_gaze_conf',
                # Computed features
                'computed_pace',
                'computed_density',
                'computed_gesture',
                # Metadata
                'notes',
                'is_difficult',
                'annotation_time_sec'
            ])
            df.to_csv(self.annotations_file, index=False)
        
        # Predictions cache schema
        if not self.predictions_file.exists():
            df = pd.DataFrame(columns=[
                'video_id',
                'filename',
                'timestamp',
                'predictions_json'  # Store full predictions as JSON
            ])
            df.to_csv(self.predictions_file, index=False)
        
        # Session log schema
        if not self.session_log_file.exists():
            df = pd.DataFrame(columns=[
                'session_id',
                'annotator',
                'start_time',
                'end_time',
                'videos_annotated',
                'notes'
            ])
            df.to_csv(self.session_log_file, index=False)
    
    def save_annotation(
        self,
        video_id: str,
        filename: str,
        annotations: Dict,
        model_predictions: Dict,
        computed_features: Dict,
        annotator: str = "default",
        notes: str = "",
        is_difficult: bool = False,
        annotation_time_sec: float = 0,
        frame_index: Optional[int] = None,
        frame_total: Optional[int] = None
    ) -> bool:
        """
        Save a single annotation.

        Args:
            video_id: Unique video identifier
            filename: Video filename
            annotations: Human annotations dict
            model_predictions: Model predictions dict
            computed_features: Computed features dict
            annotator: Annotator name/ID
            notes: Optional notes
            is_difficult: Flag for difficult cases
            annotation_time_sec: Time spent annotating

        Returns:
            True if saved successfully
        """
        try:
            # Load existing annotations
            df = pd.read_csv(self.annotations_file)
            # Ensure video_id is always string to avoid type mismatches
            df['video_id'] = df['video_id'].astype(str)
            video_id = str(video_id)

            # Create new row
            new_row = {
                'video_id': video_id,
                'frame_index': frame_index,
                'frame_total': frame_total,
                'filename': filename,
                'annotator': annotator,
                'timestamp': datetime.now().isoformat(),
                # Annotations
                'no_human_visible': annotations.get('no_human_visible', False),
                'perspective': annotations.get('perspective', ''),
                'distance': annotations.get('distance', ''),
                'gaze': annotations.get('gaze', ''),
                'pace': annotations.get('pace', ''),
                'density': annotations.get('density', ''),
                'gesture': annotations.get('gesture', ''),
                # Model predictions
                'model_perspective': model_predictions.get('perspective', {}).get('prediction', ''),
                'model_perspective_conf': model_predictions.get('perspective', {}).get('confidence', 0),
                'model_distance': model_predictions.get('distance', {}).get('prediction', ''),
                'model_distance_conf': model_predictions.get('distance', {}).get('confidence', 0),
                'model_gaze': model_predictions.get('gaze', {}).get('prediction', ''),
                'model_gaze_conf': model_predictions.get('gaze', {}).get('confidence', 0),
                # Computed features
                'computed_pace': computed_features.get('pace', ''),
                'computed_density': computed_features.get('density', ''),
                'computed_gesture': computed_features.get('gesture', ''),
                # Metadata
                'notes': notes,
                'is_difficult': is_difficult,
                'annotation_time_sec': annotation_time_sec
            }

            # Drop existing rows for this composite key
            if frame_index is not None:
                # Ensure frame_index column is comparable
                if 'frame_index' in df.columns:
                    mask = (
                        (df['video_id'] == video_id)
                        & (df['frame_index'].astype(float) == float(frame_index))
                        & (df['annotator'] == annotator)
                    )
                else:
                    mask = pd.Series(False, index=df.index)
                df = df[~mask]
            else:
                df = df[~((df['video_id'] == video_id) & (df['annotator'] == annotator))]
            # Append new row
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

            # Save
            df.to_csv(self.annotations_file, index=False)
            return True

        except Exception as e:
            print(f"Error saving annotation: {e}")
            return False
    
    def get_annotation(self, video_id: str, annotator: str = None, frame_index: Optional[int] = None) -> Optional[Dict]:
        """
        Get annotation for a specific video (and optionally frame), filtered by annotator.

        Args:
            video_id: Video identifier
            annotator: If provided, only return this annotator's annotation
            frame_index: If provided, filter to this specific frame

        Returns:
            Annotation dict or None if not found
        """
        try:
            df = pd.read_csv(self.annotations_file)
            df['video_id'] = df['video_id'].astype(str)
            video_id = str(video_id)
            mask = df['video_id'] == video_id
            if annotator:
                mask = mask & (df['annotator'] == annotator)
            if frame_index is not None and 'frame_index' in df.columns:
                mask = mask & (df['frame_index'].astype(float) == float(frame_index))
            row = df[mask]

            if len(row) == 0:
                return None

            return row.iloc[-1].to_dict()
            
        except Exception as e:
            print(f"Error getting annotation: {e}")
            return None
    
    def get_all_annotations(self) -> pd.DataFrame:
        """Get all annotations as DataFrame."""
        try:
            df = pd.read_csv(self.annotations_file)
            df['video_id'] = df['video_id'].astype(str)
            return df
        except Exception as e:
            print(f"Error loading annotations: {e}")
            return pd.DataFrame()
    
    def get_annotated_video_ids(self, annotator: str = None) -> set:
        """Get set of video IDs that have been annotated, optionally by a specific annotator."""
        try:
            df = pd.read_csv(self.annotations_file)
            df['video_id'] = df['video_id'].astype(str)
            if annotator:
                df = df[df['annotator'] == annotator]
            return set(df['video_id'].dropna().unique())
        except Exception:
            return set()
    
    def get_annotated_frame_indices(self, video_id: str, annotator: str) -> set:
        """Return set of frame indices already annotated for a video by this annotator."""
        try:
            df = pd.read_csv(self.annotations_file)
            df['video_id'] = df['video_id'].astype(str)
            video_id = str(video_id)
            mask = (df['video_id'] == video_id) & (df['annotator'] == annotator)
            if 'frame_index' in df.columns:
                return set(df.loc[mask, 'frame_index'].dropna().astype(int).tolist())
            return set()
        except Exception:
            return set()

    def is_video_fully_annotated(self, video_id: str, annotator: str, frame_total: int) -> bool:
        """Check whether all frames of a video have been annotated by this annotator."""
        annotated = self.get_annotated_frame_indices(video_id, annotator)
        return len(annotated) >= frame_total

    def get_frame_annotation_summary(self, annotator: str, frame_total: int) -> Dict[str, set]:
        """
        Single CSV read returning {video_id: set_of_annotated_frame_indices} for this annotator.

        Used by the sidebar to compute both video-level and frame-level progress efficiently.
        """
        try:
            df = pd.read_csv(self.annotations_file)
            df['video_id'] = df['video_id'].astype(str)
            df = df[df['annotator'] == annotator]
            summary: Dict[str, set] = {}
            if 'frame_index' not in df.columns:
                # Legacy rows without frame_index — treat each video_id as having no frame info
                for vid in df['video_id'].unique():
                    summary[vid] = set()
                return summary
            for vid, group in df.groupby('video_id'):
                summary[str(vid)] = set(group['frame_index'].dropna().astype(int).tolist())
            return summary
        except Exception:
            return {}

    def get_all_annotated_pairs(self, annotator: str) -> set:
        """
        Return the set of (video_id, frame_index) pairs already annotated by *annotator*.

        Used by the queue module to find the resume position efficiently.
        """
        try:
            df = pd.read_csv(self.annotations_file)
            df['video_id'] = df['video_id'].astype(str)
            df = df[df['annotator'] == annotator]
            if 'frame_index' not in df.columns:
                return set()
            pairs = set()
            for _, row in df.iterrows():
                fi = row['frame_index']
                if pd.notna(fi):
                    pairs.add((str(row['video_id']), int(fi)))
            return pairs
        except Exception:
            return set()

    def get_annotation_stats(self, annotator: str = None) -> Dict:
        """Get annotation statistics, optionally filtered to a single annotator."""
        try:
            df = pd.read_csv(self.annotations_file)
            df['video_id'] = df['video_id'].astype(str)

            if annotator:
                df = df[df['annotator'] == annotator]

            # Count unique (video_id, frame_index) pairs for frame-level count
            if 'frame_index' in df.columns and df['frame_index'].notna().any():
                frame_count = df.dropna(subset=['frame_index']).drop_duplicates(
                    subset=['video_id', 'frame_index']
                ).shape[0]
            else:
                frame_count = 0

            # Count fully-annotated videos
            total = df['video_id'].nunique()

            if total == 0 and frame_count == 0:
                return {
                    'total_annotated': 0,
                    'frame_count': 0,
                    'annotators': [],
                    'difficult_count': 0
                }

            return {
                'total_annotated': total,
                'frame_count': frame_count,
                'annotators': df['annotator'].unique().tolist(),
                'difficult_count': int(df['is_difficult'].sum()) if 'is_difficult' in df.columns else 0,
                'avg_time_sec': df['annotation_time_sec'].mean() if 'annotation_time_sec' in df.columns else 0,
                'by_annotator': df.groupby('annotator').size().to_dict()
            }
            
        except Exception as e:
            print(f"Error getting stats: {e}")
            return {'total_annotated': 0}
    
    def cache_predictions(
        self,
        video_id: str,
        filename: str,
        predictions: Dict
    ):
        """
        Cache model predictions for a video.
        
        Args:
            video_id: Video identifier
            filename: Video filename
            predictions: Full predictions dictionary
        """
        try:
            df = pd.read_csv(self.predictions_file)
            
            new_row = {
                'video_id': video_id,
                'filename': filename,
                'timestamp': datetime.now().isoformat(),
                'predictions_json': json.dumps(predictions)
            }
            
            # Update or insert
            existing_idx = df[df['video_id'] == video_id].index
            if len(existing_idx) > 0:
                for col, val in new_row.items():
                    df.loc[existing_idx[0], col] = val
            else:
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            
            df.to_csv(self.predictions_file, index=False)
            
        except Exception as e:
            print(f"Error caching predictions: {e}")
    
    def get_cached_predictions(self, video_id: str) -> Optional[Dict]:
        """Get cached predictions for a video."""
        try:
            df = pd.read_csv(self.predictions_file)
            row = df[df['video_id'] == video_id]
            
            if len(row) == 0:
                return None
            
            return json.loads(row.iloc[0]['predictions_json'])
            
        except Exception:
            return None
    
    def export_for_training(self, output_path: Optional[Path] = None) -> pd.DataFrame:
        """
        Export annotations in a format suitable for model training.
        
        Args:
            output_path: If provided, save to this path
            
        Returns:
            DataFrame with training-ready data
        """
        df = self.get_all_annotations()
        
        if len(df) == 0:
            return df
        
        # Select relevant columns for training
        training_cols = [
            'video_id',
            'filename',
            'perspective',
            'distance',
            'gaze',
            'pace',
            'density',
            'gesture'
        ]
        
        export_df = df[[c for c in training_cols if c in df.columns]].copy()
        
        # Remove rows with missing annotations
        export_df = export_df.dropna(subset=['perspective', 'distance', 'gaze'])
        
        if output_path:
            export_df.to_csv(output_path, index=False)
        
        return export_df
    
    def create_backup(self) -> Path:
        """
        Create a timestamped backup of annotations.
        
        Returns:
            Path to backup file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = self.backup_dir / f"annotations_backup_{timestamp}.csv"
        
        shutil.copy(self.annotations_file, backup_path)
        
        return backup_path
    
    def get_disagreements(self) -> pd.DataFrame:
        """
        Find cases where human annotation disagrees with model.
        
        Useful for identifying:
        - Model errors to correct
        - Difficult cases
        - Edge cases for retraining
        
        Returns:
            DataFrame with disagreement cases
        """
        df = self.get_all_annotations()
        
        if len(df) == 0:
            return df
        
        # Compare human vs model
        disagreements = []
        
        # Perspective
        if 'perspective' in df.columns and 'model_perspective' in df.columns:
            mask = (df['perspective'].notna()) & (df['model_perspective'].notna())
            mask = mask & (df['perspective'] != df['model_perspective'])
            disagreements.append(df[mask].assign(disagreement_type='perspective'))
        
        # Distance
        if 'distance' in df.columns and 'model_distance' in df.columns:
            mask = (df['distance'].notna()) & (df['model_distance'].notna())
            mask = mask & (df['distance'] != df['model_distance'])
            disagreements.append(df[mask].assign(disagreement_type='distance'))
        
        # Gaze
        if 'gaze' in df.columns and 'model_gaze' in df.columns:
            mask = (df['gaze'].notna()) & (df['model_gaze'].notna())
            mask = mask & (df['gaze'] != df['model_gaze'])
            disagreements.append(df[mask].assign(disagreement_type='gaze'))
        
        if disagreements:
            return pd.concat(disagreements, ignore_index=True)
        else:
            return pd.DataFrame()


def get_videos_needing_annotation(
    video_list: List[Dict],
    db: AnnotationDatabase,
    prioritize_low_confidence: bool = True,
    predictions_cache: Optional[Dict] = None
) -> List[Dict]:
    """
    Get list of videos that need annotation, prioritized.
    
    Args:
        video_list: List of all videos
        db: Annotation database
        prioritize_low_confidence: Put low-confidence predictions first
        predictions_cache: Optional dict of cached predictions
        
    Returns:
        Prioritized list of videos needing annotation
    """
    annotated_ids = db.get_annotated_video_ids()
    
    # Filter to unannotated
    unannotated = [
        v for v in video_list 
        if v.get('video_id', v.get('filename', '')) not in annotated_ids
    ]
    
    if prioritize_low_confidence and predictions_cache:
        # Sort by confidence (lowest first)
        def get_min_confidence(video):
            video_id = video.get('video_id', video.get('filename', ''))
            preds = predictions_cache.get(video_id, {})
            
            confidences = []
            for model_preds in preds.values():
                if isinstance(model_preds, dict) and 'confidence' in model_preds:
                    confidences.append(model_preds['confidence'])
            
            return min(confidences) if confidences else 1.0
        
        unannotated.sort(key=get_min_confidence)
    
    return unannotated
