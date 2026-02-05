"""
Running Shoe Video Classifier - Annotation Dashboard
=====================================================

A Streamlit-based tool for human coders to annotate TikTok videos
with perspective (POV) and social distance labels.

IMPORTANT DESIGN DECISIONS:
- Model predictions are NOT shown to coders to avoid anchoring bias
- Models run in the background; predictions are saved to CSV for later analysis
- Only two annotation dimensions: Perspective and Distance (plus screener)

How to run:
    streamlit run app.py

Author: Royce Choi
Purpose: Parasocial interaction research on TikTok content
"""

# =============================================================================
# IMPORTS
# =============================================================================

import streamlit as st          # Web UI framework
import pandas as pd             # Data handling
import numpy as np              # Numerical operations
from pathlib import Path        # Cross-platform file paths
import time                     # For timing annotations
import sys                      # System utilities

# Add project directory to Python path so we can import our modules
# This is needed because Streamlit runs from the app.py directory
sys.path.insert(0, str(Path(__file__).parent))

# Import our custom modules
from config import (
    VIDEO_DIR,                      # Where video files are stored
    OUTPUT_DIR,                     # Where to save annotations
    DEVICE,                         # PyTorch device (cpu/cuda/mps)
    MODEL_CONFIGS,                  # Model definitions
    ACTIVE_MODELS,                  # Which models to use
    FRAME_SAMPLE_INTERVAL,          # How often to sample frames
    MAX_FRAMES_PER_VIDEO,           # Maximum frames per video
    SUPPORTED_VIDEO_EXTENSIONS,     # Valid video file types
    find_models_dir,                # Helper to find models directory
    ensure_output_dir               # Helper to create output directory
)
from utils.video_processing import (
    list_videos,                    # List all videos in a directory
    sample_frames,                  # Extract frames from video
    get_video_info,                 # Get video metadata
    extract_video_id                # Extract ID from filename
)
from utils.database import AnnotationDatabase  # CSV-based annotation storage
from models.model_loader import ModelLoader    # Load and run ResNet models

# =============================================================================
# STREAMLIT PAGE CONFIGURATION
# =============================================================================

# Configure the Streamlit page
# This must be the first Streamlit command in the script
st.set_page_config(
    page_title="Video Annotation Tool",  # Browser tab title
    page_icon="📹",                       # Browser tab icon
    layout="wide",                        # Use full screen width
    initial_sidebar_state="expanded"      # Sidebar starts open
)

# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================

def init_session_state():
    """
    Initialize Streamlit session state variables.

    Session state persists across reruns of the app (e.g., when user clicks a button).
    We use it to store:
    - Which video we're currently viewing
    - Loaded models and video list
    - Cached predictions (so we don't recompute)
    - Database connection

    All variables are initialized with sensible defaults if they don't exist.
    """
    # Define all session state variables and their default values
    defaults = {
        'initialized': False,           # Has the app finished loading?
        'current_video_idx': 0,         # Index of current video in list
        'videos': [],                   # List of all video file info
        'predictions_cache': {},        # Cached model predictions by video_id
        'annotation_start_time': None,  # When user started annotating current video
        'annotator_name': 'default',    # Name of the human coder
        'model_loader': None,           # ModelLoader instance (or None if no models)
        'db': None,                     # AnnotationDatabase instance
        'show_instructions': False      # Whether to show coding instructions
    }

    # Set defaults only for variables that don't already exist
    # (This preserves values across reruns)
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# =============================================================================
# LOADING FUNCTIONS
# =============================================================================

@st.cache_resource
def load_models():
    """
    Load trained ResNet-50 models for inference.

    The @st.cache_resource decorator means this function only runs once,
    even if the app reruns. Models stay in memory for fast inference.

    IMPORTANT: Models are used internally to generate predictions that are
    saved to CSV, but predictions are NOT shown to human coders (to avoid bias).

    Returns:
        tuple: (ModelLoader instance or None, error message or None)
    """
    # Find where the model files are stored
    models_dir = find_models_dir()

    # Check if the directory exists
    if not models_dir.exists():
        return None, f"Models directory not found: {models_dir}"

    # Create model loader and load all active models
    loader = ModelLoader(models_dir, MODEL_CONFIGS, DEVICE)
    loader.load_all_models(ACTIVE_MODELS)

    # Check if any models were actually loaded
    if not loader.loaded_models:
        return None, "No models loaded (model files may be missing)"

    return loader, None


def load_videos(video_dir: Path):
    """
    Load list of video files from the specified directory.

    Args:
        video_dir: Path to directory containing video files

    Returns:
        tuple: (list of video info dicts, error message or None)

    Each video dict contains:
        - path: Full path to video file
        - filename: Just the filename
        - size_mb: File size in megabytes
        - video_id: Extracted ID for tracking
    """
    # Check if directory exists
    if not video_dir.exists():
        return [], f"Video directory not found: {video_dir}"

    # Get list of videos matching our supported extensions
    videos = list_videos(video_dir, SUPPORTED_VIDEO_EXTENSIONS)

    # Check if we found any videos
    if not videos:
        return [], f"No videos found in: {video_dir}"

    # Extract video ID from each filename for tracking
    # This ID is used as the primary key in our annotation database
    for v in videos:
        v['video_id'] = extract_video_id(v['filename'])

    return videos, None


# =============================================================================
# PREDICTION FUNCTIONS (Internal - not shown to users)
# =============================================================================

def get_predictions_for_video(video_path: str, frames: list) -> dict:
    """
    Get model predictions for a video.

    IMPORTANT: These predictions are saved to CSV for later analysis,
    but are NOT displayed to human coders (to avoid anchoring bias).

    Args:
        video_path: Path to the video file
        frames: List of extracted video frames (numpy arrays)

    Returns:
        dict: Model predictions keyed by model name
              Each prediction contains: prediction, confidence, model_name, etc.
    """
    # Extract video ID for caching
    video_id = extract_video_id(Path(video_path).name)

    # Check if we already have predictions cached for this video
    # (Avoids recomputing every time the app reruns)
    if video_id in st.session_state.predictions_cache:
        return st.session_state.predictions_cache[video_id]

    # If no model loader, return empty (app can work without models)
    if st.session_state.model_loader is None:
        return {}

    # Run each active model on the video frames
    predictions = {}
    loader = st.session_state.model_loader

    for model_name in ACTIVE_MODELS:
        if model_name in loader.loaded_models:
            # predict_video() runs model on all frames and aggregates results
            result = loader.predict_video(frames, model_name)
            predictions[model_name] = result

    # Cache predictions for this video
    st.session_state.predictions_cache[video_id] = predictions

    return predictions


# =============================================================================
# UI COMPONENTS
# =============================================================================

def render_sidebar():
    """
    Render the sidebar with navigation controls and progress tracking.

    The sidebar contains:
    - Annotator name input (to track who made each annotation)
    - Progress bar showing how many videos have been annotated
    - Navigation buttons (Previous, Next)
    - Jump to specific video number
    - Link to coding instructions
    """
    with st.sidebar:
        # =====================================================================
        # Header
        # =====================================================================
        st.title("📹 Video Annotation")

        # Annotator name input
        # This name is saved with each annotation for tracking purposes
        st.session_state.annotator_name = st.text_input(
            "Your Name",
            value=st.session_state.annotator_name,
            help="Enter your name to track who made each annotation"
        )

        st.divider()

        # =====================================================================
        # Progress Tracking
        # =====================================================================
        if st.session_state.videos:
            total = len(st.session_state.videos)
            current = st.session_state.current_video_idx + 1

            # Get count of annotated videos from database
            annotated = 0
            if st.session_state.db:
                stats = st.session_state.db.get_annotation_stats()
                annotated = stats.get('total_annotated', 0)

            st.subheader("Progress")

            # Visual progress bar
            st.progress(annotated / total if total > 0 else 0)

            # Text progress info
            st.write(f"**{annotated}** of {total} videos annotated")
            st.write(f"Currently viewing: #{current}")

        st.divider()

        # =====================================================================
        # Navigation Controls
        # =====================================================================
        st.subheader("Navigation")

        # Previous/Next buttons side by side
        col1, col2 = st.columns(2)

        with col1:
            # Previous button - go back one video
            if st.button("⬅️ Prev", use_container_width=True):
                if st.session_state.current_video_idx > 0:
                    st.session_state.current_video_idx -= 1
                    st.rerun()  # Refresh the page with new video

        with col2:
            # Next button - go forward one video
            if st.button("Next ➡️", use_container_width=True):
                if st.session_state.current_video_idx < len(st.session_state.videos) - 1:
                    st.session_state.current_video_idx += 1
                    st.rerun()

        # Jump to specific video number
        if st.session_state.videos:
            video_num = st.number_input(
                "Go to video #",
                min_value=1,
                max_value=len(st.session_state.videos),
                value=st.session_state.current_video_idx + 1
            )
            if st.button("Go", use_container_width=True):
                st.session_state.current_video_idx = video_num - 1
                st.rerun()

        st.divider()

        # =====================================================================
        # Instructions Toggle
        # =====================================================================
        if st.button("📖 View Instructions", use_container_width=True):
            st.session_state.show_instructions = not st.session_state.show_instructions
            st.rerun()


def render_video_player(video_path: str):
    """
    Render the video player component.

    Args:
        video_path: Full path to the video file to play

    Streamlit's built-in video player handles:
    - Play/pause controls
    - Seeking
    - Volume control
    - Fullscreen toggle
    """
    st.video(video_path)


def render_annotation_form(video_id: str, predictions: dict):
    """
    Render the annotation form for human coders.

    This is the main interface where coders select labels for each video.

    IMPORTANT DESIGN DECISIONS:
    - Model predictions are NOT shown to coders (to avoid anchoring bias)
    - Only two main annotation dimensions: Perspective and Distance
    - "No human visible" checkbox affects which options are available
    - NA option available for ambiguous cases

    Args:
        video_id: Unique identifier for this video
        predictions: Model predictions (used for saving, NOT displayed)
    """
    st.subheader("📝 Your Annotation")

    # =========================================================================
    # Load existing annotation if video was already annotated
    # =========================================================================
    existing = None
    if st.session_state.db:
        existing = st.session_state.db.get_annotation(video_id)

    # Start timing how long the annotation takes
    if st.session_state.annotation_start_time is None:
        st.session_state.annotation_start_time = time.time()

    # =========================================================================
    # Annotation Form
    # =========================================================================
    with st.form(key="annotation_form"):

        # ---------------------------------------------------------------------
        # Screener Question: Is there a human visible?
        # ---------------------------------------------------------------------
        st.markdown("### Step 1: Screener Question")
        st.markdown("**Is there a human (or part of a human) visible in this video?**")

        # Get default value from existing annotation if available
        no_human_default = False
        if existing and existing.get('no_human_visible'):
            no_human_default = bool(existing['no_human_visible'])

        # Checkbox for "No human visible"
        # When checked, some annotation options become NA automatically
        no_human_visible = st.checkbox(
            "No human visible in this video",
            value=no_human_default,
            help="Check this if the video shows ONLY products, scenery, text, or graphics without any person or body parts visible"
        )

        st.divider()

        # ---------------------------------------------------------------------
        # Main Annotation Questions
        # ---------------------------------------------------------------------
        st.markdown("### Step 2: Code the Video")

        # Create two columns for the annotation options
        col_left, col_right = st.columns(2)

        # ---------------------------------------------------------------------
        # PERSPECTIVE (POV) - Left Column
        # ---------------------------------------------------------------------
        with col_left:
            st.markdown("**Perspective (Point of View)**")

            # Help text explaining each option
            st.caption(
                "• **1st person**: Camera shows YOUR perspective (hands visible, POV shot)\n"
                "• **2nd person**: Subject talks TO YOU (eye contact, direct address)\n"
                "• **3rd person**: You're watching others (documentary style, no direct address)\n"
                "• **NA**: Cannot determine or doesn't apply"
            )

            # Perspective options
            perspective_opts = ['1st person', '2nd person', '3rd person', 'NA']

            # Determine default selection
            default_p = 0  # Default to first option
            if existing and existing.get('perspective'):
                try:
                    default_p = perspective_opts.index(existing['perspective'])
                except ValueError:
                    pass  # Keep default if not found

            # Radio buttons for perspective selection
            perspective = st.radio(
                "Select perspective:",
                perspective_opts,
                index=default_p,
                label_visibility="collapsed"  # Hide the label (we have our own header)
            )

        # ---------------------------------------------------------------------
        # SOCIAL DISTANCE - Right Column
        # ---------------------------------------------------------------------
        with col_right:
            st.markdown("**Social Distance (Camera Proximity)**")

            # If no human visible, distance is automatically NA
            if no_human_visible:
                st.info("↳ Automatically set to NA (no human visible)")
                distance = "NA"
            else:
                # Help text explaining each option
                st.caption(
                    "• **Personal**: Close-up, face fills frame, intimate feeling\n"
                    "• **Social**: Conversational distance, head-and-shoulders\n"
                    "• **Public**: Wide shot, full body, formal/distant feeling\n"
                    "• **NA**: Cannot determine or doesn't apply"
                )

                # Distance options
                distance_opts = ['Personal', 'Social', 'Public', 'NA']

                # Determine default selection
                default_d = 0
                if existing and existing.get('distance'):
                    try:
                        default_d = distance_opts.index(existing['distance'])
                    except ValueError:
                        pass

                # Radio buttons for distance selection
                distance = st.radio(
                    "Select distance:",
                    distance_opts,
                    index=default_d,
                    label_visibility="collapsed"
                )

        st.divider()

        # ---------------------------------------------------------------------
        # Notes and Difficult Flag
        # ---------------------------------------------------------------------
        st.markdown("### Step 3: Additional Info (Optional)")

        col_notes, col_flag = st.columns([3, 1])

        with col_notes:
            # Free-text notes field
            notes = st.text_input(
                "Notes (optional)",
                value=existing.get('notes', '') if existing else '',
                help="Add any notes about this video (e.g., edge cases, uncertainties)"
            )

        with col_flag:
            # Flag for difficult/ambiguous cases
            is_difficult = st.checkbox(
                "Difficult case",
                value=existing.get('is_difficult', False) if existing else False,
                help="Check this if the video was hard to code (for quality review)"
            )

        st.divider()

        # ---------------------------------------------------------------------
        # Submit Buttons
        # ---------------------------------------------------------------------
        col_btn1, col_btn2, col_spacer = st.columns([1, 1, 2])

        with col_btn1:
            # Save and move to next video
            submit_next = st.form_submit_button(
                "💾 Save & Next",
                type="primary",
                use_container_width=True
            )

        with col_btn2:
            # Save but stay on current video
            submit_only = st.form_submit_button(
                "💾 Save",
                use_container_width=True
            )

    # =========================================================================
    # Handle Form Submission
    # =========================================================================
    if submit_next or submit_only:
        # Calculate how long the annotation took
        annotation_time = 0
        if st.session_state.annotation_start_time:
            annotation_time = time.time() - st.session_state.annotation_start_time

        # Package up the human annotations
        annotations = {
            'perspective': perspective,
            'distance': distance,
            'no_human_visible': no_human_visible
        }

        # Package up model predictions (for saving, not showing)
        # This allows us to compare human vs model labels later
        model_preds = {
            'perspective': predictions.get('pov_multi', {}),
            'distance': predictions.get('social_distance_multi', {})
        }

        # Get current video info
        current_video = st.session_state.videos[st.session_state.current_video_idx]

        # Save annotation to database
        if st.session_state.db:
            success = st.session_state.db.save_annotation(
                video_id=video_id,
                filename=current_video['filename'],
                annotations=annotations,
                model_predictions=model_preds,
                computed_features={},  # Not using computed features in this version
                annotator=st.session_state.annotator_name,
                notes=notes,
                is_difficult=is_difficult,
                annotation_time_sec=annotation_time
            )

            if success:
                st.success("✅ Annotation saved!")

                # Reset the annotation timer
                st.session_state.annotation_start_time = None

                # If "Save & Next" was clicked, move to next video
                if submit_next and st.session_state.current_video_idx < len(st.session_state.videos) - 1:
                    st.session_state.current_video_idx += 1
                    st.rerun()
            else:
                st.error("❌ Failed to save annotation. Please try again.")


def render_instructions():
    """
    Render the coding instructions page.

    This shows the full coding instructions from coding_instructions.md
    to help coders understand how to annotate videos consistently.
    """
    st.subheader("📖 Coding Instructions")

    # Try to load instructions from markdown file
    instructions_path = Path(__file__).parent / "coding_instructions.md"

    if instructions_path.exists():
        with open(instructions_path, 'r') as f:
            st.markdown(f.read())
    else:
        # Fallback if file doesn't exist
        st.warning("Instructions file not found.")
        st.markdown("""
        ## Quick Guide

        **Perspective (POV)**:
        - 1st person: Camera shows your perspective (hands visible, POV)
        - 2nd person: Subject talks to you (eye contact, direct address)
        - 3rd person: Watching others (documentary style)
        - NA: Cannot determine

        **Social Distance**:
        - Personal: Close-up, face fills frame
        - Social: Conversational distance
        - Public: Wide shot, full body
        - NA: Cannot determine or no human visible
        """)

    # Close button
    if st.button("Close Instructions"):
        st.session_state.show_instructions = False
        st.rerun()


# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """
    Main application entry point.

    This function:
    1. Initializes session state
    2. Loads models and videos (on first run)
    3. Renders the appropriate UI based on current state

    The app flow is:
    - Initialize -> Load resources -> Show annotation interface
    - User interacts with form -> Save -> Move to next video
    """
    # Initialize session state variables
    init_session_state()

    # =========================================================================
    # Show Instructions Page (if requested)
    # =========================================================================
    if st.session_state.show_instructions:
        render_instructions()
        return

    # =========================================================================
    # One-time Initialization
    # =========================================================================
    if not st.session_state.initialized:

        # Load models (runs in background, predictions NOT shown to users)
        with st.spinner("Loading models..."):
            loader, error = load_models()
            if error:
                # Models are optional - app works without them
                st.warning(f"⚠️ {error}")
                st.info("The app will work without models. Predictions won't be saved.")
            else:
                st.session_state.model_loader = loader

        # Load video list
        with st.spinner("Loading videos..."):
            videos, error = load_videos(VIDEO_DIR)

            if error:
                # Videos are required - show error and exit
                st.error(f"❌ {error}")
                st.info("Please update VIDEO_DIR in config.py to point to your videos.")
                st.code(f"Current path: {VIDEO_DIR}")

                # Allow manual path entry as fallback
                manual_path = st.text_input("Or enter video directory path manually:")
                if manual_path and st.button("Load from path"):
                    videos, error = load_videos(Path(manual_path))
                    if not error:
                        st.session_state.videos = videos
                        st.rerun()
                return

            st.session_state.videos = videos

        # Create output directory and initialize database
        ensure_output_dir()
        st.session_state.db = AnnotationDatabase(OUTPUT_DIR)

        # Mark initialization as complete
        st.session_state.initialized = True
        st.rerun()

    # =========================================================================
    # Render Main Interface
    # =========================================================================

    # Render sidebar with navigation
    render_sidebar()

    # Check if we have videos to show
    if not st.session_state.videos:
        st.warning("No videos loaded. Please check the video directory path.")
        return

    # Get current video info
    current_video = st.session_state.videos[st.session_state.current_video_idx]
    video_path = current_video['path']
    video_id = current_video['video_id']

    # =========================================================================
    # Page Header
    # =========================================================================
    st.title(f"Video #{st.session_state.current_video_idx + 1}")
    st.caption(f"File: {current_video['filename']}")

    # =========================================================================
    # Main Content Layout
    # =========================================================================
    # Create two columns: video player on left, info on right
    col_video, col_info = st.columns([2, 1])

    # Video player
    with col_video:
        render_video_player(video_path)

    # Video info sidebar
    with col_info:
        st.subheader("📹 Video Info")

        # Get video metadata
        video_info = get_video_info(Path(video_path))

        if video_info and 'error' not in video_info:
            st.write(f"**Duration:** {video_info.get('duration_sec', 0):.1f} seconds")
            st.write(f"**Size:** {video_info.get('size_mb', 0):.1f} MB")
            st.write(f"**Resolution:** {video_info.get('width', 0)} x {video_info.get('height', 0)}")

        # Check if already annotated
        if st.session_state.db:
            existing = st.session_state.db.get_annotation(video_id)
            if existing:
                st.success("✅ Previously annotated")
                st.caption(f"By: {existing.get('annotator', 'Unknown')}")
            else:
                st.info("⏳ Not yet annotated")

    # =========================================================================
    # Run Model Predictions (in background)
    # =========================================================================
    # These are NOT shown to the user - they're saved for later analysis
    predictions = {}

    if st.session_state.model_loader:
        # Check if we have cached predictions
        if video_id not in st.session_state.predictions_cache:
            # Need to extract frames and run models
            with st.spinner("Processing..."):
                frames, meta = sample_frames(
                    Path(video_path),
                    FRAME_SAMPLE_INTERVAL,
                    MAX_FRAMES_PER_VIDEO
                )
                if frames:
                    predictions = get_predictions_for_video(video_path, frames)
        else:
            # Use cached predictions
            predictions = st.session_state.predictions_cache[video_id]

    # =========================================================================
    # Annotation Form
    # =========================================================================
    st.divider()
    render_annotation_form(video_id, predictions)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()
