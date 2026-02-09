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
from pathlib import Path        # Cross-platform file paths
import time                     # For timing annotations
import sys                      # System utilities

# Add project directory to Python path so we can import our modules
# This is needed because Streamlit runs from the app.py directory
sys.path.insert(0, str(Path(__file__).parent))

# Import our custom modules
from config import (
    OUTPUT_DIR,                     # Where to save annotations
    DEVICE,                         # PyTorch device (cpu/cuda/mps)
    MODEL_CONFIGS,                  # Model definitions
    ACTIVE_MODELS,                  # Which models to use
    GCS_BUCKET_NAME,                # GCS bucket for videos
    VIDEO_LIST_FILE,                # Fixed video list CSV
    find_models_dir,                # Helper to find models directory
    ensure_output_dir               # Helper to create output directory
)
from utils.video_processing import extract_video_id  # Extract ID from filename
from utils.gcs import fetch_video_bytes              # Stream video bytes from GCS
from utils.database import AnnotationDatabase        # CSV-based annotation storage

try:
    from models.model_loader import ModelLoader      # Load and run ResNet models
except ImportError:
    ModelLoader = None

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
        'annotator_name': '',             # Name of the human coder (set at login)
        'model_loader': None,           # ModelLoader instance (or None if no models)
        'db': None,                     # AnnotationDatabase instance
        'save_success': False           # Flag to show success toast after rerun
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


def load_video_list():
    """
    Load the fixed video list from CSV.

    The CSV is generated by scripts/generate_video_list.py and contains
    a fixed set of 50 videos that all coders annotate in the same order.

    Returns:
        tuple: (list of video info dicts, error message or None)

    Each video dict contains:
        - video_id: Unique identifier
        - filename: Video filename
        - gcs_path: Full blob path in GCS
    """
    if not VIDEO_LIST_FILE.exists():
        return [], f"Video list not found: {VIDEO_LIST_FILE}. Run scripts/generate_video_list.py first."

    df = pd.read_csv(VIDEO_LIST_FILE)

    if df.empty:
        return [], "Video list CSV is empty."

    videos = []
    for _, row in df.iterrows():
        videos.append({
            'video_id': str(row['video_id']),
            'filename': row['filename'],
            'gcs_path': row['gcs_path'],
        })

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
    # Make the sidebar collapse (X) button always visible when the sidebar is open.
    # Placed here (not at top level) to avoid interfering with initial_sidebar_state.
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] [data-testid="stSidebarCollapseButton"] {
            opacity: 1 !important;
            visibility: visible !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        # =====================================================================
        # Header
        # =====================================================================
        st.title("Video Annotation")
        st.caption(f"Logged in as: **{st.session_state.annotator_name}**")

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
                stats = st.session_state.db.get_annotation_stats(
                    annotator=st.session_state.annotator_name
                )
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
        # Instructions Dialog
        # =====================================================================
        if st.button("📖 View Instructions", use_container_width=True):
            render_instructions()


def render_video_player(gcs_path: str):
    """
    Render the video player by streaming bytes from GCS.

    Args:
        gcs_path: Full blob path in GCS bucket
    """
    with st.spinner("Loading video..."):
        video_bytes = fetch_video_bytes(GCS_BUCKET_NAME, gcs_path)
    st.video(video_bytes)


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
    # Load existing annotation if video was already annotated by this coder
    # =========================================================================
    existing = None
    if st.session_state.db:
        existing = st.session_state.db.get_annotation(
            video_id, annotator=st.session_state.annotator_name
        )

    # Start timing how long the annotation takes
    if st.session_state.annotation_start_time is None:
        st.session_state.annotation_start_time = time.time()

    # =========================================================================
    # Annotation Form
    #
    # Widget keys include the video index so that switching videos resets
    # all form fields to their defaults (or to the existing annotation).
    # Without this, Streamlit's widget state persists the previous video's
    # choices into the new video's form.
    # =========================================================================
    vid_key = f"v{st.session_state.current_video_idx}"

    with st.form(key=f"annotation_form_{vid_key}"):

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
            key=f"no_human_{vid_key}",
            help="Check this if the video shows ONLY products, scenery, text, or graphics without any person or body parts visible"
        )

        st.divider()

        # ---------------------------------------------------------------------
        # Main Annotation Questions
        # ---------------------------------------------------------------------
        st.markdown("### Step 2: Code the Video")

        # ---------------------------------------------------------------------
        # PERSPECTIVE (POV)
        # ---------------------------------------------------------------------
        st.markdown("**Perspective (Point of View)**")

        st.caption(
            "• **1st person**: Camera shows YOUR perspective (hands visible, POV shot)\n"
            "• **2nd person**: Subject talks TO YOU (eye contact, direct address)\n"
            "• **3rd person**: You're watching others (documentary style, no direct address)\n"
            "• **NA**: Cannot determine or doesn't apply"
        )

        perspective_opts = ['1st person', '2nd person', '3rd person', 'NA']

        default_p = 0
        if existing and existing.get('perspective'):
            try:
                default_p = perspective_opts.index(existing['perspective'])
            except ValueError:
                pass

        perspective = st.radio(
            "Select perspective:",
            perspective_opts,
            index=default_p,
            key=f"perspective_{vid_key}",
            label_visibility="collapsed"
        )

        # ---------------------------------------------------------------------
        # SOCIAL DISTANCE
        # ---------------------------------------------------------------------
        st.markdown("**Social Distance (Camera Proximity)**")

        if no_human_visible:
            st.info("↳ Automatically set to NA (no human visible)")
            distance = "NA"
        else:
            st.caption(
                "• **Personal**: Close-up, face fills frame, intimate feeling\n"
                "• **Social**: Conversational distance, head-and-shoulders\n"
                "• **Public**: Wide shot, full body, formal/distant feeling\n"
                "• **NA**: Cannot determine or doesn't apply"
            )

            distance_opts = ['Personal', 'Social', 'Public', 'NA']

            default_d = 0
            if existing and existing.get('distance'):
                try:
                    default_d = distance_opts.index(existing['distance'])
                except ValueError:
                    pass

            distance = st.radio(
                "Select distance:",
                distance_opts,
                index=default_d,
                key=f"distance_{vid_key}",
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
                key=f"notes_{vid_key}",
                help="Add any notes about this video (e.g., edge cases, uncertainties)"
            )

        with col_flag:
            # Flag for difficult/ambiguous cases
            is_difficult = st.checkbox(
                "Difficult case",
                value=existing.get('is_difficult', False) if existing else False,
                key=f"difficult_{vid_key}",
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

        if not st.session_state.db:
            st.error("Database not initialized. Please refresh the page.")
            return

        # Save annotation to database (with spinner and retry)
        with st.spinner("Saving annotation..."):
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

            # Retry once on failure (transient GCS FUSE errors)
            if not success:
                time.sleep(0.5)
                success = st.session_state.db.save_annotation(
                    video_id=video_id,
                    filename=current_video['filename'],
                    annotations=annotations,
                    model_predictions=model_preds,
                    computed_features={},
                    annotator=st.session_state.annotator_name,
                    notes=notes,
                    is_difficult=is_difficult,
                    annotation_time_sec=annotation_time
                )

        if success:
            # Reset the annotation timer
            st.session_state.annotation_start_time = None
            # Flag for toast on next rerun (st.success would vanish on rerun)
            st.session_state.save_success = True

            # If "Save & Next" was clicked, move to next video
            if submit_next and st.session_state.current_video_idx < len(st.session_state.videos) - 1:
                st.session_state.current_video_idx += 1
                st.rerun()
            else:
                # "Save" only — stay on current video, show confirmation now
                st.success("Annotation saved!")
                st.session_state.save_success = False
        else:
            st.error("Failed to save annotation. Please try again.")


@st.dialog("Coding Instructions", width="large")
def render_instructions():
    """
    Render the coding instructions as a modal dialog.

    Uses st.dialog so the main annotation view stays visible underneath
    and coders can close the dialog to return exactly where they were.
    """
    # Scroll the dialog container to the top on open.
    # Without this, Streamlit may focus on the last interactive element
    # (the Close button) and scroll to the bottom of the dialog.
    st.html(
        '<script>requestAnimationFrame(() => {'
        'const d = document.querySelector("[data-testid=stDialog] [data-testid=stVerticalBlockBorderWrapper]");'
        'if (d) d.scrollTop = 0;'
        '});</script>'
    )

    # Try to load instructions from markdown file
    instructions_path = Path(__file__).parent / "coding_instructions.md"

    if instructions_path.exists():
        with open(instructions_path, 'r') as f:
            st.markdown(f.read())
    else:
        # Fallback if file doesn't exist
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

    if st.button("Close", use_container_width=True):
        st.rerun()


# =============================================================================
# LOGIN GATE
# =============================================================================

def render_login():
    """
    Render a login screen requiring the coder to enter their name.

    Returns:
        True if logged in, False otherwise.
    """
    st.title("Video Annotation Tool")
    st.markdown("Enter your name to begin annotating.")

    name = st.text_input("Your Name", key="login_name_input")

    if st.button("Start", type="primary"):
        if name.strip():
            st.session_state.annotator_name = name.strip()
            st.rerun()
        else:
            st.warning("Please enter your name.")

    return bool(st.session_state.annotator_name)


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
    # Login Gate — must enter name before proceeding
    # =========================================================================
    if not st.session_state.annotator_name:
        render_login()
        return

    # =========================================================================
    # One-time Initialization
    # =========================================================================
    if not st.session_state.initialized:

        # Load models (runs in background, predictions NOT shown to users)
        if ModelLoader is not None:
            with st.spinner("Loading models..."):
                loader, error = load_models()
                if error:
                    # Models are optional - app works without them
                    st.warning(f"⚠️ {error}")
                    st.info("The app will work without models. Predictions won't be saved.")
                else:
                    st.session_state.model_loader = loader

        # Load video list from CSV
        with st.spinner("Loading video list..."):
            videos, error = load_video_list()

            if error:
                st.error(f"❌ {error}")
                return

            st.session_state.videos = videos

        # Create output directory and initialize database
        ensure_output_dir()
        st.session_state.db = AnnotationDatabase(OUTPUT_DIR)

        # Resume: jump coder to their first unannotated video
        annotated_ids = st.session_state.db.get_annotated_video_ids(
            annotator=st.session_state.annotator_name
        )
        for i, v in enumerate(videos):
            if v['video_id'] not in annotated_ids:
                st.session_state.current_video_idx = i
                break

        # Mark initialization as complete
        st.session_state.initialized = True
        st.rerun()

    # =========================================================================
    # Render Main Interface
    # =========================================================================

    # Show save confirmation toast (persists across the rerun triggered by Save & Next)
    if st.session_state.save_success:
        st.toast("Annotation saved!")
        st.session_state.save_success = False

    # Render sidebar with navigation
    render_sidebar()

    # Check if we have videos to show
    if not st.session_state.videos:
        st.warning("No videos loaded. Please check the video directory path.")
        return

    # Get current video info
    current_video = st.session_state.videos[st.session_state.current_video_idx]
    video_id = current_video['video_id']
    gcs_path = current_video['gcs_path']

    # =========================================================================
    # Side-by-side layout: Video (left) | Annotation form (right)
    # Columns stack vertically on narrow screens automatically.
    # =========================================================================
    video_col, form_col = st.columns([2, 3])

    with video_col:
        st.subheader(f"Video #{st.session_state.current_video_idx + 1}")
        st.caption(f"File: {current_video['filename']}")

        # Check if already annotated by this coder
        if st.session_state.db:
            existing = st.session_state.db.get_annotation(
                video_id, annotator=st.session_state.annotator_name
            )
            if existing:
                st.success("You have already annotated this video.")

        render_video_player(gcs_path)

    # Prefetch next video in background (warms the cache)
    if st.session_state.current_video_idx < len(st.session_state.videos) - 1:
        next_video = st.session_state.videos[st.session_state.current_video_idx + 1]
        fetch_video_bytes(GCS_BUCKET_NAME, next_video['gcs_path'])

    with form_col:
        predictions = {}
        render_annotation_form(video_id, predictions)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()
