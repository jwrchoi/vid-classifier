"""
Running Shoe Video Classifier - Annotation Dashboard
=====================================================

A Streamlit-based tool for human coders to annotate TikTok video frames
with perspective (POV) and social distance labels.

IMPORTANT DESIGN DECISIONS:
- Frames are presented in randomized order with NO video grouping
  to prevent consistency bias (coders cannot anchor on earlier frames
  from the same video).
- Model predictions are NOT shown to coders to avoid anchoring bias.
- If an active-learning queue CSV exists, it determines frame order.

How to run:
    streamlit run app.py

Author: Royce Choi
Purpose: Parasocial interaction research on TikTok content
"""

# =============================================================================
# IMPORTS
# =============================================================================

import streamlit as st
import pandas as pd
from pathlib import Path
import time
import sys
import io

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    OUTPUT_DIR,
    DEVICE,
    MODEL_CONFIGS,
    ACTIVE_MODELS,
    GCS_BUCKET_NAME,
    VIDEO_LIST_FILE,
    FRAMES_PER_VIDEO,
    QUEUE_CSV_PATH,
    QUEUE_SEED_SALT,
    find_models_dir,
    ensure_output_dir,
)
from utils.video_processing import extract_video_id
from utils.gcs import fetch_video_bytes, fetch_video_frames
from utils.database import AnnotationDatabase
from utils.queue import get_effective_queue, find_resume_position

try:
    from models.model_loader import ModelLoader
except ImportError:
    ModelLoader = None

# =============================================================================
# STREAMLIT PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Video Annotation Tool",
    page_icon="📹",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================

def init_session_state():
    """Initialize Streamlit session state variables."""
    defaults = {
        'initialized': False,
        # Queue-based navigation
        'frame_queue': [],          # list of (video_id, frame_index) tuples
        'queue_position': 0,        # current index in frame_queue
        # Caches
        'frame_cache': {},          # video_id -> list[PNG bytes]
        'video_by_id': {},          # video_id -> video info dict
        'videos': [],               # full video list (for reference)
        'predictions_cache': {},
        # Annotation state
        'annotation_start_time': None,
        'annotator_name': '',
        'model_loader': None,
        'db': None,
        'save_success': False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# =============================================================================
# LOADING FUNCTIONS
# =============================================================================

@st.cache_resource
def load_models():
    """Load trained ResNet-50 models for inference (cached)."""
    models_dir = find_models_dir()
    if not models_dir.exists():
        return None, f"Models directory not found: {models_dir}"
    loader = ModelLoader(models_dir, MODEL_CONFIGS, DEVICE)
    loader.load_all_models(ACTIVE_MODELS)
    if not loader.loaded_models:
        return None, "No models loaded (model files may be missing)"
    return loader, None


def load_video_list():
    """Load the fixed video list from CSV."""
    if not VIDEO_LIST_FILE.exists():
        return [], f"Video list not found: {VIDEO_LIST_FILE}."
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
# PREDICTION HELPERS
# =============================================================================

def get_predictions_for_frame(gcs_path: str, frame_png_bytes: bytes) -> dict:
    """Run models on a single frame. Predictions are saved, NOT shown."""
    if st.session_state.model_loader is None:
        return {}
    loader = st.session_state.model_loader
    frame_np = _png_to_numpy(frame_png_bytes)
    predictions = {}
    for model_name in ACTIVE_MODELS:
        if model_name in loader.loaded_models:
            result = loader.predict_video([frame_np], model_name)
            predictions[model_name] = result
    return predictions


def _png_to_numpy(png_bytes):
    """Convert PNG bytes to a numpy array (RGB)."""
    from PIL import Image
    import numpy as np
    img = Image.open(io.BytesIO(png_bytes))
    return np.array(img)


# =============================================================================
# FRAME LOADING
# =============================================================================

def ensure_frames_cached(video_id: str):
    """Load frames for *video_id* into the frame cache if not already there."""
    if video_id not in st.session_state.frame_cache:
        video_info = st.session_state.video_by_id.get(video_id)
        if video_info:
            frames = fetch_video_frames(GCS_BUCKET_NAME, video_info['gcs_path'])
            st.session_state.frame_cache[video_id] = frames


def prefetch_upcoming(queue, position, lookahead=2):
    """Warm the Streamlit cache for upcoming video_ids."""
    seen = set()
    for i in range(position + 1, min(position + 1 + lookahead * FRAMES_PER_VIDEO, len(queue))):
        vid_id = queue[i][0]
        if vid_id not in seen and vid_id not in st.session_state.frame_cache:
            seen.add(vid_id)
            video_info = st.session_state.video_by_id.get(vid_id)
            if video_info:
                fetch_video_frames(GCS_BUCKET_NAME, video_info['gcs_path'])


# =============================================================================
# UI COMPONENTS
# =============================================================================

def render_sidebar():
    """Render simplified sidebar with queue-based progress and navigation."""
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
        st.title("Video Annotation")
        st.caption(f"Logged in as: **{st.session_state.annotator_name}**")
        st.divider()

        # Progress
        queue = st.session_state.frame_queue
        pos = st.session_state.queue_position
        total = len(queue) if queue else 0

        if total > 0:
            # Count how many queue items have been annotated
            annotated_pairs = set()
            if st.session_state.db:
                annotated_pairs = st.session_state.db.get_all_annotated_pairs(
                    st.session_state.annotator_name
                )
            annotated_count = sum(1 for p in queue if p in annotated_pairs)

            st.subheader("Progress")
            st.progress(annotated_count / total)
            st.write(f"**{annotated_count}** / {total} frames annotated")
            st.write(f"Viewing: Frame {pos + 1} of {total}")

        st.divider()

        # Navigation
        st.subheader("Navigation")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("⬅️ Prev", use_container_width=True):
                if pos > 0:
                    st.session_state.queue_position = pos - 1
                    st.rerun()
        with col2:
            if st.button("Next ➡️", use_container_width=True):
                if pos < total - 1:
                    st.session_state.queue_position = pos + 1
                    st.rerun()

        # Jump to frame
        if total > 0:
            frame_num = st.number_input(
                "Go to frame #",
                min_value=1,
                max_value=total,
                value=pos + 1,
            )
            if st.button("Go", use_container_width=True):
                st.session_state.queue_position = frame_num - 1
                st.rerun()

        st.divider()

        if st.button("📖 View Instructions", use_container_width=True):
            render_instructions()


def render_single_frame(video_id: str, frame_index: int):
    """Display a single frame — no thumbnails, no video info."""
    ensure_frames_cached(video_id)
    frames = st.session_state.frame_cache.get(video_id, [])
    if not frames:
        st.error("Could not load frames for this item.")
        return
    fi = min(frame_index, len(frames) - 1)
    st.image(frames[fi], use_container_width=True)


def render_annotation_form(video_id: str, predictions: dict, frame_index: int):
    """Render the annotation form for the current frame."""
    st.subheader("📝 Your Annotation")

    existing = None
    if st.session_state.db:
        existing = st.session_state.db.get_annotation(
            video_id, annotator=st.session_state.annotator_name,
            frame_index=frame_index,
        )

    if st.session_state.annotation_start_time is None:
        st.session_state.annotation_start_time = time.time()

    # Widget key based on queue position (unique per frame slot)
    qpos = st.session_state.queue_position
    vid_key = f"q{qpos}"

    with st.form(key=f"annotation_form_{vid_key}"):

        # Step 1: Screener
        st.markdown("### Step 1: Screener Question")
        st.markdown("**Is there a human (or part of a human) visible in this frame?**")

        no_human_default = False
        if existing and existing.get('no_human_visible'):
            no_human_default = bool(existing['no_human_visible'])

        no_human_visible = st.checkbox(
            "No human visible in this frame",
            value=no_human_default,
            key=f"no_human_{vid_key}",
            help="Check this if the frame shows ONLY products, scenery, text, or graphics without any person or body parts visible",
        )

        st.divider()

        # Step 2: Labels
        st.markdown("### Step 2: Code the Frame")

        # Perspective
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
            label_visibility="collapsed",
        )

        # Distance
        st.markdown("**Social Distance (Camera Proximity)**")
        if no_human_visible:
            st.caption("↳ Will be saved as NA because \"No human visible\" is checked. "
                       "Uncheck the box above and re-submit to choose a different value.")
        st.caption(
            "• **Personal**: Close-up, face fills frame, intimate feeling\n"
            "• **Social**: Conversational distance, head-and-shoulders\n"
            "• **Public**: Wide shot, full body, formal/distant feeling\n"
            "• **NA**: Cannot determine or doesn't apply"
        )
        distance_opts = ['Personal', 'Social', 'Public', 'NA']
        default_d = 0
        if no_human_visible:
            default_d = distance_opts.index('NA')
        elif existing and existing.get('distance'):
            try:
                default_d = distance_opts.index(existing['distance'])
            except ValueError:
                pass
        distance = st.radio(
            "Select distance:",
            distance_opts,
            index=default_d,
            key=f"distance_{vid_key}",
            label_visibility="collapsed",
        )

        st.divider()

        # Step 3: Notes
        st.markdown("### Step 3: Additional Info (Optional)")
        col_notes, col_flag = st.columns([3, 1])
        with col_notes:
            notes = st.text_input(
                "Notes (optional)",
                value=existing.get('notes', '') if existing else '',
                key=f"notes_{vid_key}",
                help="Add any notes about this frame",
            )
        with col_flag:
            is_difficult = st.checkbox(
                "Difficult case",
                value=existing.get('is_difficult', False) if existing else False,
                key=f"difficult_{vid_key}",
                help="Check this if the frame was hard to code",
            )

        st.divider()

        col_btn1, col_btn2, col_spacer = st.columns([1, 1, 2])
        with col_btn1:
            submit_next = st.form_submit_button(
                "💾 Save & Next", type="primary", use_container_width=True,
            )
        with col_btn2:
            submit_only = st.form_submit_button(
                "💾 Save", use_container_width=True,
            )

    # Handle submission
    if submit_next or submit_only:
        annotation_time = 0
        if st.session_state.annotation_start_time:
            annotation_time = time.time() - st.session_state.annotation_start_time

        annotations = {
            'perspective': perspective,
            'distance': 'NA' if no_human_visible else distance,
            'no_human_visible': no_human_visible,
        }

        model_preds = {
            'perspective': predictions.get('pov_multi', {}),
            'distance': predictions.get('social_distance_multi', {}),
        }

        video_info = st.session_state.video_by_id.get(video_id, {})

        if not st.session_state.db:
            st.error("Database not initialized. Please refresh the page.")
            return

        with st.spinner("Saving annotation..."):
            success = st.session_state.db.save_annotation(
                video_id=video_id,
                filename=video_info.get('filename', ''),
                annotations=annotations,
                model_predictions=model_preds,
                computed_features={},
                annotator=st.session_state.annotator_name,
                notes=notes,
                is_difficult=is_difficult,
                annotation_time_sec=annotation_time,
                frame_index=frame_index,
                frame_total=FRAMES_PER_VIDEO,
            )
            if not success:
                time.sleep(0.5)
                success = st.session_state.db.save_annotation(
                    video_id=video_id,
                    filename=video_info.get('filename', ''),
                    annotations=annotations,
                    model_predictions=model_preds,
                    computed_features={},
                    annotator=st.session_state.annotator_name,
                    notes=notes,
                    is_difficult=is_difficult,
                    annotation_time_sec=annotation_time,
                    frame_index=frame_index,
                    frame_total=FRAMES_PER_VIDEO,
                )

        if success:
            st.session_state.annotation_start_time = None
            st.session_state.save_success = True
            if submit_next:
                queue = st.session_state.frame_queue
                pos = st.session_state.queue_position
                if pos < len(queue) - 1:
                    st.session_state.queue_position = pos + 1
                    st.rerun()
            else:
                st.success("Annotation saved!")
                st.session_state.save_success = False
        else:
            st.error("Failed to save annotation. Please try again.")


@st.dialog("Coding Instructions", width="large")
def render_instructions():
    """Render coding instructions as a modal dialog."""
    st.html(
        '<script>requestAnimationFrame(() => {'
        'const d = document.querySelector("[data-testid=stDialog] [data-testid=stVerticalBlockBorderWrapper]");'
        'if (d) d.scrollTop = 0;'
        '});</script>'
    )
    instructions_path = Path(__file__).parent / "coding_instructions.md"
    if instructions_path.exists():
        with open(instructions_path, 'r') as f:
            st.markdown(f.read())
    else:
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
    """Render a login screen requiring the coder to enter their name."""
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
    """Main application entry point."""
    init_session_state()

    # Login gate
    if not st.session_state.annotator_name:
        render_login()
        return

    # One-time initialization
    if not st.session_state.initialized:

        # Load models (optional)
        if ModelLoader is not None:
            with st.spinner("Loading models..."):
                loader, error = load_models()
                if error:
                    st.warning(f"⚠️ {error}")
                    st.info("The app will work without models. Predictions won't be saved.")
                else:
                    st.session_state.model_loader = loader

        # Load video list
        with st.spinner("Loading video list..."):
            videos, error = load_video_list()
            if error:
                st.error(f"❌ {error}")
                return
            st.session_state.videos = videos
            st.session_state.video_by_id = {v['video_id']: v for v in videos}

        # Create output directory and database
        ensure_output_dir()
        st.session_state.db = AnnotationDatabase(OUTPUT_DIR)

        # Purge legacy video-level rows (frame_index=NaN) once per session
        n_purged = st.session_state.db.purge_legacy_rows()
        if n_purged:
            st.info(f"Cleaned up {n_purged} legacy annotation(s). A backup was created.")

        # Build queue
        with st.spinner("Building frame queue..."):
            queue = get_effective_queue(
                videos=videos,
                frames_per_video=FRAMES_PER_VIDEO,
                annotator=st.session_state.annotator_name,
                queue_csv_path=QUEUE_CSV_PATH,
                seed_salt=QUEUE_SEED_SALT,
            )
            st.session_state.frame_queue = queue

            # Resume position
            annotated_pairs = st.session_state.db.get_all_annotated_pairs(
                st.session_state.annotator_name
            )
            st.session_state.queue_position = find_resume_position(queue, annotated_pairs)

        st.session_state.initialized = True
        st.rerun()

    # =========================================================================
    # Render Main Interface
    # =========================================================================

    if st.session_state.save_success:
        st.toast("Annotation saved!")
        st.session_state.save_success = False

    render_sidebar()

    queue = st.session_state.frame_queue
    if not queue:
        st.warning("No frames to annotate. Please check the video list.")
        return

    pos = st.session_state.queue_position
    video_id, frame_index = queue[pos]

    # Load frames BEFORE creating columns so the GCS download doesn't
    # block the right column (annotation form) from rendering.
    with st.spinner("Loading frame..."):
        ensure_frames_cached(video_id)

    # Side-by-side layout
    frame_col, form_col = st.columns([2, 3])

    with frame_col:
        st.subheader(f"Frame {pos + 1} of {len(queue)}")

        # Show if already annotated
        if st.session_state.db:
            existing = st.session_state.db.get_annotation(
                video_id, annotator=st.session_state.annotator_name,
                frame_index=frame_index,
            )
            if existing:
                st.info("You've annotated this frame. You can update your response below.")

        render_single_frame(video_id, frame_index)

    with form_col:
        predictions = {}
        frames = st.session_state.frame_cache.get(video_id, [])
        if st.session_state.model_loader and frames and frame_index < len(frames):
            predictions = get_predictions_for_frame(
                st.session_state.video_by_id[video_id]['gcs_path'],
                frames[frame_index],
            )
        render_annotation_form(video_id, predictions, frame_index=frame_index)

    # Prefetch upcoming video frames AFTER both columns have rendered
    prefetch_upcoming(queue, pos)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()
