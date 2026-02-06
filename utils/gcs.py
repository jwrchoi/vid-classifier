"""
Google Cloud Storage utilities for streaming videos.

Uses Application Default Credentials (ADC) for authentication.
Run `gcloud auth application-default login` to set up credentials.
"""

import streamlit as st
import google.auth
from google.cloud import storage

from config import GCS_BUCKET_NAME


def get_gcs_client():
    """
    Return an authenticated GCS client.

    Strips the ADC quota_project to avoid "User project specified in the
    request is invalid" errors when the default quota project doesn't
    match the bucket's project.

    Returns:
        google.cloud.storage.Client
    """
    credentials, project = google.auth.default()
    credentials = credentials.with_quota_project(None)
    return storage.Client(project=project, credentials=credentials)


def list_video_blobs(bucket_name, prefix, extensions=('.mp4', '.mov', '.avi', '.mkv', '.webm')):
    """
    List video blobs in a GCS bucket under the given prefix.

    Args:
        bucket_name: GCS bucket name
        prefix: Blob prefix (e.g. "videos/")
        extensions: Tuple of valid video file extensions

    Returns:
        List of blob names matching the extensions
    """
    client = get_gcs_client()
    bucket = client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=prefix)

    video_blobs = []
    for blob in blobs:
        if any(blob.name.lower().endswith(ext) for ext in extensions):
            video_blobs.append(blob.name)

    return sorted(video_blobs)


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_video_bytes(bucket_name, blob_name):
    """
    Download a video blob as bytes. Cached for 1 hour.

    Args:
        bucket_name: GCS bucket name
        blob_name: Full blob path (e.g. "videos/video123.mp4")

    Returns:
        bytes: Video file content
    """
    client = get_gcs_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    return blob.download_as_bytes()
