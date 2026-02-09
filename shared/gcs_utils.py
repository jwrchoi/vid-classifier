"""
Google Cloud Storage utilities shared across the monorepo.
==========================================================

Provides authenticated GCS access for downloading videos,
listing blobs, and uploading feature outputs.

Authentication uses Application Default Credentials (ADC).
Run `gcloud auth application-default login` to set up credentials.
"""

import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import google.auth
from google.cloud import storage

from .config import GCS_BUCKET_NAME, SUPPORTED_VIDEO_EXTENSIONS


def get_gcs_client() -> storage.Client:
    """
    Return an authenticated GCS client.

    Strips the ADC quota_project to avoid "User project specified in the
    request is invalid" errors when the default quota project doesn't
    match the bucket's project.
    """
    credentials, project = google.auth.default()
    credentials = credentials.with_quota_project(None)
    return storage.Client(project=project, credentials=credentials)


def list_video_blobs(
    bucket_name: str = GCS_BUCKET_NAME,
    prefix: str = "videos/",
    extensions: Tuple[str, ...] = tuple(SUPPORTED_VIDEO_EXTENSIONS),
) -> List[str]:
    """
    List video blob names in a GCS bucket under the given prefix.

    Args:
        bucket_name: GCS bucket name.
        prefix: Blob prefix (folder path), e.g. "videos/".
        extensions: Tuple of valid video file extensions.

    Returns:
        Sorted list of blob names matching the extensions.
    """
    client = get_gcs_client()
    bucket = client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=prefix)

    video_blobs = [
        blob.name
        for blob in blobs
        if any(blob.name.lower().endswith(ext) for ext in extensions)
    ]
    return sorted(video_blobs)


def download_video_to_temp(
    blob_name: str,
    bucket_name: str = GCS_BUCKET_NAME,
) -> Path:
    """
    Download a video blob to a local temp file and return the path.

    The caller is responsible for cleaning up the temp file when done.
    Useful for feature extraction pipelines that need a local file path
    (e.g. OpenCV, PySceneDetect).

    Args:
        blob_name: Full blob path, e.g. "videos/video123.mp4".
        bucket_name: GCS bucket name.

    Returns:
        Path to the downloaded temp file.
    """
    client = get_gcs_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    # Preserve the original file extension so OpenCV can detect the codec.
    suffix = Path(blob_name).suffix or ".mp4"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    blob.download_to_filename(tmp.name)
    tmp.close()
    return Path(tmp.name)


def upload_file_to_gcs(
    local_path: Path,
    destination_blob_name: str,
    bucket_name: str = GCS_BUCKET_NAME,
) -> str:
    """
    Upload a local file to GCS.

    Args:
        local_path: Path to the local file to upload.
        destination_blob_name: Target blob name in the bucket.
        bucket_name: GCS bucket name.

    Returns:
        The gs:// URI of the uploaded blob.
    """
    client = get_gcs_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(str(local_path))
    return f"gs://{bucket_name}/{destination_blob_name}"
