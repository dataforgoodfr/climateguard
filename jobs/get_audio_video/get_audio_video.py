#!/usr/bin/env python3
"""
Get audio/video script.

Queries labelstudio_task_aggregate table, downloads video from GCS,
extracts a 6-minute segment, and uploads to Google Drive.
"""

import io
import json
import logging
import os
import tempfile
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import requests
from google.cloud import storage
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload
from moviepy.editor import VideoFileClip
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Configuration via env vars / secrets
GCP_CREDENTIALS_JSON_FILE = os.environ.get("GCP_CREDENTIALS_JSON_FILE", "/run/secrets/gcp_credentials_json")
GDRIVE_API_KEY = os.environ.get("GDRIVE_API_KEY_FILE", "/run/secrets/gdrive-api-key")
API_LABEL_STUDIO_KEY = os.environ.get("API_LABEL_STUDIO_KEY", "/run/secrets/gdrive-api-key")
GDRIVE_FOLDER_PATH = os.environ.get("GDRIVE_FOLDER_PATH")  # e.g., "climate/misinformation"
GCS_BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME")
POSTGRES_HOST = os.environ.get("POSTGRES_HOST")
POSTGRES_DB = os.environ.get("POSTGRES_DB", "labelstudio")
POSTGRES_USER = os.environ.get("POSTGRES_USER")
POSTGRES_PASSWORD = os.environ.get("POSTGRES_PASSWORD")
LABEL_STUDIO_URL = os.environ.get("LABEL_STUDIO_URL")


def read_secret(secret_path: str) -> str:
    """Read secret from file, handling Docker secrets."""
    if os.path.exists(secret_path):
        with open(secret_path, "r") as f:
            return f.read().strip()
    # Fallback to env var if file doesn't exist (for local testing)
    return os.environ.get(secret_path)


def get_db_session():
    """Connect to PostgreSQL and return session."""
    url = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}/{POSTGRES_DB}"
    engine = create_engine(url)
    Session = sessionmaker(bind=engine)
    return Session()


def get_gcs_client():
    """Create GCS client from secret file."""
    gcp_creds_json = read_secret(GCP_CREDENTIALS_JSON_FILE)
    if not gcp_creds_json:
        raise ValueError("GCP credentials not found")
    creds_info = json.loads(gcp_creds_json)
    credentials = service_account.Credentials.from_service_account_info(creds_info)
    return storage.Client(credentials=credentials)


def get_gdrive_api_key() -> str:
    """Get Google Drive API key from env var or secret file."""
    api_key = read_secret(GCP_CREDENTIALS_JSON_FILE)
    if not api_key:
        raise ValueError("Google Drive API key not found")
    return api_key


def get_gdrive_service():
    """Build Google Drive service using API key."""
    api_key = get_gdrive_api_key()
    return build("drive", "v3", developerKey=api_key)


def query_labelstudio_tasks(session, limit: int = 1):
    """Query labelstudio_task_aggregate table."""
    query = "SELECT id, data FROM labelstudio_task_aggregate ORDER BY id LIMIT :limit"
    return pd.read_sql(query, session.bind, params={"limit": limit})


def parse_timestamp(data_json: dict) -> datetime:
    """Extract timestamp from data.item.start."""
    return datetime.fromisoformat(data_json["item"]["start"].replace("+00:00", ""))


def floor_to_hour(dt: datetime) -> datetime:
    """Floor datetime to beginning of hour."""
    return dt.replace(minute=0, second=0, microsecond=0)


def download_video_from_gcs(bucket_name: str, filename: str) -> bytes:
    """Download video bytes from GCS."""
    client = get_gcs_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(filename)
    if not blob.exists():
        raise FileNotFoundError(f"Video file not found in GCS: {filename}")
    return blob.download_as_bytes()


def extract_video_segment(video_bytes: bytes, start_time: datetime, end_time: datetime) -> bytes:
    """Extract 6-minute segment from video: 2 min before to 4 min after start_time."""
    # Write bytes to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as f:
        f.write(video_bytes)
        temp_path = f.name

    output_path = None
    try:
        clip = VideoFileClip(temp_path)
        # Calculate subclip times (in seconds)
        t_start = max(0, (start_time - timedelta(minutes=2)).timestamp())
        t_end = (end_time + timedelta(minutes=4)).timestamp()
        segment = clip.subclip(t_start, t_end)

        # Write segment to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as out_f:
            output_path = out_f.name
        segment.write_videofile(output_path, codec="libx264", audio_codec="aac")

        with open(output_path, "rb") as f:
            return f.read()
    finally:
        os.remove(temp_path)
        if output_path and os.path.exists(output_path):
            os.remove(output_path)


def create_or_get_folder(service, name: str, parent_id: Optional[str] = None) -> str:
    """Create folder or get existing folder ID."""
    query = f"name='{name}' and mimeType='application/vnd.google-apps.folder'"
    if parent_id:
        query += f" and '{parent_id}' in parents"
    results = service.files().list(q=query, fields="files(id, name)").execute()
    if results.get("files"):
        return results["files"][0]["id"]

    folder_metadata = {"name": name, "mimeType": "application/vnd.google-apps.folder"}
    if parent_id:
        folder_metadata["parents"] = [parent_id]
    return service.files().create(body=folder_metadata, fields="id").execute()["id"]


def upload_to_drive(service, folder_path: str, filename: str, content: bytes) -> str:
    """Upload file to Google Drive and return share link."""
    # Create or get folder
    folder_parts = folder_path.split("/")
    parent_id = None
    for part in folder_parts:
        parent_id = create_or_get_folder(service, part, parent_id)

    # Upload file
    file_metadata = {
        "name": filename,
        "parents": [parent_id]
    }
    media = MediaIoBaseUpload(io.BytesIO(content), mimetype="video/mp4")
    file = service.files().create(body=file_metadata, media_body=media, fields="id").execute()

    # Make public
    service.permissions().create(
        fileId=file["id"],
        body={"type": "anyone", "role": "reader"}
    ).execute()

    # Get share link
    return f"https://drive.google.com/file/d/{file['id']}/view?usp=sharing"


def update_labelstudio_task(task_id: int, url_mediatree: str) -> None:
    """Update a Labelstudio task with the new URL."""
    if not LABEL_STUDIO_URL or not API_LABEL_STUDIO_KEY:
        logging.warning("LABEL_STUDIO_URL or API_LABEL_STUDIO_KEY not set, skipping Labelstudio update")
        return

    # First, get the current task data
    get_url = f"{LABEL_STUDIO_URL}/api/tasks/{task_id}/"
    headers = {"Authorization": f"Token {API_LABEL_STUDIO_KEY}"}

    response = requests.get(get_url, headers=headers)
    if response.status_code != 200:
        logging.error(f"Failed to get task {task_id}: {response.status_code} {response.text}")
        return

    task_data = response.json()
    data = task_data.get("data", {})

    # Update the data with the new url_mediatree
    data["item"]["url_mediatree"] = url_mediatree

    # Patch the task with updated data
    patch_url = f"{LABEL_STUDIO_URL}/api/tasks/{task_id}/"
    patch_response = requests.patch(patch_url, json={"data": data}, headers=headers)

    if patch_response.status_code not in (200, 201):
        logging.error(f"Failed to update task {task_id}: {patch_response.status_code} {patch_response.text}")
        return

    logging.info(f"Successfully updated task {task_id} with url_mediatree: {url_mediatree}")


def main():
    logging.info("Starting get_audio_video script")

    if not GCS_BUCKET_NAME:
        raise ValueError("GCS_BUCKET_NAME environment variable not set")
    if not GDRIVE_FOLDER_PATH:
        raise ValueError("GDRIVE_FOLDER_PATH environment variable not set")

    session = get_db_session()
    tasks = query_labelstudio_tasks(session, limit=1)

    if tasks.empty:
        logging.warning("No tasks found in labelstudio_task_aggregate")
        return

    for _, row in tasks.iterrows():
        task_id = row["id"]
        data = json.loads(row["data"])
        timestamp = parse_timestamp(data)
        hour_floor = floor_to_hour(timestamp)
        filename = f"recording_{hour_floor.strftime('%Y%m%d%H%M%S')}.mp4"

        logging.info(f"Processing task {task_id}")
        logging.info(f"Timestamp: {timestamp}, Filename: {filename}")

        video_bytes = download_video_from_gcs(GCS_BUCKET_NAME, filename)
        logging.info(f"Downloaded video ({len(video_bytes)} bytes)")

        segment_bytes = extract_video_segment(video_bytes, timestamp, timestamp)
        logging.info(f"Extracted segment ({len(segment_bytes)} bytes)")

        drive_service = get_gdrive_service()
        share_link = upload_to_drive(drive_service, GDRIVE_FOLDER_PATH, filename, segment_bytes)
        logging.info(f"Uploaded to Drive: {share_link}")

        # Update Labelstudio task with the new URL
        update_labelstudio_task(task_id, share_link)

        print(share_link)


if __name__ == "__main__":
    main()