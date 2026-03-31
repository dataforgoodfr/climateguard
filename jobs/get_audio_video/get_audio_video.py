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
import time
from datetime import datetime, timedelta

import pandas as pd
import requests
from google.auth.transport.requests import Request
from google.cloud import storage
from google.oauth2 import service_account
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload
from googleapiclient.http import MediaFileUpload
import pickle

from moviepy import VideoFileClip
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Configuration via env vars / secrets
GCP_CREDENTIALS_JSON_FILE = os.environ.get("GCP_CREDENTIALS_JSON_FILE", "/run/secrets/gcp_credentials_json")
GDRIVE_CREDENTIALS_JSON_FILE = os.environ.get("GDRIVE_CREDENTIALS_JSON_FILE", "/run/secrets/gdrive_credentials_json")
API_LABEL_STUDIO_KEY = os.environ.get("API_LABEL_STUDIO_KEY", "/run/secrets/labelstudio_api_key")
GDRIVE_FOLDER_PATH = os.environ.get("GDRIVE_FOLDER_PATH")  # e.g., "climate/misinformation"
GCS_BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME")
POSTGRES_HOST = os.environ.get("POSTGRES_HOST")
POSTGRES_DB = os.environ.get("POSTGRES_DB", "labelstudio")
POSTGRES_USER = os.environ.get("POSTGRES_USER")
POSTGRES_PASSWORD = os.environ.get("POSTGRES_PASSWORD")
POSTGRES_PORT = os.environ.get("POSTGRES_PORT")
LABEL_STUDIO_URL = os.environ.get("LABEL_STUDIO_URL")
HEALTH_ENDPOINT = f"{LABEL_STUDIO_URL}/health"


def read_secret(secret_path: str) -> str:
    """Read secret from file, handling Docker secrets."""
    if os.path.exists(secret_path):
        with open(secret_path, "r") as f:
            return f.read().strip()
    # Fallback to env var if file doesn't exist (for local testing)
    return os.environ.get(secret_path)


def get_db_session():
    """Connect to PostgreSQL and return session."""
    url = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
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


def authenticate_gdrive() -> Credentials:
    """Authenticate via service account."""
    creds = None

    creds = service_account.Credentials.from_service_account_file(
        GDRIVE_CREDENTIALS_JSON_FILE,
        scopes=["https://www.googleapis.com/auth/drive.file"]
    )
    return creds


def get_gdrive_service():
    """Build Google Drive service using API key."""
    creds = authenticate_gdrive()
    return build("drive", "v3", credentials=creds)


def query_labelstudio_tasks(session, limit: int = 1, country: str = 'germany'):
    """Query labelstudio_task_aggregate table."""
    query = f"""SELECT 
    id, 
    data 
    FROM task
    where id=7603 
    and (data::jsonb #>> ARRAY['item', 'start'])::timestamp >= '2025-09-24'
    and (data::jsonb #>> ARRAY['item', 'channel_name'])::TEXT in ('daserste', 'zdf')
    ORDER BY id LIMIT {int(limit)}"""
    # query = f"""SELECT 
    # id, 
    # data 
    # FROM labelstudio_task_aggregate
    # WHERE country='{country}'
    # and id=6875
    # and (data::jsonb #>> ARRAY['item', 'channel_title'])::TEXT in ('daserste', 'zdf')
    # ORDER BY id LIMIT {int(limit)}"""
    return pd.read_sql(query, session.bind)


def parse_timestamp(data_json: dict) -> datetime:
    """Extract timestamp from data.item.start."""
    return datetime.fromisoformat(data_json["item"]["start"].replace("+00:00", ""))


def floor_to_2hours(dt: datetime) -> datetime:
    """Floor datetime to beginning of nearest 2-hour chunk (0, 2, 4, ..., 22)."""
    return dt.replace(minute=0, second=0, microsecond=0, hour=(dt.hour // 2) * 2)


def download_video_from_gcs(bucket_name: str, filename: str, channel_title: str) -> bytes:
    """Download video bytes from GCS, trying both 01 and 02 ms suffixes."""
    client = get_gcs_client()
    bucket = client.bucket(bucket_name)

    # Try both 01 and 02 ms suffixes
    for support in ("Videos", "Audios"):
        for ms_suffix in ["01", "02"]:
            candidate = filename.replace("00.mp4", f"{ms_suffix}.mp4")
            if support == 'Audios':
                candidate = candidate.replace(".mp4", ".mp3")
            candidate = os.path.join(
                channel_title,
                support,
                candidate
            )
            logging.info(f"Trying candidate {candidate}")
            blob = bucket.blob(candidate)
            if blob.exists():
                logging.info(f"Found video file: {candidate}")
                return blob.download_as_bytes()
            logging.debug(f"Video file not found: {candidate}")

    raise FileNotFoundError(f"Video file not found in GCS for base {filename} (tried 01/02 ms suffixes)")


def extract_video_segment(video_bytes: bytes, timestamp: datetime, video_start_time: datetime) -> bytes:
    """Extract 6-minute segment from video: 2 min before to 4 min after timestamp.

    Args:
        video_bytes: Raw video file bytes
        timestamp: The absolute timestamp of the event
        video_start_time: The start time of the video (from filename)
    """
    # Write bytes to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as f:
        f.write(video_bytes)
        temp_path = f.name

    output_path = None
    try:
        clip = VideoFileClip(temp_path)
        # Calculate relative offset from video start (in seconds)
        offset_seconds = (timestamp - video_start_time).total_seconds()
        # 2 min before to 4 min after the event
        t_start = max(0, offset_seconds - 120)  # 2 minutes before
        t_end = offset_seconds + 240  # 4 minutes after
        segment = clip.subclipped(t_start, t_end)

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


def upload_to_drive(service, folder_id: str, filename: str, content: bytes) -> str:
    """Upload file to Google Drive folder and return share link."""
    # Upload file
    file_metadata = {
        "name": filename,
        "parents": [folder_id]
    }
    media = MediaIoBaseUpload(io.BytesIO(content), mimetype="video/mp4")
    file = service.files().create(body=file_metadata, media_body=media, fields="id", supportsAllDrives=True).execute()
    logging.info(file)
    # Make public
    service.permissions().create(
        fileId=file["id"],
        body={"type": "anyone", "role": "reader"},
        supportsAllDrives=True,
    ).execute()

    # Get share link
    return f"https://drive.google.com/file/d/{file['id']}/view?usp=sharing"

def pretty_print_POST(req):
    """
    At this point it is completely built and ready
    to be fired; it is "prepared".

    However pay attention at the formatting used in 
    this function because it is programmed to be pretty 
    printed and may differ from the actual request.
    """
    print('{}\n{}\r\n{}\r\n\r'.format(
        '-----------START-----------',
        req.method + ' ' + req.url,
        '\r\n'.join('{}: {}'.format(k, v) for k, v in req.headers.items()),
    ))

def update_labelstudio_task(task_id: int, url_mediatree: str) -> None:
    """Update a Labelstudio task with the new URL."""
    if not LABEL_STUDIO_URL or not API_LABEL_STUDIO_KEY:
        logging.warning("LABEL_STUDIO_URL or API_LABEL_STUDIO_KEY not set, skipping Labelstudio update")
        return

    # First, get the current task data
    get_url = f"{LABEL_STUDIO_URL.rstrip('/')}/api/tasks/{task_id}/"
    headers = {"Authorization": f"Token {read_secret(API_LABEL_STUDIO_KEY)}"}

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


def wait_for_health(url, timeout=240, interval=5):
    """Waits until the given URL returns a 200 status or timeout is reached."""
    logging.info("Waiting for the service to become healthy...")
    start_time = time.time()

    while time.time() - start_time < timeout:
        try:
            logging.info(f"Sending get to {url}")
            headers = {"Content-Type": "application/json"}
            response = requests.get(url, timeout=5, headers=headers)
            logging.info(f"Response {response.content}")
            if response.status_code == 200:
                logging.info("Label Studio Service is healthy!")
                return True
        except requests.RequestException as e:
            logging.info(f"Health check failed: {e}")

        logging.info(f"Label Studio Service not ready yet {url}. Retrying...")
        time.sleep(interval)

    logging.warning("Timed out waiting for the service to become healthy.")
    return False


def main():
    logging.info("Starting get_audio_video script")

    if not GCS_BUCKET_NAME:
        raise ValueError("GCS_BUCKET_NAME environment variable not set")
    if not GDRIVE_FOLDER_PATH:
        raise ValueError("GDRIVE_FOLDER_PATH environment variable not set")

    session = get_db_session()
    tasks = query_labelstudio_tasks(session, limit=100)

    if tasks.empty:
        logging.warning("No tasks found")
        return

    # Parse tasks and build download map: (filename, channel) -> list of task info
    task_info_list = []
    download_map = {}  # key: (filename, channel) -> video_bytes

    for _, row in tasks.iterrows():
        task_id = row["id"]
        data = row["data"]
        channel_title = data["item"]["channel_title"].replace(" ", "")
        timestamp = parse_timestamp(data)
        hour_floor = floor_to_2hours(timestamp)
        filename = f"recording_{hour_floor.strftime('%Y%m%d%H%M%S')}.mp4"

        task_info = {
            "task_id": task_id,
            "data": data,
            "channel_title": channel_title,
            "timestamp": timestamp,
            "filename": filename,
        }
        task_info_list.append(task_info)

        # Track unique (filename, channel) pairs for batch download
        key = (filename, channel_title)
        if key not in download_map:
            download_map[key] = None  # Will be filled with video_bytes

    # Download all unique videos
    logging.info(f"Downloading {len(download_map)} unique video chunks for {len(task_info_list)} tasks")
    for (filename, channel_title) in download_map.keys():
        try:
            video_bytes = download_video_from_gcs(GCS_BUCKET_NAME, filename, channel_title)
            download_map[(filename, channel_title)] = video_bytes
            logging.info(f"Downloaded {filename} for channel {channel_title} ({len(video_bytes)} bytes)")
        except FileNotFoundError as e:
            logging.error(f"Failed to download {filename} for channel {channel_title}: {e}")
            download_map[(filename, channel_title)] = None

    # Process each task using already-downloaded videos
    drive_service = get_gdrive_service()
    for task_info in task_info_list:
        task_id = task_info["task_id"]
        timestamp = task_info["timestamp"]
        filename = task_info["filename"]
        channel_title = task_info["channel_title"]
        data = task_info["data"]

        key = (filename, channel_title)
        video_bytes = download_map.get(key)

        if video_bytes is None:
            logging.error(f"Skipping task {task_id} - video not available")
            continue

        logging.info(f"Processing task {task_id}")
        logging.info(f"Timestamp: {timestamp}, Filename: {filename}")

        # Parse video start time from filename (format: recording_YYYYMMDDHHMMSS.mp4)
        ts_str = filename.replace("recording_", "").replace(".mp4", "").replace(".mp3", "")
        # Handle 01/02 ms suffix - take first 14 chars (YYYYMMDDHHMMSS)
        ts_str = ts_str[:14]
        video_start_time = datetime.strptime(ts_str, "%Y%m%d%H%M%S")

        segment_bytes = extract_video_segment(video_bytes, timestamp, video_start_time)
        logging.info(f"Extracted segment ({len(segment_bytes)} bytes)")

        share_link = upload_to_drive(drive_service, GDRIVE_FOLDER_PATH, filename, segment_bytes)
        logging.info(f"Uploaded to Drive: {share_link}")

        # Update Labelstudio task with the new URL
        if wait_for_health(HEALTH_ENDPOINT):
            update_labelstudio_task(task_id, share_link)

        print(share_link)


if __name__ == "__main__":
    main()