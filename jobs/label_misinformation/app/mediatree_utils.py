import os
import logging
import urllib.request
import requests
from datetime import datetime
from typing import List
import modin.pandas as pd
from secret_utils import get_secret_docker
from whisper_utils import transform_mp4_to_mp3
from typing import Optional
import ray

mediatree_password = get_secret_docker("MEDIATREE_PASSWORD")
AUTH_URL: str = os.environ.get("MEDIATREE_AUTH_URL")
mediatree_user = get_secret_docker("MEDIATREE_USER")
# https://keywords.mediatree.fr/api/docs/#/paths/~1api~1export~1single/get
API_BASE_URL = os.environ.get("KEYWORDS_URL")


def get_auth_token(password=mediatree_password, user_name=mediatree_user):
    logging.info(f"Getting a token")
    try:
        post_arguments = {"grant_type": "password", "username": user_name, "password": password}
        response = requests.post(AUTH_URL, data=post_arguments)
        output = response.json()
        token = output["data"]["access_token"]
        return token
    except Exception as err:
        logging.error("Could not get token %s:(%s) %s" % (type(err).__name__, err))


def get_start_and_end_of_chunk(start):
    two_minutes = 120
    timestamp = str(int(start.timestamp()))
    timestamp_end = str(int(start.timestamp() + two_minutes))

    return timestamp, timestamp_end


def get_url_mediatree(date, channel) -> str:
    # https://keywords.mediatree.fr/player/?fifo=france-inter&start_cts=1729447079&end_cts=1729447201&position_cts=1729447080
    timestamp, timestamp_end = get_start_and_end_of_chunk(date)
    return f"https://keywords.mediatree.fr/player/?fifo={channel}&start_cts={timestamp}&end_cts={timestamp_end}&position_cts={timestamp}"


def get_mediatree_single_export_url(token, channel_name, start, end):
    return f"{API_BASE_URL}?token={token}&channel={channel_name}&cts_in={start}&cts_out={end}&format=mp4"


# to mock tests
def get_response_single_export_api(single_export_api):
    return requests.get(single_export_api)


def fetch_video_url(row, token):
    """Fetches a single video URL based on a DataFrame row."""
    try:
        start, end = get_start_and_end_of_chunk(datetime.fromisoformat(row["start"]))
        channel_name = row["channel_name"]
        logging.info(f"Fetching URL for {channel_name} {start} {end}...")
        logging.error(f"Token : {token}")
        single_export_api = get_mediatree_single_export_url(token, channel_name, start, end)
        logging.info(f"Fetching URL for {channel_name} [{start}-{end}]: {single_export_api}")

        response = get_response_single_export_api(single_export_api)
        if response.status_code != 200:
            logging.error(
                f"Failed to fetch URL for {channel_name} [{start}-{end}]: {response.status_code}"
            )
            return None  # Keep column as None for failed cases

        output = response.json()
        url = output.get("src", "")

        logging.info(f"Got response URL: {url}")

        if not url:
            logging.warning(f"No video URL found for {channel_name} [{start}-{end}]")
            return None
        logging.debug(f"url is {url}")
        return url

    except Exception as e:
        logging.error(f"Error fetching video URL: {e}")
        return None


def get_video_urls(df: pd.DataFrame) -> pd.DataFrame:
    """
    Retrieves video URLs and adds them directly to the DataFrame using apply().

    :param df: DataFrame with columns ["start", "channel_name"]
    :return: Updated DataFrame with a new "media_url" column
    """
    logging.info("Fetching video URLs for downloading...")
    token = get_auth_token()
    logging.info("got token")

    logging.info(f"{df["channel_name"].head()}")
    df["media_url"] = df.apply(lambda row: fetch_video_url(row, token), axis=1)

    logging.debug(
        f"Updated DataFrame with media URLs:\n{df[['channel_name', 'start', 'media_url']].head()}"
    )
    return df


def download_media(row) -> Optional[bytes]:
    """Downloads a media file and returns its binary content."""
    url = row["media_url"]
    if not url:
        logging.warning(f"Skipping empty URL for {row['channel_name']} at {row['start']}")
        return None

    try:
        logging.info(f"Downloading: {url}")
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an error for bad responses (4xx, 5xx)

        media: bytes = response.content
        if url.endswith(".mp4"):
            media = transform_mp4_to_mp3(media)
            return media
        else:
            return media  # Return binary MP3 data

    except Exception as e:
        logging.error(f"Error downloading {url}: {e}")
        return None


def add_medias_to_df(df: pd.DataFrame):
    """
    Downloads videos from URLs and saves them to dataframe
    """
    logging.info(f"Downloading medias..")

    # add  "media_url" column
    df = get_video_urls(df)
    df["media"] = df.apply(lambda row: download_media(row), axis=1)

    logging.debug(
        f"Updated DataFrame with media files:\n{df[['channel_name', 'start', 'media_url', 'media']].head()}"
    )
    return df


def get_new_plaintext_from_whisper(df: pd.DataFrame):
    df = add_medias_to_df(df)
    # for each element inside df apply new whisperation with column "media" and apply it to df[WHISPER_COLUMN_NAME]
    # TODO:
    # df[WHISPER_COLUMN_NAME] =  df.apply(lambda row: whisper(row), axis=1) # generalfillfile(pos,0,[],"model_name","whisper")
    return df
