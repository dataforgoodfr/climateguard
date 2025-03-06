import os
import logging
import urllib.request
import requests
from datetime import datetime
from typing import List
import modin.pandas as pd
from secret_utils import get_secret_docker
from whisper_utils import *
mediatree_password = get_secret_docker("MEDIATREE_PASSWORD")
AUTH_URL : str =  os.environ.get("MEDIATREE_AUTH_URL") 
mediatree_user = get_secret_docker("MEDIATREE_USER")
# https://keywords.mediatree.fr/api/docs/#/paths/~1api~1export~1single/get
API_BASE_URL = os.environ.get("KEYWORDS_URL") 

def get_auth_token(password=mediatree_password, user_name=mediatree_user):
    logging.info(f"Getting a token for user {user_name}")
    try:
        post_arguments = {
            'grant_type': 'password'
            , 'username': user_name
            , 'password': password
        }
        response = requests.post(
            AUTH_URL, 
            data=post_arguments
        )
        output = response.json()
        token = output['data']['access_token']
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
    return  f"{API_BASE_URL}?token={token}&channel={channel_name}&cts_in={start}&cts_out={end}&format=mp4"

def get_video_urls(df: pd.DataFrame) -> List[str]:
    """
    Retrieves video URLs based on position data.

    :param df: DataFrame with columns ["start", "channel_name"]
    :return: List of video URLs
    """
    logging.info(f"Getting videos urls to download files...")
    urls = []
    token = get_auth_token()
    for i in range(len(df)):
        try:
            start, end = get_start_and_end_of_chunk(datetime.fromisoformat(df["start"][i]))
            channel_name = df["channel_name"][i]

            single_export_api = get_mediatree_single_export_url(token, channel_name, start, end)
            logging.info(f"Fetching URL for {channel_name} [{start}-{end}]: {single_export_api}")
            response = requests.get(single_export_api)

            if response.status_code != 200:
                logging.error(f"Failed to fetch URL for {channel_name} [{start}-{end}]: {response.status_code}")
                continue
            output = response.json()
            url = output.get("src", "")
            logging.info(f"Got response url: {url}")
            if not url:
                logging.warning(f"No video URL found for {channel_name} [{start}-{end}]")
                continue
            urls.append(url)

        except Exception as e:
            logging.error(f"Error fetching video URL: {e}")
            urls.append("")
    logging.debug(f"videos urls with {urls}")
    return urls


def download_medias(df: pd.DataFrame, foldername: str):
    """
    Downloads videos from URLs and saves them to a folder.

    :param df: DataFrame with columns ["Start", "Channel Name", "ID"]
    :param foldername: Folder path where videos will be stored
    """
    logging.info(f"Downloading medias from {foldername}...")
    os.makedirs(foldername, exist_ok=True)
    urls = get_video_urls(df)
    output = []
    for i, url in enumerate(urls):
        if not url:
            logging.warning(f"Skipping empty URL {url} for index {i}")
            continue

        file_path = os.path.join(foldername, df["channel_name"][i] + df["start"][i] + url[-4:])
        try:
            logging.info(f"Downloading: {url} -> {file_path}")
            urllib.request.urlretrieve(url, file_path)
            output.append(file_path)

        except Exception as e:
            logging.error(f"Error downloading {url}: {e}")

    return output

def get_new_plaintext_from_whisper(df: pd.DataFrame):
    folder_audio = "output/audio/"
    os.makedirs(os.path.dirname(folder_audio), exist_ok=True)
    videos_url = get_video_urls(df)
    download_medias(df, foldername=folder_audio)
    transform_mp4_to_mp3(folder_audio)

    # for each element inside folder_audio apply new whisperation and apply it to df[WHISPER_COLUMN_NAME]
    # TODO: how to do link between audio files and df row ? start/channel
    # df[WHISPER_COLUMN_NAME] = generalfillfile(pos,0,[],"model_name","whisper")
    return df