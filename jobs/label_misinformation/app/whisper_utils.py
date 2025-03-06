import shutil
import os
import logging
import modin.pandas as pd
from moviepy import VideoFileClip

WHISPER_COLUMN_NAME = "plaintext_whisper"

# less cost to send audio instead of video to LLM
def transform_mp4_to_mp3(foldername: str):
    for filename in os.listdir(foldername):
        logging.info(f"Reading {foldername}/{filename}...")
        
        f = os.path.join(foldername, filename)
        logging.info(f"Reading {f}...")
        try :
            if not filename.endswith(".mp4"):
                logging.info(f"Skipping non-mp4 file: {filename}")
                continue
            else:
                final_filename = f"{foldername}/{filename.removesuffix(".mp4")}.mp3"
                logging.info("Transforming mp4 to mp3... {final_filename}")
                VideoFileClip(f).audio.write_audiofile(final_filename)
        except Exception as e:
            shutil.copyfile(foldername+filename, foldername[:-1]+"_Audio/"+filename)
            logging.error(e)
            continue
    logging.info(f"Transformed all videos to audio from folder : {foldername}...")


def send_audio_to_api(audio_file) -> str :
    # TODO LLM connection
    return "my new plaintext"