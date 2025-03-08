from mediatree_utils import get_url_mediatree
from whisper_utils import WHISPER_COLUMN_NAME
import modin.pandas as pd
import base64

# @see https://github.com/HumanSignal/label-studio/issues/1492#issuecomment-924522609   
def encode_audio_base64(audio_bytes):
    return f"data:audio/mp3;base64,{base64.b64encode(audio_bytes).decode('utf-8')}"

def get_label_studio_format(row) -> str:
    url_mediatree = url_mediatree = get_url_mediatree(row["start"], row["channel_name"])
    start_time = (
        row["start"].isoformat() if isinstance(row["start"], pd.Timestamp) else row["start"]
    )

    # TODO make it labelstudio compatible
    # media = encode_audio_base64(row['media'])

    task_data = {
        "data": {
            "item": {
                "plaintext": row["plaintext"],
                WHISPER_COLUMN_NAME: row[WHISPER_COLUMN_NAME],
                # "media": media,  # TODO make it labelstudio compatible
                "start": start_time,
                "channel_title": row["channel_title"],
                "channel_name": row["channel_name"],
                "channel_program": row["channel_program"],
                "channel_program_type": row["channel_program_type"],
                "model_name": row["model_name"],
                "model_result": row["model_result"],
                "model_reason": row["model_reason"],
                "year": row["year"],
                "month": row["month"],
                "day": row["day"],
                "channel": row["channel"],
                "url_mediatree": url_mediatree,
            }
        },
        "annotations": [],
        "predictions": [],
    }

    return task_data
