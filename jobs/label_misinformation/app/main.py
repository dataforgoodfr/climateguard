import openai
import logging
import os
import sys
from date_utils import *
from s3_utils import *
from sentry_sdk.crons import monitor
from sentry_utils import *
from logging_utils import *
# In[2]:
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def get_completion(prompt, model):
    openai.api_key = get_secret_docker("OPENAI_API_KEY")
    messages = [{"role": "user", "content": prompt}]
    logging.debug(f"Send {messages}")
    response = openai.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0, 
    )
    return response.choices[0].message.content

def generalfillfile(row, model, column_plaintext) -> int:
    prompt = f"""
    You are an assistant helping editors to moderate TV and radio content.
    You will be provided with a transcript delimited by triple backticks.
    Bear in mind that the transcript may be missing punctuation and may be of very low quality, with incorrect vocabulary, cuts in the wrong places, or may include some phonetic transcription.

    Does the text promote climate change misinformation that undermines well-established scientific consensus, such as denying the existence of climate change or the factors that contribute to it ?

    Answer just a number on a scale of 0 - does not promote - to 10 - certainly promotes. 


    text: '''{row[column_plaintext]}'''
    """
    try:
        response = get_completion(prompt, model)
        logging.info(f"Response API: {response} for text {row[column_plaintext]}")

        return int(response)
    except Exception as e :
        logging.error(f"Error : {e}")
        raise Exception

def get_channels():
    if(os.environ.get("ENV") == "docker" or os.environ.get("CHANNEL") is not None):
        default_channel = os.environ.get("CHANNEL") or "france2"
        logging.warning(f"Only one channel of env var CHANNEL {default_channel} (default to france2) is used")

        channels = [default_channel]
    else: #prod  - all channels
        logging.warning("All channels are used")
        return ["tf1", "france2", "fr3-idf", "m6", "arte", "d8", "bfmtv", "lci", "franceinfotv", "itele",
        "europe1", "france-culture", "france-inter", "sud-radio", "rmc", "rtl", "france24", "france-info", "rfi"]

    return channels

def detect_misinformation(df_news, model_name, min_misinformation_score = 10) -> pd.DataFrame:
    df_news["model_name"] = model_name
    df_news["model_result"] = df_news.apply(
        lambda row: generalfillfile(row, model_name, "plaintext"),
        axis=1  # Apply function to each row
    )
    logging.info(f"model_result Examples : {df_news.head(10)}")
    misinformation_only_news = df_news[df_news["model_result"] >= min_misinformation_score].reset_index(drop=True)
    logging.info("Schema misinformation_only_news :\n%s", misinformation_only_news.dtypes)
    return misinformation_only_news

@monitor(monitor_slug='label-misinformation')
def main():
    logger = getLogger()
    pd.set_option('display.max_columns', None) 
    sentry_init()
    model_name = get_secret_docker("MODEL_NAME")
    app_name = os.getenv("APP_NAME", "")
    date: datetime = set_date(os.getenv("DATE", ""))
    bucket_input = os.getenv("BUCKET_INPUT", "")
    bucket_output = os.getenv("BUCKET_OUTPUT", "")
    min_misinformation_score = int(os.getenv("MIN_MISINFORMATION_SCORE", 10))
    logging.info(f"Starting app {app_name} with model {model_name} for date {date} with bucketinput {bucket_input} and bucket output {bucket_output}, min_misinformation_score to keep is {min_misinformation_score} out of 10...")
    openai_api_key = get_secret_docker("OPENAI_API_KEY")
    openai.api_key = openai_api_key

    channels = get_channels()
    for channel in channels: 
        try:
            s3_client = get_s3_client()
            logging.info(f"saving channel : {channel} inside bucket {bucket_output} folder {app_name}")
            # if the date/channel has already been saved or not 
            if not check_if_object_exists_in_s3(day=date, channel=channel, s3_client=s3_client, bucket=bucket_output, root_folder=app_name):
                df_news = read_folder_from_s3(date=date, channel=channel, bucket=bucket_input)
                logging.info("Schema from API before formatting :\n%s", df_news.dtypes)
                df_news= df_news[['plaintext', 'start', 'channel_title','channel_name', 'channel_program', 'channel_program_type']]

                misinformation_only_news = detect_misinformation(df_news, model_name = model_name, min_misinformation_score = min_misinformation_score)
                number_of_disinformation = len(misinformation_only_news)
            
                if number_of_disinformation > 0:
                    logging.info(f"Misinformation detected {len(misinformation_only_news)} rows")
                    logging.info(f"Examples : {misinformation_only_news.head(10)}")

                    # save TSV format (CSV)
                    save_to_s3(misinformation_only_news, channel=channel,date=date, s3_client=s3_client, \
                            bucket=bucket_output, folder_inside_bucket=app_name)

                    # TODO save using LabelStudio's API
                    # (or use CSV import from S3)
                else:
                    logging.info(f"No misinformation detected for channel {channel} on {date}")
            else:
                logging.info(f"Skipping as already saved before: {channel} inside bucket {bucket_output} folder {app_name}")
        except Exception as err:
            logging.error(f"continuing loop - but met error with {channel} - day {date}: error : {err}")
            continue

    logging.info("Exiting with success")
    sys.exit(0)

if __name__ == "__main__":
    main()