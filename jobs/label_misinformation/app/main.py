import openai
import logging
import os
from pipeline import Pipeline, SinglePromptPipeline, PipelineInput, PipelineOutput
import ray
import sys
from date_utils import *
from s3_utils import *
from sentry_sdk.crons import monitor
from sentry_utils import *
from whisper_utils import *
from mediatree_utils import *
from secret_utils import *
from logging_utils import *
import modin.pandas as pd

# In[2]:
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def get_channels():
    if os.environ.get("ENV") == "docker" or os.environ.get("CHANNEL") is not None:
        default_channel = os.environ.get("CHANNEL") or "france2"
        logging.warning(
            f"Only one channel of env var CHANNEL {default_channel} (default to france2) is used"
        )

        channels = [default_channel]
    else:  # prod  - all channels
        logging.warning("All channels are used")
        return [
            "tf1",
            "france2",
            "fr3-idf",
            "m6",
            "arte",
            "bfmtv",
            "lci",
            "franceinfotv",
            "itele",
            "europe1",
            "france-culture",
            "france-inter",
            "sud-radio",
            "rmc",
            "rtl",
            "france24",
            "france-info",
            "rfi",
        ]

    return channels


def extract_model_result(transcript, pipeline: Pipeline):
    try:
        result = pipeline.process(PipelineInput(transcript=transcript))

        return {
                    "model_result": result.score,
                    "model_reason": result.reason
                }
    except:
        return {
            "model_result": 0,
            "model_reason": "empty"
        }

def detect_misinformation(
    df_news: pd.DataFrame, pipeline: Pipeline, min_misinformation_score: int = 10, model_name: str = ""
) -> pd.DataFrame:
    """Execute the pipeline on a dataframe and filters on min_score"""
    try:
        df_news["model_result"] = df_news["plaintext"].apply(
            lambda transcript: extract_model_result(transcript, pipeline)
        )
        df_news['model_reason'] = df_news['model_result'].apply(lambda x: x['model_reason'])
        df_news['model_result'] = df_news['model_result'].apply(lambda x: x['model_result'])
        df_news['model_name'] = model_name
    except Exception as e:
        logging.error(f"Error during apply: {e}")
        raise
    logging.info(f"model_result Examples : {df_news.head(10)}")
    
    misinformation_only_news = df_news[
        df_news["model_result"] >= min_misinformation_score
    ].reset_index(drop=True)
    logging.info("Schema misinformation_only_news :\n%s", misinformation_only_news.dtypes)
    return misinformation_only_news


@monitor(monitor_slug="label-misinformation")
def main():
    logger = getLogger()
    # ray.init()
    ray.init(address='auto', runtime_env={"working_dir": "./"})
    pd.set_option("display.max_columns", None)
    sentry_init()

    model_name = get_secret_docker("MODEL_NAME")
    app_name = os.getenv("APP_NAME", "")
    # a security nets in case scaleway servers are done to replay data
    number_of_previous_days = int(os.environ.get("NUMBER_OF_PREVIOUS_DAYS", 7))
    logging.info(f"Number of previous days to check for missing date (NUMBER_OF_PREVIOUS_DAYS): {number_of_previous_days}")
    date_env: str = os.getenv("DATE", "")
    bucket_input = os.getenv("BUCKET_INPUT", "")
    bucket_output = os.getenv("BUCKET_OUTPUT", "")
    min_misinformation_score = int(os.getenv("MIN_MISINFORMATION_SCORE", 10))
    logging.info(
        f"Starting app {app_name} with model {model_name} for date {date_env} with bucketinput {bucket_input} and bucket output {bucket_output}, min_misinformation_score to keep is {min_misinformation_score} out of 10..."
    )
    openai_api_key = get_secret_docker("OPENAI_API_KEY")

    pipeline = SinglePromptPipeline(model_name=model_name, api_key=openai_api_key)
    try:
        date_range = get_date_range(date_env, minus_days=number_of_previous_days)
        logging.info(f"Number of days to query : {len(date_range)} - day_range : {date_range}")
        channels = get_channels()
        for date in date_range:
            for channel in channels:
                try:
                    s3_client = get_s3_client()
                    logging.info(
                        f"processing date {date} for channel : {channel} inside bucket {bucket_output} folder {app_name}"
                    )
                    # if the date/channel has already been saved or not
                    if check_if_object_exists_in_s3(
                        day=date,
                        channel=channel,
                        s3_client=s3_client,
                        bucket=bucket_output,
                        root_folder=app_name,
                    ):
                        logging.info(
                            f"Skipping as already saved before: {channel} inside bucket {bucket_output} folder {app_name}"
                        )
                        continue

                    df_news = read_folder_from_s3(date=date, channel=channel, bucket=bucket_input)
                    logging.debug("Schema from API before formatting :\n%s", df_news.dtypes)
                    df_news = df_news[
                        [
                            "plaintext",
                            "start",
                            "channel_title",
                            "channel_name",
                            "channel_program",
                            "channel_program_type",
                        ]
                    ]

                    # Run the pipeline on the dataframe
                    misinformation_only_news = detect_misinformation(
                        df_news,
                        pipeline=pipeline,
                        min_misinformation_score=min_misinformation_score,
                        model_name=model_name,
                    )

                    number_of_disinformation = len(misinformation_only_news)

                    if number_of_disinformation > 0:
                        logging.warning(f"Misinformation detected {len(misinformation_only_news)} rows")
                        logging.info(f"Examples : {misinformation_only_news.head(10)}")

                        # improve plaintext from mediatree 
                        logging.info("improve plaintext from mediatree")
                        df_news[WHISPER_COLUMN_NAME] = df_news['plaintext'].apply(lambda x: get_new_plaintext_from_whisper(x['plaintext'])) 

                        # save JSON LabelStudio format
                        save_to_s3(
                            misinformation_only_news,
                            channel=channel,
                            date=date,
                            s3_client=s3_client,
                            bucket=bucket_output,
                            folder_inside_bucket=app_name,
                        )

                        # TODO maybe save using LabelStudio's API
                        # right now, JSON import from S3 are used from Cloud Storage on LabelStudio
                    else:
                        logging.info(
                            f"No misinformation detected for channel {channel} on {date} - saving a empty file to not requery it"
                        )
                        save_to_s3(
                            misinformation_only_news,
                            channel=channel,
                            date=date,
                            s3_client=s3_client,
                            bucket=bucket_output,
                            folder_inside_bucket=app_name,
                        )
                except Exception as err:
                    logging.error(
                        f"continuing loop - but met error with {channel} - day {date}: error : {err}"
                    )
                    continue

        logging.info("Exiting with success")
        sys.exit(0)
    except Exception as err:
        logging.fatal("Main crash (%s) %s" % (type(err).__name__, err))
        sys.exit(1)

if __name__ == "__main__":
    main()
