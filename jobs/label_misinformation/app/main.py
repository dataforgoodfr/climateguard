import logging
import os
import sys

import modin.pandas as pd
import ray
from country import Country, CountryCollection, get_countries
from date_utils import get_date_range
from labelstudio_utils import wait_and_sync_label_studio
from logging_utils import getLogger
from mediatree_utils import get_new_plaintext_from_whisper, mediatree_check_secrets
from pg_utils import (
    get_db_session,
    get_keywords_for_a_day_and_channel,
    is_there_data_for_this_day_safe_guard,
)
from pipeline import Pipeline, PipelineInput, SinglePromptPipeline
from s3_utils import check_if_object_exists_in_s3, get_s3_client, save_to_s3
from secret_utils import get_secret_docker
from sentry_sdk.crons import monitor
from sentry_utils import sentry_close, sentry_init
from whisper_utils import get_videofile_mp4_buffer, transform_mp4_to_mp3

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def get_channels(country):
    if os.environ.get("ENV") == "docker" or os.environ.get("CHANNEL") is not None:
        default_channel = os.environ.get("CHANNEL") or "france2"
        logging.warning(
            f"Only one channel of env var CHANNEL {default_channel} (default to france2) is used"
        )

        channels = [default_channel]
    else:  # prod  - all channels
        logging.warning("All channels are used")
        return country.channels

    return channels


def extract_model_result(transcript, pipeline: Pipeline):
    try:
        result = pipeline.process(PipelineInput(transcript=transcript))

        return {"model_result": result.score, "model_reason": result.reason}
    except:
        return {"model_result": 0, "model_reason": "empty"}


def detect_misinformation(
    df_news: pd.DataFrame,
    pipeline: Pipeline,
    min_misinformation_score: int = 10,
    model_name: str = "",
) -> pd.DataFrame:
    """Execute the pipeline on a dataframe and filters on min_score"""
    try:
        df_news["model_result"] = df_news["plaintext"].apply(
            lambda transcript: extract_model_result(transcript, pipeline)
        )
        df_news["model_reason"] = df_news["model_result"].apply(
            lambda x: x["model_reason"]
        )
        df_news["model_result"] = df_news["model_result"].apply(
            lambda x: x["model_result"]
        )
        df_news["model_name"] = model_name
    except Exception as e:
        logging.error(f"Error during apply: {e}")
        raise
    logging.info(f"model_result Examples : {df_news.head(10)}")

    misinformation_only_news = df_news[
        df_news["model_result"] >= min_misinformation_score
    ].reset_index(drop=True)
    logging.info(
        "Schema misinformation_only_news :\n%s", misinformation_only_news.dtypes
    )
    return misinformation_only_news


@monitor(monitor_slug="label-misinformation")
def main(country: Country):
    pd.set_option("display.max_columns", None)
    sentry_init()

    # model_name = get_secret_docker("MODEL_NAME")
    app_name = os.getenv("APP_NAME", "")
    # a security nets in case scaleway servers are done to replay data
    number_of_previous_days = int(os.environ.get("NUMBER_OF_PREVIOUS_DAYS", 7))
    logging.info(
        f"Number of previous days to check for missing date (NUMBER_OF_PREVIOUS_DAYS): {number_of_previous_days}"
    )
    date_env: str = os.getenv("DATE", "")
    bucket_input = os.getenv("BUCKET_INPUT", "")
    # bucket_output = os.getenv("BUCKET_OUTPUT", "")
    bucket_output_folder = os.getenv("BUCKET_OUTPUT_FOLDER", "")
    # country = get_country_or_collection_from_name(os.getenv("COUNTRY", "france"))
    min_misinformation_score = int(os.getenv("MIN_MISINFORMATION_SCORE", 10))

    openai_api_key = get_secret_docker("OPENAI_API_KEY")

    mediatree_check_secrets()

    try:
        bucket_output = country.bucket
        model_name = country.model
        logging.info(
            (f"Starting app {app_name} for country {country.name} "
                f"with model {model_name} for date {date_env} "
                f"with bucketinput {bucket_input} and bucket output {bucket_output}, "
                f"min_misinformation_score to keep is {min_misinformation_score} out of 10...")
        )
        # For the moment the prompt does not change according to the different countries
        # If this changes we need to parametrize the country here
        pipeline = SinglePromptPipeline(
            model_name=model_name, api_key=openai_api_key
        )

        date_range = get_date_range(date_env, minus_days=number_of_previous_days)
        logging.info(
            f"Number of days to query : {len(date_range)} - day_range : {date_range}"
        )
        channels = get_channels(country)
        session = get_db_session()
        for date in date_range:
            was_the_day_processed_in_keywords = (
                is_there_data_for_this_day_safe_guard(
                    session=session, date=date, country=country
                )
            )
            if was_the_day_processed_in_keywords:
                for channel in channels:
                    try:
                        s3_client = get_s3_client()
                        logging.info(
                            f"processing date {date} for channel : {channel} inside bucket {bucket_output} folder {bucket_output_folder}"
                        )
                        # if the date/channel has already been saved or not
                        if check_if_object_exists_in_s3(
                            day=date,
                            channel=channel,
                            s3_client=s3_client,
                            bucket=bucket_output,
                            root_folder=bucket_output_folder,
                            country=country,
                        ):
                            logging.info(
                                f"Skipping as already saved before: {channel} inside bucket {bucket_output} folder {bucket_output_folder}"
                            )
                            continue

                        df_news = get_keywords_for_a_day_and_channel(
                            session=session,
                            date=date,
                            country=country,
                            channel_name=channel,
                        )

                        logging.debug(
                            "Schema from API before formatting :\n%s",
                            df_news.dtypes,
                        )
                        df_news = df_news[
                            [
                                "id",
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
                            logging.warning(
                                f"""Misinformation detected: {len(misinformation_only_news)} rows:
                                {misinformation_only_news.head(10)}
                                """
                            )

                            # ray has problem with tiny dataframes
                            misinformation_only_news = (
                                misinformation_only_news._to_pandas()
                            )

                            df_whispered = get_new_plaintext_from_whisper(
                                misinformation_only_news
                            )

                            # save JSON LabelStudio format
                            save_to_s3(
                                df_whispered,
                                channel=channel,
                                date=date,
                                s3_client=s3_client,
                                bucket=bucket_output,
                                folder_inside_bucket=bucket_output_folder,
                                country=country,
                            )
                        else:
                            logging.info(
                                f"Nothing detected for channel {channel} on {date} - saving a empty file to not re-query it"
                            )
                            save_to_s3(
                                misinformation_only_news,
                                channel=channel,
                                date=date,
                                s3_client=s3_client,
                                bucket=bucket_output,
                                folder_inside_bucket=bucket_output_folder,
                                country=country,
                            )
                    except Exception as err:
                        logging.error(
                            f"continuing loop - but met error with {channel} - day {date}: error : {err}"
                        )
                        continue
            else:
                logging.error(
                    f"This day {date} was not processed by job-mediatree, skipping as it will be tomorrow"
                )
                continue

        logging.info("Exiting with success")
        return True
    except Exception as err:
        logging.fatal("Main crash (%s) %s" % (type(err).__name__, err))
        sys.exit(1)


if __name__ == "__main__":
    ray.init(log_to_driver=True)
    logger = getLogger()
    countries: CountryCollection = get_countries(os.getenv("COUNTRY", "france"))
    for country in countries:
        main(country)
        # sync label studio only if there are new data
        wait_and_sync_label_studio(country.label_studio_id)
        sentry_close()
    sys.exit(0)
