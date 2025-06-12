import logging
import json
import os
import sys
from datetime import datetime

import modin.pandas as pd
import ray
from country import Country, CountryCollection, get_countries
from date_utils import get_date_range
from labelstudio_utils import edit_labelstudio_record_data, wait_and_sync_label_studio
from logging_utils import getLogger
from mediatree_utils import get_new_plaintext_from_whisper, mediatree_check_secrets
from pg_utils import (
    connect_to_db,
    get_db_session,
    get_keywords_for_period_and_channels,
    get_labelstudio_records_period,
)
from pipeline import Pipeline, PipelineInput, BertPipeline
from s3_utils import get_s3_client, save_to_s3
from secret_utils import get_secret_docker
from sentry_sdk.crons import monitor
from sentry_utils import sentry_close, sentry_init
from whisper_utils import WHISPER_COLUMN_NAME

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
        df_news["prompt_version"] = pipeline.prompt_version
        df_news["pipeline_version"] = pipeline.version
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

    app_name = os.getenv("APP_NAME", "")
    # a security nets in case scaleway servers are done to replay data
    number_of_previous_days = int(os.environ.get("NUMBER_OF_PREVIOUS_DAYS", 7))
    logging.info(
        f"Number of previous days to check for missing date (NUMBER_OF_PREVIOUS_DAYS): {number_of_previous_days}"
    )
    date_env: str = os.getenv("DATE", "")
    bucket_output_folder = os.getenv("BUCKET_OUTPUT_FOLDER", "")
    min_misinformation_score = int(os.getenv("MIN_MISINFORMATION_SCORE", 10))

    openai_api_key = get_secret_docker("OPENAI_API_KEY")

    mediatree_check_secrets()

    try:
        s3_client = get_s3_client()
        bucket_output = os.getenv("BUCKET_OUTPUT_SHADOW", "")
        model_name = os.getenv("SHADOW_MODEL", "")
        logging.info(
            (
                f"Starting app {app_name} for country {country.name} "
                f"with model {model_name} for date {date_env} "
                f"with bucket output {bucket_output}, "
                f"min_misinformation_score to keep is {min_misinformation_score} out of 10..."
            )
        )
        # For the moment the prompt does not change according to the different countries
        # If this changes we need to parametrize the country here

        pipeline = BertPipeline(
            model_name=model_name,
            tokenizer_name=model_name,
            chunk_size=512,
            chunk_overlap=256,
            batch_size=32,
        )

        date_range = get_date_range(date_env, minus_days=number_of_previous_days)
        logging.info(
            f"Number of days to query : {len(date_range)} - day_range : {date_range}"
        )
        channels = get_channels(country)
        session = get_db_session()
        labelstudio_db_session = get_db_session(
            engine=connect_to_db(
                db_database=os.environ.get("POSTGRES_DB_LS", "labelstudio")
            )
        )

        labelstudio_df = get_labelstudio_records_period(
            labelstudio_db_session,
            date_range.min(),
            date_range.max(),
            channels,
            country,
        )
        labelstudio_df.index = (
            labelstudio_df
            .apply(
                lambda row: json.loads(row["data"])["item"]["id"],
                axis=1,
            )
            .str
            .replace('"', '')
        )
        keywords_df = get_keywords_for_period_and_channels(
            session,
            date_range.min(),
            date_range.max(),
            channels,
            country,
        )
        data_input_batch = [PipelineInput(id=row["id"], transcript=row["plaintext"]) for idx, row in keywords_df.iterrows()]
        shadow_pipeline_outputs = pipeline.batch_process(data_input_batch)
        output_df = pd.DataFrame(
            data=[[output.id, output.score, output.probability] for output in shadow_pipeline_outputs],
            columns=["id", "shadow_model_result", "probability"]
        )
        output_df["shadow_model_name"] = model_name
        output_df["shadow_prompt_version"] = ""
        output_df["shadow_pipeline_version"] = pipeline.version
        output_df["shadow_model_reason"] = output_df.probability


        # adding shadow_model_result and probability to merged_df
        # Columns: id, start, channel_program, channel_program_type, channel_title, channel_name, plaintext, country, shadow_model_result, probability
        merged_df = keywords_df.merge(
            output_df,
            left_on="id",
            right_on="id"
        )
        merged_df["date"] = pd.to_datetime(merged_df.start)
        merged_df["year"] = merged_df["date"].dt.year
        merged_df["month"] = merged_df["date"].dt.month
        merged_df["day"] = merged_df["date"].dt.day

        new_records = merged_df.loc[(merged_df["shadow_model_result"] == 1) & (~merged_df.id.isin(labelstudio_df.index))]
        records_labelstudio = merged_df.loc[(merged_df["shadow_model_result"] == 1) & (merged_df.id.isin(labelstudio_df.index))]
        
        # add whisper to data that is not present in labelstudio
        new_records = get_new_plaintext_from_whisper(
            new_records
        )
        new_records = new_records.dropna(subset=[WHISPER_COLUMN_NAME])
        new_records["model_result"] = 0
        new_records["model_reason"] = ""
        new_records["model_name"] = country.model
        new_records["prompt_version"] = country.prompt_version
        new_records["pipeline_version"] = f"{country.model}/{country.prompt_version}"

        # Saving records that are not in labelstudio and shadow_model_result is 1
        groups = new_records.groupby(["country", "year", "month", "day", "channel"])
        for columns, group in groups:
            # How do we inject the new results in the labelstudio record ?
            save_to_s3(
                group,
                channel=columns[4],
                date=datetime(columns[1], columns[2], columns[3]),
                s3_client=s3_client,
                bucket=bucket_output,
                folder_inside_bucket=bucket_output_folder,
                country=country,
                shadow=True,
            )

        # Saving shadow model scores for all annotations in labelstudio
        records_labelstudio_dict = records_labelstudio.to_dict(orient="records")
        for labelstudio_id, row in labelstudio_df.iterrows():
            shadow_result = records_labelstudio_dict[labelstudio_id]
            updated_record = edit_labelstudio_record_data(row, shadow_result)

            key = (
                f"country={country.name}/"
                f"year={updated_record['year']}/"
                f"month={updated_record['month']:1}/"
                f"day={updated_record['day']:1}/"
                f"channel={updated_record['channel_name']}/"
            )

            s3_client.put_object(
                Body=json.dumps(updated_record),
                Bucket=bucket_output,
                Key=key,
            )

        logging.info("Exiting with success")
        return True
    except Exception as err:
        logging.fatal("Main crash (%s) %s" % (type(err).__name__, err))
        sys.exit(1)


if __name__ == "__main__":
    ray.init(log_to_driver=True)
    logger = getLogger()
    countries: CountryCollection = get_countries(os.getenv("COUNTRY", "france"))
    model_name = os.getenv("SHADOW_LABELSTUDIO_ID", "")
    for country in countries:
        main(country)
        # sync label studio only if there are new data
        wait_and_sync_label_studio(model_name)

    sentry_close()  # monitoring
    sys.exit(0)
