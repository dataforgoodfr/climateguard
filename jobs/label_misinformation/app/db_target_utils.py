import hashlib
import logging
from datetime import datetime

import modin.pandas as pd
from country import Country, FRANCE_COUNTRY
from labelstudio_utils import get_label_studio_format
from pg_utils import (
    LabelStudioTaskAggregate,
    get_next_task_aggregate_id,
    task_global_completion,
)
from sqlalchemy import delete, insert
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session
from whisper_utils import WHISPER_COLUMN_NAME


def get_consistent_hash(my_pk) -> str:
    obj_str = str(my_pk)
    sha256 = hashlib.sha256()
    sha256.update(obj_str.encode("utf-8"))
    return sha256.hexdigest()


def create_hash_id(
    df: pd.DataFrame, column_name: str, id_column: str = "id", position: int = 0
) -> pd.DataFrame:
    """Insert a deterministic hash column built from id_column + project_id + country."""
    df.insert(
        position,
        column_name,
        (df[id_column].astype(str) + df.project_id.astype(str) + df.country).apply(
            get_consistent_hash
        ),
    )
    return df


def save_to_db(
    df: pd.DataFrame,
    channel: str,
    date: pd.Timestamp,
    session: Session,
    country: Country = FRANCE_COUNTRY,
) -> None:
    """Write misinformation rows to the labelstudio_task_aggregate target table."""
    logging.info(
        f"Saving DF with {len(df)} elements to target DB for {date}, country {country.name} and channel {channel}"
    )

    if len(df) == 0:
        logging.info("Nothing to save to target DB")
        return

    df = df.copy()
    df["year"] = date.year
    df["month"] = date.month
    df["day"] = date.day
    df["channel"] = channel

    # snapshot the labelstudio-formatted payload before mutating id/country below
    df["item"] = df.apply(lambda row: get_label_studio_format(row)["data"]["item"], axis=1)
    df["data"] = df["item"].apply(lambda item: {"item": item})

    df["original_id"] = df["id"]
    df["project_id"] = int(country.label_studio_project)
    df["country"] = country.name
    df = create_hash_id(df, "task_aggregate_id", id_column="original_id")

    next_id = get_next_task_aggregate_id(session)
    df["id"] = range(next_id, next_id + len(df))
    df["inner_id"] = df["id"]

    now = datetime.utcnow()
    aggregate_rows = [
        {
            "task_aggregate_id": row["task_aggregate_id"],
            "id": int(row["id"]),
            "data": row["data"],
            "created_at": now,
            "updated_at": now,
            "is_labeled": False,
            "project_id": row["project_id"],
            "meta": None,
            "overlap": 1,
            "file_upload_id": None,
            "updated_by_id": None,
            "inner_id": int(row["inner_id"]),
            "total_annotations": 0,
            "cancelled_annotations": 0,
            "total_predictions": 0,
            "comment_count": 0,
            "last_comment_updated_at": None,
            "unresolved_comment_count": 0,
            "country": row["country"],
        }
        for _, row in df.iterrows()
    ]

    completion_rows = [
        {
            # not yet annotated: filled in downstream once a labelstudio annotation lands
            "task_completion_aggregate_id": None,
            "task_aggregate_id": row["task_aggregate_id"],
            "task_id": int(row["id"]),
            "created_at": now,
            "updated_at": now,
            "is_labeled": False,
            "project_id": row["project_id"],
            "country": row["country"],
            "data_item_id": row["item"]["id"],
            "data_item_channel": row["item"]["channel"],
            "data_item_channel_name": row["item"]["channel_name"],
            "data_item_channel_title": row["item"]["channel_title"],
            "data_item_channel_program": row["item"]["channel_program"],
            "data_item_channel_program_type": row["item"]["channel_program_type"],
            "data_item_day": row["day"],
            "data_item_month": row["month"],
            "data_item_year": row["year"],
            "data_item_start": row["start"],
            "data_item_model_name": row["item"]["model_name"],
            "data_item_model_reason": row["item"]["model_reason"],
            "data_item_model_result": row["item"]["model_result"],
            "data_item_plaintext": row["item"]["plaintext"],
            "data_item_plaintext_whisper": row["item"].get(WHISPER_COLUMN_NAME),
            "data_item_url_mediatree": row["item"]["url_mediatree"],
        }
        for _, row in df.iterrows()
    ]

    try:
        statement = pg_insert(LabelStudioTaskAggregate).values(aggregate_rows)
        statement = statement.on_conflict_do_nothing(
            index_elements=[LabelStudioTaskAggregate.task_aggregate_id]
        )
        session.execute(statement)

        # task_global_completion has no unique constraint of its own to rely on
        # ON CONFLICT, so guard against duplicates on rerun by keying off the
        # same deterministic task_aggregate_id hash: drop any existing rows for
        # this batch's ids first, then insert fresh.
        task_aggregate_ids = [row["task_aggregate_id"] for row in completion_rows]
        session.execute(
            delete(task_global_completion).where(
                task_global_completion.c.task_aggregate_id.in_(task_aggregate_ids)
            )
        )
        session.execute(insert(task_global_completion).values(completion_rows))
        session.commit()
        logging.info(
            f"Saved {len(aggregate_rows)} rows to target DB tables "
            "labelstudio_task_aggregate and analytics.task_global_completion"
        )
    except Exception as err:
        session.rollback()
        logging.fatal("save_to_db (%s) %s" % (type(err).__name__, err))
        raise
