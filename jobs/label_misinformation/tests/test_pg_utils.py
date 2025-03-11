import pytest
import sys, os
sys.path.append(os.path.abspath('/app'))
from app.pg_utils import *
from datetime import datetime
import logging

def save_to_pg(df, table, conn):
    number_of_elements = len(df)
    logging.info(f"Saving {number_of_elements} elements to PG table '{table}'")
    try:
        logging.debug("Schema before saving\n%s", df.dtypes)
        df.to_sql(
            table,
            index=False,
            con=conn,
            if_exists="append",
            chunksize=1000,
            # method=insert_or_do_nothing_on_conflict,  # pandas does not handle conflict natively
            # dtype={"keywords_with_timestamp": JSON, "theme": JSON, "srt": JSON}, # only for keywords
        )
        logging.info("Saved dataframe to PG")
        return len(df)
    except Exception as err:
        logging.error("Could not save : \n %s" % (err))
        return 0

def empty_tables(session = None):
    if(os.environ.get("POSTGRES_HOST") == "postgres_db" or os.environ.get("POSTGRES_HOST") == "localhost"):
        logging.warning("""Doing: Empty table Keywords""")
        session.query(Keywords).delete()
        session.commit()
        logging.warning("""Done: Empty table Keywords""")

def test_get_keywords_for_a_day_and_channel():
    create_tables()
    session = get_db_session()
    empty_tables(session)
    conn = connect_to_db()
    start = pd.to_datetime("2024-12-12 10:10:10")
    channel_name= "itele"
    channel_title= "Cnews"
    primary_key1 = "id1"
    primary_key2 = "id2"
    channel_program= "program"
    channel_program_type= "program_type"

    list = [{
        "id" : primary_key1,
        "start": start,
        "plaintext": "cheese pizza habitabilité de la planète conditions de vie sur terre animal",
        "channel_name": channel_name,
        "channel_title": channel_title,
        "number_of_keywords_climat": 1,
        "channel_program_type": channel_program_type,
        "channel_program": channel_program,
    },
    {
        "id" : primary_key2,
        "start": start,
        "plaintext": "hello world",
        "channel_name": channel_name,
        "channel_title": channel_title,
        "number_of_keywords_climat": 3,
        "channel_program_type": channel_program_type,
        "channel_program": channel_program,
    },
    { # should be ignored as not the right channel_name
        "id" : "id3",
        "start": start,
        "plaintext": "hello world",
        "channel_name": "fake tv news we should ignore",
        "channel_title": channel_title,
        "number_of_keywords_climat": 3,
        "channel_program_type": channel_program_type,
        "channel_program": channel_program,
    }]
    dataframe_to_save = pd.DataFrame(list)
    save_to_pg(dataframe_to_save, table=keywords_table, conn=conn)
    dataframe_to_save.drop(columns=["number_of_keywords_climat"], inplace=True)
    dataframe_to_save.drop(dataframe_to_save.index[-1], inplace=True)
    output = get_keywords_for_a_day_and_channel(session, date=start, channel_name=channel_name)
    
    pd.testing.assert_frame_equal(dataframe_to_save, output,check_like=True)