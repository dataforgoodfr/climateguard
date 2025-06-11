import json
import logging
import os
from datetime import datetime, timedelta
from typing import Union, List

import modin.pandas as pd
from country import (
    ALL_COUNTRIES,
    BELGIUM_COUNTRY,
    BRAZIL_COUNTRY,
    FRANCE_COUNTRY,
    LEGACY_COUNTRIES,
    Country,
    CountryCollection,
    get_country_or_collection_from_name,
)
from sqlalchemy import (
    ARRAY,
    JSON,
    URL,
    BigInteger,
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    and_,
    cast,
    create_engine,
    func,
    literal_column,
    or_,
    select,
    text,
)
from sqlalchemy.orm import Session, declarative_base, sessionmaker

Base = declarative_base()
BaseLS = declarative_base()

keywords_table = "keywords"
labelstudio_task_table = "task"


class Keywords(Base):
    __tablename__ = keywords_table

    id = Column(Text, primary_key=True)
    channel_name = Column(String, nullable=False)
    channel_title = Column(String, nullable=True)
    channel_program = Column(String, nullable=True)  #  arcom - alembic handles this
    channel_program_type = Column(
        String, nullable=True
    )  # arcom - (magazine, journal etc) alembic handles this
    channel_radio = Column(Boolean, nullable=True)
    start = Column(DateTime())
    plaintext = Column(Text)
    theme = Column(
        JSON
    )  # keyword.py  # ALTER TABLE keywords ALTER theme TYPE json USING to_json(theme);
    created_at = Column(
        DateTime(timezone=True), server_default=text("(now() at time zone 'utc')")
    )  # ALTER TABLE ONLY keywords ALTER COLUMN created_at SET DEFAULT (now() at time zone 'utc');
    updated_at = Column(
        DateTime(),
        default=datetime.now,
        onupdate=text("now() at time zone 'Europe/Paris'"),
        nullable=True,
    )
    keywords_with_timestamp = Column(
        JSON
    )  # ALTER TABLE keywords ADD keywords_with_timestamp json;
    number_of_keywords_climat = Column(
        Integer
    )  # sum of all climatique counters without duplicate (like number_of_keywords)
    number_of_keywords = Column(
        Integer
    )  # sum of all climatique counters without duplicate (like number_of_keywords)
    country = Column(Text)


class LabelStudioTask(BaseLS):
    __tablename__ = labelstudio_task_table
    # column_name,data_type,character_maximum_length,column_default,is_nullable
    id = Column(Integer, nullable=False, primary_key=True)
    data = Column(JSON, nullable=False)
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)
    is_labeled = Column(Boolean, nullable=False)
    project_id = Column(Integer, nullable=True)
    meta = Column(JSON, nullable=True)
    overlap = Column(Integer, nullable=False)
    file_upload_id = Column(Integer, nullable=True)
    updated_by_id = Column(Integer, nullable=True)
    inner_id = Column(BigInteger, nullable=True)
    total_annotations = Column(Integer, nullable=False)
    cancelled_annotations = Column(Integer, nullable=False)
    total_predictions = Column(Integer, nullable=False)
    comment_count = Column(Integer, nullable=False)
    last_comment_updated_at = Column(DateTime, nullable=True)
    unresolved_comment_count = Column(Integer, nullable=False)


def connect_to_db(db_database:str = None):
    if db_database is None:
        db_database = os.environ.get("POSTGRES_DB", "barometre")
    DB_USER = os.environ.get("POSTGRES_USER", "user")
    DB_HOST = os.environ.get("POSTGRES_HOST", "localhost")
    DB_PORT = os.environ.get("POSTGRES_PORT", 5432)
    DB_PASSWORD = os.environ.get("POSTGRES_PASSWORD", "password")

    logging.info("Connect to the host %s for DB %s" % (DB_HOST, db_database))

    url = URL.create(
        drivername="postgresql",
        username=DB_USER,
        host=DB_HOST,
        database=db_database,
        port=DB_PORT,
        password=DB_PASSWORD,
    )

    engine = create_engine(url)

    return engine


def get_db_session(engine=None):
    if engine is None:
        engine = connect_to_db()
    Session = sessionmaker(bind=engine)
    return Session()


def create_tables(conn=None, label_studio=False):
    logging.info("""Create tables in the PostgreSQL database""")
    try:
        if conn is None:
            engine = connect_to_db()
        else:
            engine = conn

        if label_studio:
            BaseLS.metadata.create_all(engine, checkfirst=True)
        else:
            Base.metadata.create_all(engine, checkfirst=True)
    except Exception as error:
        logging.error(error)
    finally:
        if engine is not None:
            engine.dispose()


def is_there_data_for_this_day_safe_guard(
    session: Session,
    date: datetime,
    country: Union[Country, CountryCollection] = FRANCE_COUNTRY,
) -> bool:
    logging.info(f"Was the previous mediatree job well executed for {date}")

    statement = select(func.count()).select_from(Keywords)
    # filter records where 'start' is within the same day
    start_of_day = date.replace(hour=0, minute=0, second=0, microsecond=0)
    end_of_day = start_of_day + timedelta(days=1)
    if not country == ALL_COUNTRIES:
        statement = statement.filter(Keywords.country == country.name)
    statement = statement.filter(
        and_(Keywords.start >= start_of_day, Keywords.start < end_of_day)
    )

    output = session.execute(statement).scalar()
    logging.info(f"Previous mediatree job got {output} elements saved")
    return output > 0


def get_keywords_for_a_day_and_channel(
    session: Session,
    date: datetime,
    channel_name: str,
    country: Union[Country, CountryCollection] = FRANCE_COUNTRY,
    limit: int = 10000,
    ids_to_avoid: List[str] = [],
) -> pd.DataFrame:
    logging.info(
        f"Getting keywords table from {date} and channel_name : {channel_name}, for country {country.name}"
    )

    statement = (
        select(
            Keywords.id,
            Keywords.start,
            Keywords.channel_program,
            Keywords.channel_program_type,
            Keywords.channel_title,
            Keywords.channel_name,
            Keywords.plaintext,
            Keywords.country,
        )
        .select_from(Keywords)
        .limit(limit)
    )
    if country == ALL_COUNTRIES:
        statement = statement.filter(
            or_(Keywords.number_of_keywords_climat > 0, Keywords.number_of_keywords > 0)
        )
    elif country in LEGACY_COUNTRIES:  # preserve legacy format
        statement = statement.filter(Keywords.country == country.name)
        statement = statement.filter(Keywords.number_of_keywords_climat > 0)
    else:
        statement = statement.filter(Keywords.country == country.name)
        statement = statement.filter(Keywords.number_of_keywords > 0)
    statement = statement.filter(Keywords.channel_name == channel_name)

    # filter records where 'start' is within the same day
    start_of_day = date.replace(hour=0, minute=0, second=0, microsecond=0)
    end_of_day = start_of_day + timedelta(days=1)
    statement = statement.filter(
        and_(Keywords.start >= start_of_day, Keywords.start < end_of_day)
    )
    if ids_to_avoid:
        statement = statement.filter(Keywords.id.notin_(ids_to_avoid))
        
    output = session.execute(statement).fetchall()

    columns = [
        "id",
        "start",
        "channel_program",
        "channel_program_type",
        "channel_title",
        "channel_name",
        "plaintext",
        "country",
    ]
    dataframe = pd.DataFrame(output, columns=columns)

    logging.info(f"Got {len(dataframe)} keywords from SQL Table Keywords")
    return dataframe


def get_keywords_for_period_and_channels(
    session: Session,
    date_start: datetime,
    date_end: datetime,
    channels: List[str],
    country: Union[Country, CountryCollection] = FRANCE_COUNTRY,
    limit: int = 10000,
    ids_to_avoid: List[str] = [],
) -> pd.DataFrame:
    logging.info(
        f"Getting keywords table from {date_start} to date {date_end}, for country {country.name}. Available channels are: \n{', '.join(channels)}"
    )

    statement = (
        select(
            Keywords.id,
            Keywords.start,
            Keywords.channel_program,
            Keywords.channel_program_type,
            Keywords.channel_title,
            Keywords.channel_name,
            Keywords.plaintext,
            Keywords.country,
        )
        .select_from(Keywords)
        .limit(limit)
    )
    if country == ALL_COUNTRIES:
        statement = statement.filter(
            or_(Keywords.number_of_keywords_climat > 0, Keywords.number_of_keywords > 0)
        )
    elif country in LEGACY_COUNTRIES:  # preserve legacy format
        statement = statement.filter(Keywords.country == country.name)
        statement = statement.filter(Keywords.number_of_keywords_climat > 0)
    else:
        statement = statement.filter(Keywords.country == country.name)
        statement = statement.filter(Keywords.number_of_keywords > 0)
    statement = statement.filter(Keywords.channel_name in channels)

    # filter records where 'start' is within the same day
    start_of_period = date_start.replace(hour=0, minute=0, second=0, microsecond=0)
    end_of_period = date_end.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
    statement = statement.filter(
        and_(Keywords.start >= start_of_period, Keywords.start < end_of_period)
    )
    if ids_to_avoid:
        statement = statement.filter(Keywords.id.notin_(ids_to_avoid))
        
    output = session.execute(statement).fetchall()

    columns = [
        "id",
        "start",
        "channel_program",
        "channel_program_type",
        "channel_title",
        "channel_name",
        "plaintext",
        "country",
    ]
    dataframe = pd.DataFrame(output, columns=columns)

    logging.info(f"Got {len(dataframe)} keywords from SQL Table Keywords")
    return dataframe


def get_labelstudio_ids(
    session: Session,
    date: datetime,
    channel_name: str,
    country: Union[Country, CountryCollection] = FRANCE_COUNTRY,
) -> List[str]:
    logging.info(
        f"Getting ids present in labelstudio table from {date} for country {country.name} and channel_name : {channel_name} for project {country.label_studio_project}"
    )
    # TODO FILTER BY PROJECT IN LABELSTUDIO

    start_of_day = date.replace(hour=0, minute=0, second=0, microsecond=0)
    end_of_day = start_of_day + timedelta(days=1)

    statement = (
        select(cast(LabelStudioTask.data["item"]["id"], Text))
        .distinct()
        .select_from(LabelStudioTask)
        .where(
            and_(
                cast(LabelStudioTask.data.op("#>>")(literal_column("ARRAY['item','channel_name']")), Text) == channel_name,
                cast(LabelStudioTask.data.op("#>>")(literal_column("ARRAY['item','start']")), DateTime) >= start_of_day,
                cast(LabelStudioTask.data.op("#>>")(literal_column("ARRAY['item','start']")), DateTime) < end_of_day,
                cast(LabelStudioTask.project_id, Integer) == int(country.label_studio_project),
            )
        )
    )
    try:
        output = session.execute(statement).fetchall()
    except Exception as e:
        session.rollback()  # ← this resets the transaction
        raise  # or log the error
    columns = ["id"]
    dataframe = pd.DataFrame(output, columns=columns)
    
    logging.info(f"Got {len(dataframe)} ids in labelstudio for date {date} for country {country.name} and channel_name : {channel_name}")
    return dataframe.id.str.replace('"', '').unique().tolist()


def get_labelstudio_records_period(
    session: Session,
    date_start: datetime,
    date_end: datetime,
    channels: List[str],
    country: Union[Country, CountryCollection] = FRANCE_COUNTRY,
) -> List[str]:
    logging.info(
        f"Getting keywords table from {date_start} to date {date_end}, "
        f"for country {country.name} (sourcing labelstudio project id: {country.label_studio_project}). "
        f"Available channels are: \n{', '.join(channels)}"
    )

    start_of_period = date_start.replace(hour=0, minute=0, second=0, microsecond=0)
    end_of_period = date_end.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
    statement = (
        select(
            LabelStudioTask.id,
            LabelStudioTask.data,
            LabelStudioTask.created_at,
            LabelStudioTask.updated_at,
            LabelStudioTask.is_labeled,
            LabelStudioTask.project_id,
            LabelStudioTask.meta,
            LabelStudioTask.overlap,
            LabelStudioTask.file_upload_id,
            LabelStudioTask.updated_by_id,
            LabelStudioTask.inner_id,
            LabelStudioTask.total_annotations,
            LabelStudioTask.cancelled_annotations,
            LabelStudioTask.total_predictions,
            LabelStudioTask.comment_count,
            LabelStudioTask.last_comment_updated_at,
            LabelStudioTask.unresolved_comment_count,
        )
        .select_from(LabelStudioTask)
        .where(
            and_(
                cast(LabelStudioTask.data.op("#>>")(literal_column("ARRAY['item','channel_name']")), Text) in channels,
                cast(LabelStudioTask.data.op("#>>")(literal_column("ARRAY['item','start']")), DateTime) >= start_of_period,
                cast(LabelStudioTask.data.op("#>>")(literal_column("ARRAY['item','start']")), DateTime) < end_of_period,
                cast(LabelStudioTask.project_id, Integer) == int(country.label_studio_project),
            )
        )
    )
    try:
        output = session.execute(statement).fetchall()
    except Exception as e:
        session.rollback()  # ← this resets the transaction
        raise  # or log the error
    columns = ["id"]
    dataframe = pd.DataFrame(output, columns=columns)
    
    logging.info(f"Found {len(dataframe)} records in labelstudio.")
    return dataframe
