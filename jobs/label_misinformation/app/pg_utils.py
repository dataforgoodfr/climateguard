import json
import logging
import os
from typing import Union
from datetime import datetime, timedelta

import modin.pandas as pd
from country import (
    ALL_COUNTRIES,
    BRAZIL_COUNTRY,
    BELGIUM_COUNTRY,
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
    create_engine,
    func,
    or_,
    select,
    text,
)
from sqlalchemy.orm import Session, declarative_base, sessionmaker

Base = declarative_base()

keywords_table = "keywords"
class Keywords(Base):
    __tablename__ = keywords_table

    id = Column(Text, primary_key=True)
    channel_name = Column(String, nullable=False)
    channel_title = Column(String, nullable=True)
    channel_program = Column(String, nullable=True) #  arcom - alembic handles this
    channel_program_type = Column(String, nullable=True) # arcom - (magazine, journal etc) alembic handles this
    channel_radio = Column(Boolean, nullable=True)
    start = Column(DateTime())
    plaintext= Column(Text)
    theme=Column(JSON) #keyword.py  # ALTER TABLE keywords ALTER theme TYPE json USING to_json(theme);
    created_at = Column(DateTime(timezone=True), server_default=text("(now() at time zone 'utc')")) # ALTER TABLE ONLY keywords ALTER COLUMN created_at SET DEFAULT (now() at time zone 'utc');
    updated_at = Column(DateTime(), default=datetime.now, onupdate=text("now() at time zone 'Europe/Paris'"), nullable=True)
    keywords_with_timestamp = Column(JSON) # ALTER TABLE keywords ADD keywords_with_timestamp json;
    number_of_keywords_climat = Column(Integer) # sum of all climatique counters without duplicate (like number_of_keywords)
    number_of_keywords = Column(Integer) # sum of all climatique counters without duplicate (like number_of_keywords)
    country = Column(Text)
  
def connect_to_db():
    DB_DATABASE = os.environ.get("POSTGRES_DB", "barometre")
    DB_USER = os.environ.get("POSTGRES_USER", "user")
    DB_HOST = os.environ.get("POSTGRES_HOST", "localhost")
    DB_PORT = os.environ.get("POSTGRES_PORT", 5432)
    DB_PASSWORD = os.environ.get("POSTGRES_PASSWORD", "password")

    logging.info("Connect to the host %s for DB %s" % (DB_HOST, DB_DATABASE))

    url = URL.create(
        drivername="postgresql",
        username=DB_USER,
        host=DB_HOST,
        database=DB_DATABASE,
        port=DB_PORT,
        password=DB_PASSWORD,
    )

    engine = create_engine(url)

    return engine

def get_db_session(engine = None):
    if engine is None:
        engine = connect_to_db()
    Session = sessionmaker(bind=engine)
    return Session()

def create_tables(conn=None):
    logging.info("""Create tables in the PostgreSQL database""")
    try:
        if conn is None :
            engine = connect_to_db()
        else:
            engine = conn

        Base.metadata.create_all(engine, checkfirst=True)
    except (Exception) as error:
        logging.error(error)
    finally:
        if engine is not None:
            engine.dispose()


def is_there_data_for_this_day_safe_guard(session: Session, date: datetime, country: Union[Country, CountryCollection] = FRANCE_COUNTRY) -> bool:
    logging.info(f"Was the previous mediatree job well executed for {date}")

    statement = select(
                func.count()
            ).select_from(Keywords) \

    # filter records where 'start' is within the same day
    start_of_day = date.replace(hour=0, minute=0, second=0, microsecond=0)
    end_of_day = start_of_day + timedelta(days=1)
    if not country == ALL_COUNTRIES:
        statement = statement.filter(Keywords.country == country.name)
    statement = statement.filter(
        and_(
            Keywords.start >= start_of_day,
            Keywords.start < end_of_day
        )
    )

    output = session.execute(statement).scalar()
    logging.info(f"Previous mediatree job got {output} elements saved")
    return output > 0

def get_keywords_for_a_day_and_channel(session: Session, date: datetime, channel_name: str, country: Union[Country, CountryCollection] = FRANCE_COUNTRY, limit: int = 10000) -> pd.DataFrame:
    logging.info(f"Getting keywords table from {date} and channel_name : {channel_name}, for country {country.name}")

    statement = select(
                Keywords.id,
                Keywords.start,
                Keywords.channel_program,
                Keywords.channel_program_type,
                Keywords.channel_title,
                Keywords.channel_name,
                Keywords.plaintext,
            ).select_from(Keywords) \
    .limit(limit)     
    if country == ALL_COUNTRIES:
        statement = statement.filter(
            or_(
                Keywords.number_of_keywords_climat > 0, 
                Keywords.number_of_keywords > 0
            )
        )
    elif country in LEGACY_COUNTRIES: # preserve legacy format
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
        and_(
            Keywords.start >= start_of_day,
            Keywords.start < end_of_day
        )
    )

    output = session.execute(statement).fetchall()

    columns = ["id", "start", "channel_program", "channel_program_type", "channel_title", "channel_name", "plaintext"]
    dataframe = pd.DataFrame(output, columns=columns)

    logging.info(f"Got {len(dataframe)} keywords from SQL Table Keywords")
    return dataframe
