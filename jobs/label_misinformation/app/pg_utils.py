import logging
from datetime import datetime

from sqlalchemy import Column, DateTime, String, Text, Boolean, ARRAY, JSON, Integer, Table, MetaData, ForeignKey, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
import pandas as pd
from sqlalchemy import text
import os
import json
from datetime import datetime, timedelta

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

def get_keywords_for_a_day(session: Session, date: datetime, limit: int = 10000) -> list:
    logging.debug(f"Getting {batch_size} elements from offset {offset}")

    statement = select(
                Stop_Word.id,
                Keywords.start,
                Keywords.channel_program,
                Keywords.channel_program_type,
                Keywords.channel_title,
                Keywords.channel_name,
                Keywords.plaintext,
            ).select_from(Keywords) \
    .limit(limit)     

    statement = statement.filter(Keywords.number_of_keywords_climat > 0)

    # filter records where 'start' is within the same day
    start_of_day = date.replace(hour=0, minute=0, second=0, microsecond=0)
    end_of_day = start_of_day + timedelta(days=1)
    statement = statement.filter(
        and_(
            Keywords.start >= start_of_day,
            Keywords.start < end_of_day
        )
    )

    return session.execute(statement).fetchall()
