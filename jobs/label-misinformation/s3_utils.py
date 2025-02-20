import logging
import os
import boto3
import modin.pandas as pd
from sentry_sdk.crons import monitor
import shutil
import sys

def get_secret_docker(secret_name):
    secret_value = os.environ.get(secret_name, "")

    if secret_value and os.path.exists(secret_value):
        with open(secret_value, "r") as file:
            return file.read().strip()
    return secret_value

# Configuration for Scaleway Object Storage
ACCESS_KEY = get_secret_docker('BUCKET_ACCESS')
SECRET_KEY = get_secret_docker("BUCKET_SECRET")
REGION = 'fr-par'
ENDPOINT_URL = f'https://s3.{REGION}.scw.cloud'
boto3.setup_default_session(region_name=REGION)

def get_s3_client():
    s3_client = boto3.client(
        service_name='s3',
        region_name=REGION,
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY,
        endpoint_url=ENDPOINT_URL,
    )
    return s3_client

def get_bucket_key(date, channel, filename:str="*", suffix:str="parquet"):
    (year, month, day) = (date.year, date.month, date.day)
    return f'year={year}/month={month:1}/day={day:1}/channel={channel}/{filename}.{suffix}'

def get_bucket_key_folder(date, channel, root_folder = None):
    (year, month, day) = (date.year, date.month, date.day)
    key = f'year={year}/month={month:1}/day={day:1}/channel={channel}/'
    if root_folder is not None:
        return f"{root_folder}/{key}"
    
    return key
     


def read_folder_from_s3(date, channel: str, bucket: str):
    s3_path: str = get_bucket_key_folder(date=date, channel=channel)
    s3_key: tuple[str] = f"s3://{bucket}/{s3_path}"
    logging.info(f"Reading S3 folder {s3_key}")
   
    df = pd.read_parquet(path=s3_key,
                                storage_options={
                                "key": ACCESS_KEY,
                                "secret": SECRET_KEY,
                                "endpoint_url": ENDPOINT_URL,
                              })

    logging.info(f"read {len(df)} rows from S3")
    return df



def check_if_object_exists_in_s3(day, channel, s3_client, bucket: str, root_folder = None) -> bool:
    folder_prefix = get_bucket_key_folder(day, channel, root_folder=root_folder)
    
    logging.debug(f"Checking if folder exists: {folder_prefix}")
    try:
        response = s3_client.list_objects_v2(Bucket=bucket, Prefix=folder_prefix, MaxKeys=1)
        if "Contents" in response:
            logging.info(f"Folder exists in S3: {folder_prefix}")
            return True
        else:
            logging.info(f"Folder does not exist in S3: {folder_prefix}")
            return False
    except Exception as e:
        logging.error(f"Error while checking folder in S3: {folder_prefix}\n{e}")
        return False

def upload_folder_to_s3(local_folder, bucket_name, base_s3_path, s3_client):
    logging.info(f"Reading local folder {local_folder} and uploading to S3")
    for root, _, files in os.walk(local_folder):
        logging.info(f"Reading files {len(files)}")
        for file in files:
            logging.info(f"Reading {file}")
            local_file_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_file_path, local_folder)
            s3_key = os.path.join(base_s3_path, relative_path).replace("\\", "/")  # replace backslashes for S3 compatibility
            
            logging.info(f"Uploading to bucket {bucket_name} key : {s3_key}")
            s3_client.upload_file(local_file_path, bucket_name, s3_key)
            logging.info(f"Uploaded: {s3_key}")
            # Delete the local folder after successful upload
            shutil.rmtree(local_folder)
            logging.info(f"Deleted local folder: {local_folder}")


def save_to_s3(df: pd.DataFrame, channel: str, date: pd.Timestamp, s3_client, bucket: str, folder_inside_bucket = None):
    logging.info(f"Saving DF with {len(df)} elements to S3 for {date} and channel {channel}")

    # to create partitions
    object_key = get_bucket_key(date, channel)
    logging.debug(f"Uploading partition: {object_key}")

    try:
        # add partition columns year, month, day to dataframe
        df['year'] = date.year
        df['month'] = date.month
        df['day'] = date.day
        df['channel'] = channel # channel_name from mediatree's api

        df = df._to_pandas() # collect data accross ray workers to avoid multiple subfolders

        based_path = "s3/parquet"
        local_parquet = based_path
        if folder_inside_bucket is not None:
            local_parquet = f"{based_path}/{folder_inside_bucket}"
        df.to_parquet(local_parquet,
                       compression='gzip'
                       ,partition_cols=['year', 'month', 'day', 'channel'])

        #saving full_path folder parquet to s3
        s3_path = f"{get_bucket_key_folder(date, channel, root_folder=folder_inside_bucket)}"
        local_folder = f"{based_path}/{s3_path}"
        upload_folder_to_s3(local_folder, bucket, s3_path, s3_client=s3_client)
        
    except Exception as err:
        logging.fatal("get_and_save_api_data (%s) %s" % (type(err).__name__, err))
        raise Exception