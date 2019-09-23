import argparse
import os
import json
from pathlib import Path
from google.cloud import storage
from oauth2client.service_account import ServiceAccountCredentials


def get_project_root() -> Path:
    """Returns project root folder."""
    return Path(__file__).parent.parent


def get_root():
    return str(get_project_root())


def from_env(name, default, bool=False):
    val = os.getenv(name, default)
    if val.lower() in ("false", "true"):
        return val.lower() == "true"
    return os.getenv(name, default)


def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-c', '--config',
        dest='config',
        metavar='C',
        default='None',
        help='The Configuration file')
    args = argparser.parse_args()
    return args


def upload_blob(source_file_name, destination_blob_name, bucket_name="accent-models"):
    """Uploads a file to the bucket."""
    storage_client = storage.Client.from_service_account_json(os.path.join(get_root(),
                                                                           'creds.json'))
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

    print('File {} uploaded to {}.'.format(
        source_file_name,
        destination_blob_name))

# with open('creds.json') as f:
#     print("here")
# upload_blob("../requirements.txt", "test/te/requirements.txt")