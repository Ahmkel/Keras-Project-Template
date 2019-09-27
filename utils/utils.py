import argparse
import os
import json
from pathlib import Path
from google.cloud import storage
from oauth2client.service_account import ServiceAccountCredentials

from utils.dirs import verify_folder


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


def parent_folder(file_path):
    return os.path.dirname(file_path)


def get_blob(file_name,  bucket_name="accent-models"):
    storage_client = storage.Client.from_service_account_json(os.path.join(get_root(),
                                                                           'creds.json'))
    bucket = storage_client.get_bucket(bucket_name)
    # blob_path = "/".join((bucket_name, file_name))
    blob_path = file_name
    blob = bucket.blob(blob_path)
    if not blob.exists():
        print("blob doesnt exist", blob_path)
        return ""

    out_path = os.path.join(get_root(),
                            "saved_models",
                            file_name)

    verify_folder(parent_folder(out_path))
    blob.download_to_filename(out_path)

    return out_path

