import argparse
import os
import json
from pathlib import Path

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
