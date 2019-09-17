import argparse
import os


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
