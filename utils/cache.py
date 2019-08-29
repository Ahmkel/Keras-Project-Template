import os
import numpy as np
from functools import wraps


def load_from_file(file_path):
    arr = np.load(file_path + '.npy')
    return arr


def save_to_file(file_name, np_array):
    np.save(file_name + '.npy', np_array)


def np_cache(func):

    @wraps(func)
    def cached(*args, **kwargs):

        file_path = args[0] if args else kwargs["file_path"]

        # check if we already have it cached locally
        if os.path.exists(file_path + ".npy"):
            return load_from_file(file_path)

        # call the process function
        value = func(file_path)

        # save in a file
        save_to_file(file_path, value)

        return value

    return cached