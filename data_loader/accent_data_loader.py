import multiprocessing
import os

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from base.base_data_loader import BaseDataLoader

from data_loader.csv_parser import split_people, to_categorical
from utils.sound import SoundUtils, process_sound_file
from tqdm import tqdm


class AccentDataLoader(BaseDataLoader):

    DATASET_DIR = "datasets"
    TRAINING_FILES_DIR = "training_files"
    AUDIO_FILES_DIR = "audio"

    AUDIO_PATH = os.path.join(DATASET_DIR, AUDIO_FILES_DIR)

    @staticmethod
    def csv_path(file_name):

        return os.path.join(AccentDataLoader.DATASET_DIR,
                            AccentDataLoader.TRAINING_FILES_DIR,
                            file_name)

    def _csv_path(self):

        return self.csv_path(self.config.data_loader.data_file)

    @staticmethod
    def _audio_path(name):
        return os.path.join(AccentDataLoader.DATASET_DIR,
                            AccentDataLoader.AUDIO_FILES_DIR,
                            "{}.wav".format(name))

    @staticmethod
    def preproccess_data(X):
        return Parallel(n_jobs=multiprocessing.cpu_count(), backend='multiprocessing')(
                           delayed(process_sound_file)(file_path=AccentDataLoader._audio_path(name)) for name in tqdm(X))

    def __init__(self, config):
        super(AccentDataLoader, self).__init__(config)

        self.X_train = []
        self.y_train = []
        self.X_validation = []
        self.y_validation = []

        self.init_model()

    def init_model(self):

        # Load metadata
        filtered_df = pd.read_csv(self._csv_path())
        # Shuffle the data frame
        filtered_df = filtered_df.sample(frac=1).reset_index(drop=True)

        # Train test split
        X_train, X_test, y_train, y_test = split_people(filtered_df)

        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)

        # Get resampled wav files using multiprocessing
        if self.config.debug:
            print('Loading wav files....')

        X_train = self.preproccess_data(X_train)
        X_test = self.preproccess_data(X_test)

        # Create segments from MFCCs
        X_train, y_train = SoundUtils.make_segments(X_train, y_train)
        X_validation, y_validation = SoundUtils.make_segments(X_test, y_test)

        print("Entering main")

        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_validation = np.array(X_validation)
        y_validation = np.array(y_validation)

        # Get row, column, and class sizes
        rows = X_train[0].shape[0]
        cols = X_train[0].shape[1]
        val_rows = X_validation[0].shape[0]
        val_cols = X_validation[0].shape[1]
        num_classes = len(y_train[0])

        # input image dimensions to feed into 2D ConvNet Input layer
        input_shape = (rows, cols, 1)
        self.X_train = X_train.reshape(X_train.shape[0], rows, cols, 1)
        self.y_train = y_train
        self.X_validation = X_validation.reshape(X_validation.shape[0], val_rows, val_cols, 1)
        self.y_validation = y_validation

    def get_train_data(self):
        return self.X_train, self.y_train

    def get_validation_data(self):
        return self.X_validation, self.y_validation
