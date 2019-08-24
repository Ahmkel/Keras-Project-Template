import multiprocessing
from collections import Counter

import librosa
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from base.base_data_loader import BaseDataLoader
from data_loader import sound_utils

from data_loader.csv_parser import split_people, to_categorical, filter_df
from data_loader.sound_utils import SoundUtils


class BaseAccentDataLoader(BaseDataLoader):

    def __init__(self, config):
        super(BaseAccentDataLoader, self).__init__(config)

        # Load metadata
        df = pd.read_csv("data_loader/bio_data_small.csv")
        # Filter metadata to retrieve only files desired
        filtered_df = filter_df(df)

        # Train test split
        X_train, X_test, y_train, y_test = split_people(filtered_df)

        # Get statistics
        train_count = Counter(y_train)
        test_count = Counter(y_test)

        print("Entering main")

        # import ipdb;
        # ipdb.set_trace()

        acc_to_beat = test_count.most_common(1)[0][1] / float(np.sum(list(test_count.values())))

        # To categorical
        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)

        # Get resampled wav files using multiprocessing
        if config.debug:
            print('Loading wav files....')
        pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
        X_train = map(sound_utils.get_wav, X_train)
        X_test = map(sound_utils.get_wav, X_test)

        # Convert to MFCC
        if config.debug:
            print('Converting to MFCC....')
        # X_train = pool.map(sound_utils.to_mfcc, X_train)
        # X_test = pool.map(sound_utils.to_mfcc, X_test)
        X_train = Parallel(n_jobs=multiprocessing.cpu_count(), backend='multiprocessing')(
            delayed(sound_utils.to_mfcc)(wav=data) for data in X_train)
        X_test = Parallel(n_jobs=multiprocessing.cpu_count(), backend='multiprocessing')(
            delayed(sound_utils.to_mfcc)(wav=data) for data in X_test)

        if config.debug:
            print('Finished MFCC conversion....')

        # Create segments from MFCCs
        X_train, y_train = SoundUtils.make_segments(X_train, y_train)
        X_validation, y_validation = SoundUtils.make_segments(X_test, y_test)

        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)
        self.X_validation = np.array(X_validation)
        self.y_validation = np.array(y_validation)

        # Get row, column, and class sizes
        rows = self.X_train[0].shape[0]
        cols = self.X_train[0].shape[1]
        val_rows = self.X_validation[0].shape[0]
        val_cols = self.X_validation[0].shape[1]

        self.X_train = self.X_train.reshape(self.X_train.shape[0], rows, cols, 1)
        self.X_validation = self.X_validation.reshape(self.X_validation.shape[0], val_rows, val_cols, 1)

        print('X_train shape:', self.X_train.shape)
        print(self.X_train.shape[0], 'training samples')

        # input image dimensions to feed into 2D ConvNet Input layer
        config.model.input_shape = (rows, cols, 1)
        config.model.num_classes = len(self.y_train[0])

    def get_train_data(self):
        return self.X_train, self.y_train

    def get_test_data(self):
        return self.X_validation, self.y_validation
        # return self.X_test, self.y_test

    def get_validation_data(self):
        return self.X_validation, self.y_validation
