import multiprocessing
from collections import Counter

import librosa
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from base.base_data_loader import BaseDataLoader
from data_loader import sound_utils

from data_loader.csv_parser import split_people, to_categorical, filter_df, find_classes, create_labels
from data_loader.sound_data_generator import DataGenerator
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

        classes = find_classes(y_train)
        # y_train = to_categorical(y_train)
        # y_test = to_categorical(y_test)

        self.labels = {}
        self.partition = {}

        self.partition['train'] = list(X_train)
        self.partition['validation'] = list(X_test)

        for i, data in enumerate(y_train):
            self.labels[self.partition['train'][i]] = classes[data]

        for i, data in enumerate(y_test):
            self.labels[self.partition['validation'][i]] = classes[data]

        print("Entering main")

        # values = next(generator)
        # self.X_validation = np.array(X_validation)
        # self.y_validation = np.array(y_validation)

    def get_train_generator(self):
        return DataGenerator(self.partition['train'], self.labels, self.config)

    def get_validation_generator(self):
        return DataGenerator(self.partition['validation'], self.labels, self.config)
