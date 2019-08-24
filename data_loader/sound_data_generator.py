import multiprocessing

import numpy as np
from joblib import Parallel, delayed
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import Sequence

from data_loader.csv_parser import to_categorical
from data_loader.sound_utils import process_sound_file, SoundUtils


class DataGenerator(Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, config, dim=(32, 32, 32), n_channels=1,
                 n_classes=3, shuffle=True):
        'Initialization'
        self.config = config
        self.dim = dim
        self.batch_size = self.config.trainer.batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle

        self.img_gen = ImageDataGenerator(width_shift_range=0.05)
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        """Generates one batch of data depending on batch size"""
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1) * self.batch_size]

        # Find list of IDs
        batch_id_list = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(batch_id_list)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, batch_id_list):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        # X = np.empty(self.batch_size, *self.dim, self.n_channels)
        # y = np.empty(self.batch_size, dtype=int)
        # y = []

        # Convert to MFCC
        if self.config.debug:
            print('Converting to MFCC....')

        X_batch = Parallel(n_jobs=multiprocessing.cpu_count(), backend='multiprocessing')(
                           delayed(process_sound_file)(name=name) for name in batch_id_list)

        y_batch = [self.labels[x] for x in batch_id_list]

        y_batch = to_categorical(y_batch)

        X_batch, y_batch = SoundUtils.make_segments(X_batch, y_batch)

        X_batch = np.array(X_batch)
        y_batch = np.array(y_batch)

        # Get row, column, and class sizes
        rows = X_batch[0].shape[0]
        cols = X_batch[0].shape[1]

        # self.config.model.input_shape = (rows, cols, 1)
        # self.config.model.num_classes = len(y_batch[0])

        X_batch = X_batch.reshape(X_batch.shape[0], rows, cols, 1)

        if self.config.debug:
            print('Converting to Images....')

        i = 0
        x_list = []
        y_list = []
        for x, y in self.img_gen.flow(X_batch, y_batch, batch_size=1):
            x_list.append(x)
            y_list.append(y)
            i += 1
            if i > self.config.trainer.batch_size:
                break  # otherwise the generator would loop indefinitely

        return x_list, y_list
        # return np.array([np.resize(imread(file_name), (200, 200)) for file_name in X_batch]),\
        #        np.array(y_batch)
