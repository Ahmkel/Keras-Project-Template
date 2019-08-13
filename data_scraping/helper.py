import os

import librosa
from keras import utils

RATE = 24000
N_MFCC = 13


def to_categorical(y):
    '''
    Converts list of languages into a binary class matrix
    :param y (list): list of languages
    :return (numpy array): binary class matrix
    '''
    lang_dict = {}
    for index, language in enumerate(set(y)):
        lang_dict[language] = index
    y = list(map(lambda x: lang_dict[x], y))
    return utils.to_categorical(y, len(lang_dict))


def get_wav(file_name, base_folder=None):
    '''
    Load wav file from disk and down-samples to RATE
    :param language_num (list): list of file names
    :return (numpy array): Down-sampled wav file
    '''

    if not base_folder:
        base_folder = "../audio"
    file_path = os.path.join(base_folder, file_name)
    y, sr = librosa.load('{}.wav'.format(file_path))
    return (librosa.core.resample(y=y, orig_sr=sr, target_sr=RATE, scale=True))


def to_mfcc(wav):
    '''
    Converts wav file to Mel Frequency Ceptral Coefficients
    :param wav (numpy array): Wav form
    :return (2d numpy array: MFCC
    '''
    return (librosa.feature.mfcc(y=wav, sr=RATE, n_mfcc=N_MFCC))