import os
import subprocess

import librosa
import numpy as np
from pydub import AudioSegment
from sklearn.preprocessing import MinMaxScaler

from utils.cache import np_cache

SILENCE_THRESHOLD = .01
RATE = 24000
N_MFCC = 13
COL_SIZE = 30


def get_wav(file_path):
    '''
    Load wav file from disk and down-samples to RATE
    :param language_num (list): list of file names
    :return (numpy array): Down-sampled wav file
    '''

    y, sr = librosa.load(file_path)
    return librosa.core.resample(y=y, orig_sr=sr, target_sr=RATE, scale=True)


def to_mfcc(wav):
    '''
    Converts wav file to Mel Frequency Ceptral Coefficients
    :param wav (numpy array): Wav form
    :return (2d numpy array: MFCC
    '''
    # print("started")
    mfcc = librosa.feature.mfcc(y=wav, sr=RATE, n_mfcc=N_MFCC)

    return mfcc


@np_cache
def process_sound_file(file_path):
    """
    process a sound file - reading a wav and converting it to mfcc
    :param file_name:
    :return:
    """

    return to_mfcc(get_wav(file_path))


class SoundUtils:

    @staticmethod
    def segment_request_file(request_file):
        SoundUtils.trim_file(request_file)
        y, sr = librosa.load(request_file)
        wav = librosa.core.resample(y=y, orig_sr=sr, target_sr=RATE, scale=True)
        # wav = SoundUtils.remove_silence(wav)
        mfcc = to_mfcc(wav)
        segments = SoundUtils.segment_one(mfcc)

        return segments

    def detect_leading_silence(sound, silence_threshold=-50.0, chunk_size=10):
        '''
        sound is a pydub.AudioSegment
        silence_threshold in dB
        chunk_size in ms

        iterate over chunks until you find the first one with sound
        '''
        trim_ms = 0  # ms

        assert chunk_size > 0  # to avoid infinite loop
        while sound[trim_ms:trim_ms + chunk_size].dBFS < silence_threshold and trim_ms < len(sound):
            trim_ms += chunk_size

        return trim_ms

    @staticmethod
    def get_ext(filename):
        return os.path.splitext(filename)[1]

    @staticmethod
    def set_ext(filename, ext):
        return os.path.splitext(filename)[0] + '.' + ext

    @staticmethod
    def needs_conversion(file_path):
        # every none wav file needs a conversion
        return SoundUtils.get_ext(file_path) != ".wav"

    @staticmethod
    def convert(file_path, convert_to_ext="wav"):
        new_path = SoundUtils.set_ext(file_path, ext=convert_to_ext)
        subprocess.run(['ffmpeg', '-i', file_path, new_path])
        return new_path

    @staticmethod
    def trim_file(file_path, out=None):

        # checks if the file needs to be converted first
        if SoundUtils.needs_conversion(file_path):
            file_path = SoundUtils.convert(file_path)

        # in case we require a conversion from oga to wav for trimming
        if SoundUtils.get_ext(file_path) == ".oga":
            new_path = SoundUtils.set_ext(file_path, ext="wav")
            subprocess.run(['ffmpeg', '-i', file_path, new_path])
            file_path = new_path

        sound = AudioSegment.from_file(file_path, format="wav")

        start_trim = SoundUtils.detect_leading_silence(sound)
        end_trim = SoundUtils.detect_leading_silence(sound.reverse())
        duration = len(sound)
        trimmed_sound = sound[start_trim:duration - end_trim]
        output_path = file_path if not out else out
        trimmed_sound.export(output_path, format="wav")

    @staticmethod
    def remove_silence(wav, thresh=0.04, chunk=5000):
        '''
        Searches wav form for segments of silence. If wav form values are lower than 'thresh' for 'chunk' samples, the values will be removed
        :param wav (np array): Wav array to be filtered
        :return (np array): Wav array with silence removed
        '''

        tf_list = []
        for x in range(int(len(wav) / chunk)):
            if (np.any(wav[chunk * x:chunk * (x + 1)] >= thresh) or np.any(wav[chunk * x:chunk * (x + 1)] <= -thresh)):
                tf_list.extend([True] * chunk)
            else:
                tf_list.extend([False] * chunk)

        tf_list.extend((len(wav) - len(tf_list)) * [False])
        return (wav[tf_list])

    # @staticmethod
    # def detect_leading_silence(sound, silence_threshold=-50.0, chunk_size=10):
    #     trim_ms = 0  # ms
    #
    #     assert chunk_size > 0  # to avoid infinite loop
    #     while sound[trim_ms:trim_ms + chunk_size].dBFS < silence_threshold and trim_ms < len(sound):
    #         trim_ms += chunk_size
    #
    #     return trim_ms

    @staticmethod
    def normalize_mfcc(mfcc):
        '''
        Normalize mfcc
        :param mfcc:
        :return:
        '''
        mms = MinMaxScaler()
        return (mms.fit_transform(np.abs(mfcc)))

    @staticmethod
    def make_segments(mfccs, labels):
        '''
        Makes segments of mfccs and attaches them to the labels
        :param mfccs: list of mfccs
        :param labels: list of labels
        :return (tuple): Segments with labels
        '''
        segments = []
        seg_labels = []
        for mfcc, label in zip(mfccs, labels):
            for start in range(0, int(mfcc.shape[1] / COL_SIZE)):
                segments.append(mfcc[:, start * COL_SIZE:(start + 1) * COL_SIZE])
                seg_labels.append(label)
        return (segments, seg_labels)

    @staticmethod
    def segment_one(mfcc):
        '''
        Creates segments from on mfcc image. If last segments is not long enough to be length of columns divided by COL_SIZE
        :param mfcc (numpy array): MFCC array
        :return (numpy array): Segmented MFCC array
        '''
        segments = []
        for start in range(0, int(mfcc.shape[1] / COL_SIZE)):
            segments.append(mfcc[:, start * COL_SIZE:(start + 1) * COL_SIZE])
        return (np.array(segments))

    @staticmethod
    def create_segmented_mfccs(X_train):
        '''
        Creates segmented MFCCs from X_train
        :param X_train: list of MFCCs
        :return: segmented mfccs
        '''
        segmented_mfccs = []
        for mfcc in X_train:
            segmented_mfccs.append(SoundUtils.segment_one(mfcc))
        return (segmented_mfccs)

