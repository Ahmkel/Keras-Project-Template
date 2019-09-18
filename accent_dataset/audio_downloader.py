import pathlib
import sys

import pandas as pd
import urllib.request
import os
import multiprocessing as mp
from pydub import AudioSegment

from utils.utils import get_root


class AudioDownloader:

    def __init__(self, csv_filepath, destination_folder=None, wait=1.5, debug=False):
        '''
        Initializes GetAudio class object
        :param destination_folder (str): Folder where audio files will be saved
        :param wait (float): Length (in seconds) between web requests
        :param debug (bool): Outputs status indicators to console when True
        '''
        self.csv_filepath = csv_filepath
        self.audio_df = pd.read_csv(csv_filepath)
        self.url = 'http://chnm.gmu.edu/accent/soundtracks/{}.mp3'
        if not destination_folder:
            self.destination_folder = os.path.join(get_root(), "datasets", "audio")
        else:
            self.destination_folder = destination_folder
        self.wait = wait
        self.debug = debug

    def check_path(self):
        '''
        Checks if self.distination_folder exists. If not, a folder called self.destination_folder is created
        '''
        pathlib.Path(self.destination_folder).mkdir(parents=True, exist_ok=True)

    def download(self, url, output_file):

        # create the output file path
        sound_file_path = os.path.join(self.destination_folder, "{}.wav".format(output_file))

        if self.debug:
            print('downloading {} to {}'.format(output_file, sound_file_path))
        (filename, headers) = urllib.request.urlretrieve(url)
        sound = AudioSegment.from_mp3(filename)

        sound.export(sound_file_path, format="wav")

    def get_audio(self):
        '''
        Retrieves all audio files from 'language_num' column of self.audio_df
        If audio file already exists, move on to the next
        :return (int): Number of audio files downloaded
        '''

        self.check_path()

        audio_urls = []
        file_name = []

        # Create all the necessary URLs for download
        for lang_num in self.audio_df['language_num']:
            if not os.path.exists(self.destination_folder + '{}.wav'.format(lang_num)):
                audio_urls.append(self.url.format(lang_num))
                file_name.append(lang_num)

        if self.debug:
            print('Downloading {} files'.format(len(audio_urls)))
        pool = mp.Pool(processes=mp.cpu_count())
        # create a *map, unpacks variables
        res = pool.starmap(self.download, zip(audio_urls, file_name))

        return len(audio_urls)


# if __name__ == '__main__':
#     '''
#     Example console command
#     python GetAudio.py audio_metadata.csv
#     '''
#     csv_file = sys.argv[1]
#     ga = AudioDownloader(csv_filepath=csv_file,
#                          debug=True)
#     ga.get_audio()

