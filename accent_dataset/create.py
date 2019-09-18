import os

from data_loader.accent_data_loader import AccentDataLoader
from accent_dataset.constants import DEFAULT_LANGUAGES
from accent_dataset.web_scraping import scrape


def dataset_with_only_usa_natives(input_file="", download=True):
    languages = DEFAULT_LANGUAGES

    scrape(destination_file="usa_english_speakers.csv",
           languages=languages,
           only_usa=True,
           download=download,
           input_file=input_file)


def dataset_with_all_english_speakers(languages,
                                      input_file="",
                                      download=True):
    scrape(destination_file="all_english_speakers.csv",
           languages=languages,
           download=download,
           input_file=input_file)


if __name__ == '__main__':

    ################################################
    # Creates a CSV file with the given languages and
    # download the relevant audio files
    ################################################
    dataset_with_all_english_speakers(download=True,
                                      languages=DEFAULT_LANGUAGES)
    dataset_with_only_usa_natives(input_file=AccentDataLoader.csv_path("all_english_speakers.csv"))

    ################################################
    # Uncomment if the sound CSV is already present
    # and you want only to download the audio file
    ################################################

    # dataset_with_all_english_speakers(download=True,
    #                                   languages=DEFAULT_LANGUAGES,
    #                                   input_file=AccentDataLoader.csv_path("all_english_speakers.csv"))



