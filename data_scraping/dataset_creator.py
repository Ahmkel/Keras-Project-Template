from data_loader.accent_data_loader import AccentDataLoader
from data_scraping.constants import DEFAULT_LANGUAGES
from data_scraping.fromwebsite import scrape


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

    # Creates a CSV files with all the links to sound files in the
    dataset_with_all_english_speakers(download=True,
                                      languages=DEFAULT_LANGUAGES)
    dataset_with_only_usa_natives(input_file="../" + AccentDataLoader.csv_path("all_english_speakers.csv"))
    # dataset_with_all_english_speakers(languages=DEFAULT_LANGUAGES,
    #                                   input_file="../" + AccentDataLoader.csv_path("all_speakers.csv"),


