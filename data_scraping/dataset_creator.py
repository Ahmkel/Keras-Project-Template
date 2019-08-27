from data_loader.accent_data_loader import AccentDataLoader
from data_scraping.fromwebsite import scrape

DEFAULT_LANGUAGES = ["english",
                     "russian",
                     "arabic",
                     "french",
                     "spanish",
                     "amharic",
                     "hebrew"]


def dataset_with_only_usa_natives(input_file="", download=True):
    languages = DEFAULT_LANGUAGES
    # languages = ["english"]

    scrape(destination_file="only_usa_native_speakers.csv",
           languages=languages,
           only_usa=True,
           download=download,
           input_file=input_file)


def dataset_with_all_english_speakers(download=True):
    languages = DEFAULT_LANGUAGES
    scrape(destination_file="all_english_speakers.csv",
           languages=languages)


if __name__ == '__main__':
    # download once all names
    # dataset_with_all_english_speakers(download=True)

    # filter a seperate csv
    dataset_with_only_usa_natives(input_file="../"+AccentDataLoader.csv_path("all_english_speakers.csv"),
                                  download=False)
