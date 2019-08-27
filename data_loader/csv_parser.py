from sklearn.model_selection import train_test_split
from keras import utils


def filter_df(df):
    '''
    Function to filter audio files based on df columns
    df column options: [age,age_of_english_onset,age_sex,birth_place,english_learning_method,
    english_residence,length_of_english_residence,native_language,other_languages,sex]
    :param df (DataFrame): Full unfiltered DataFrame
    :return (DataFrame): Filtered DataFrame
    '''

    arabic = df[df.native_language == 'arabic']
    mandarin = df[df.native_language == 'mandarin']
    english = df[df.native_language == 'english']
    russian = df[df.native_language == 'english']
    amharic = df[df.native_language == 'english']
    french = df[df.native_language == 'english']

    mandarin = mandarin[mandarin.length_of_english_residence < 10]
    arabic = arabic[arabic.length_of_english_residence < 10]

    df = df.append(english)
    df = df.append(arabic)
    df = df.append(mandarin)

    return df


def split_people(df, test_size=0.2):
    '''
    Create train test split of DataFrame
    :param df (DataFrame): Pandas DataFrame of audio files to be split
    :param test_size (float): Percentage of total files to be split into test
    :return X_train, X_test, y_train, y_test (tuple): Xs are list of df['language_num'] and Ys are df['native_language']
    '''

    return train_test_split(df['language_num'], df['native_language'], test_size=test_size, random_state=1234)


def to_categorical(y):
    '''
    Converts list of languages into a binary class matrix
    :param y (list): list of languages
    :return (numpy array): binary class matrix
    '''
    lang_dict = {"english": 0,
                 "other": 1}
    # for index, language in enumerate(set(y)):
    #     lang_dict[language] = index

    # go over all the result and convert the name of the language to its index
    # because we have only two classes, it's either english or not
    # y = list(map(lambda x: lang_dict[x] if x == "english" else lang_dict["other"], y))
    y = list(map(lambda x: lang_dict["english"] if x == "english" else lang_dict["other"], y))

    return utils.to_categorical(y, len(lang_dict))


def find_classes(y):
    lang_dict = {}
    for index, language in enumerate(set(y)):
        lang_dict[language] = index

    return lang_dict


def create_labels(y, classes):
    pass

# if __name__ == '__main__':
#     '''
#     Console command example:
#     python bio_data.csv
#     '''
#
#     csv_file = sys.argv[1]
#     df = pd.read_csv(csv_file)
#     filtered_df = filter_df(df)
#     print(split_people(filtered_df))
