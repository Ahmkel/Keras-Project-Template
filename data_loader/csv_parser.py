from sklearn.model_selection import train_test_split
from keras import utils


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
