import os
from collections import Counter

import numpy as np
from keras.engine.saving import load_model

from utils.dirs import verify_folder
from utils.utils import from_env, get_project_root, get_root, get_blob


def load_model_from_cloud(model_path):
    blob_path = get_blob(model_path)
    # verify_folder(blob_path)
    return load_model(blob_path)


def load_local_model(model_path):
    return load_model(os.path.join(get_root(), model_path))


def predict_class_audio(MFCCs):
    '''
    Predict class based on MFCC samples
    :param MFCCs: Numpy array of MFCCs
    :param model: Trained model
    :return: Predicted class of MFCC segment group
    '''
    global model
    MFCCs = MFCCs.reshape(MFCCs.shape[0], MFCCs.shape[1], MFCCs.shape[2], 1)
    y_predicted = model.predict_classes(MFCCs, verbose=0)
    return Counter(list(y_predicted)).most_common(1)[0][0]

def load(from_cloud=True):
    # The current served model based on the experiment type

    MODEL_TYPE = from_env('MODEL_TYPE', 'all_english_speakers')
    MODEL_NUM = from_env('MODEL_NUM', "d6fb4d1597eb437cabd308274c911a3a")

    # load once for the application
    model_path = "/".join((MODEL_TYPE, MODEL_NUM, "model.h5"))

    if not from_cloud:
        model = load_local_model(model_path)
    else:
        model = load_model_from_cloud(model_path)

    # BUG fix - initializing the model with an empty vector
    model.predict(np.zeros((1, 13, 30, 1)))

load()