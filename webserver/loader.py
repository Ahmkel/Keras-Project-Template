import os
from collections import Counter

import numpy as np
from keras.engine.saving import load_model

from utils.utils import from_env, get_project_root, get_root


def load_local_model(model_path):
    model = load_model(model_path)
    return model


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


# The current served model based on the experiment type

MODEL_TYPE = from_env('MODEL_TYPE', 'usa_english_speakers')
MODEL_NUM = from_env('MODEL_NUM', "ba71cbd87cf240d0a9f4e9584982366d")

# load once for the application
model_path = os.path.join(get_root(),
                          "saved_models",
                          MODEL_TYPE, MODEL_NUM, "model.h5")
print("HERERR")
print(model_path)
model = load_local_model(model_path)
# BUG fix - initializing the model with an empty vector
model.predict(np.zeros((1, 13, 30, 1)))