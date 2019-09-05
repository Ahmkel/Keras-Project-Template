import os
from collections import Counter

import numpy as np
from keras.engine.saving import load_model


def get_model_path(folder, name):
    return os.path.join(folder, name)


def load_local_model(model_name, folder_path="../saved_models/"):
    model_path = get_model_path(folder_path, model_name)
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


# load once for the application
model = load_local_model("model1.h5")
# BUG fix - initializing the modle with an empty vector
model.predict(np.zeros((1, 13, 30, 1)))