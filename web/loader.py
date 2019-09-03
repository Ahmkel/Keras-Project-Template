import os
from collections import Counter

from keras.engine.saving import load_model
from data_loader.sound_utils import SoundUtils


def predict_class_audio(MFCCs, model):
    '''
    Predict class based on MFCC samples
    :param MFCCs: Numpy array of MFCCs
    :param model: Trained model
    :return: Predicted class of MFCC segment group
    '''
    MFCCs = MFCCs.reshape(MFCCs.shape[0], MFCCs.shape[1], MFCCs.shape[2], 1)
    y_predicted = model.predict_classes(MFCCs, verbose=0)
    return Counter(list(y_predicted)).most_common(1)[0][0]


class ModelLoader:

    def __init__(self, model_name):
        self.folder_path = "../saved_models/"
        self.model_name = model_name

    def _get_model_path(self):
        return os.path.join(self.folder_path, self.model_name)

    def load_model(self):
        model = load_model(self._get_model_path())
        return model
