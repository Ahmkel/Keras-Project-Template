import librosa

from base.base_model import BaseModel
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten


class AccentModel(BaseModel):

    def __init__(self, config):
        super(AccentModel, self).__init__(config)
        self.build_model()

    def build_model(self):
        self.model = Sequential()

        self.model.add(Conv2D(32, kernel_size=(3, 3), activation='relu',
                              data_format="channels_last",
                              input_shape=self.config.model.input_shape))

        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(rate=0.25))

        self.model.add(Flatten())
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dropout(rate=0.5))

        self.model.add(Dense(self.config.model.num_classes, activation='softmax'))
        self.model.compile(loss='categorical_crossentropy',
                           optimizer='adadelta',
                           metrics=['accuracy'])