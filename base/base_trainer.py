class BaseTrain(object):
    def __init__(self, model,
                 training_data,
                 validation_data,
                 config):
        self.model = model
        self.training_data = training_data
        self.validation_data = validation_data
        self.config = config

    def train(self):
        raise NotImplementedError
