class BaseTrain(object):
    def __init__(self, model, data, config):
        self.model = model
        self.data = data
        self.config = config

    def train(self):
        raise NotImplementedError
