class BasePreprocess(object):
    def __init__(self,data):
        self.data = data

    def preprocess(self):
        #implement data preprocessing
        raise NotImplementedError

     #other supporting methods
