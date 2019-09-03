import datetime
import os

from keras.preprocessing.image import img_to_array, ImageDataGenerator
from keras.applications import imagenet_utils
from PIL import Image
import numpy as np
import flask
import io

# initialize our Flask application and the Keras model
from werkzeug.utils import secure_filename

from data_loader.sound_utils import SoundUtils
from web.loader import predict_class_audio, ModelLoader

app = flask.Flask(__name__)


def load_model():
    loader = ModelLoader("model1.h5")
    return loader.load_model()


@app.route("/", methods=["GET"])
def index():
    return "Welcome to the PyImageSearch Keras REST API!"


def save_file(request):

    format = "%Y-%m-%dT%H:%M:%S"
    now = datetime.datetime.utcnow().strftime(format)

    try:
        file = request.files['file']
    except:
        file = None

    if file:
        filename = now + '_' + file.filename
        filename = secure_filename(filename)
        full_path = os.path.join("../web/uploads", filename)
        file.save(full_path)
        file_uploaded = True

    else:
        full_path = None
        file_uploaded = False

    return file_uploaded, full_path


@app.route("/predict", methods=["POST"])
def predict():
    # initialize the data dictionary that will be returned from the
    # view
    data = {"success": False}

    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        if flask.request.files.get("file"):

            # read the sound file
            status, sound_file = save_file(flask.request)
            model = load_model()
            # sound_file = os.path.join("../web/uploads", sound_file)
            mfcc = SoundUtils.segment_request_file(sound_file)
            prediction = predict_class_audio(mfcc, model)

            data["predictions"] = str(prediction)

            # indicate that the request was a success
            data["success"] = True

    # return the data dictionary as a JSON response
    return flask.jsonify(data)


# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
        "please wait until server has fully started"))
    app.config['UPLOAD_FOLDER'] = "web/uploads"
    app.run()
