import datetime
import os
import urllib.request

import flask

# initialize our Flask application and the Keras model
from werkzeug.utils import secure_filename

from utils.sound import SoundUtils
from webserver.loader import predict_class_audio
from os import path
from pydub import AudioSegment

app = flask.Flask(__name__)
app.config['UPLOAD_FOLDER'] = "../webserver/uploads"

def convert_mp3(current_file):
    # files
    dst = current_file + ".wav"

    # convert wav to mp3
    sound = AudioSegment.from_mp3(current_file)
    sound.export(dst, format="wav")

    return dst

@app.route("/", methods=["GET"])
def index():
    return "Welcome to the AccentTrainer Keras REST API!"


def save_file(request):

    format = "%Y-%m-%dT%H:%M:%S"
    now = datetime.datetime.utcnow().strftime(format)

    try:
        file = request.files['file']
    except:
        file = None

    if not file:
        full_path = None
        file_uploaded = False
        return file_uploaded, full_path

    filename = file.filename
    # if file.filename.endswith('.mp3'):
    #     filename = convert_mp3(file.filename)
    # else:
    #     filename = file.filename

    filename = now + '_' + filename
    filename = secure_filename(filename)
    full_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(full_path)
    file_uploaded = True

    return file_uploaded, full_path


def download_file(file_path_url):
    name = file_path_url.split("/")[-1]
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], name)
    current_file = urllib.request.urlretrieve(file_path_url, save_path)
    return save_path

def get_prediction(sound_file):
    mfcc = SoundUtils.segment_request_file(sound_file)
    prediction = predict_class_audio(mfcc)

    return str(prediction)


@app.route("/bot", methods=["POST"])
def bot():
    response = {"success": False}

    if flask.request.method == "POST":
        data = flask.request.get_json()
        if "path" not in data:
            return "missing path"
        print(data)

        try:
            file_path = download_file(data["path"])

            prediction = get_prediction(file_path)

            # indicate that the request was a success
            response["success"] = True
            response["predictions"] = prediction
            print(response)

        except Exception as e:
            return flask.jsonify({"error": e})

    return flask.jsonify(response)


@app.route("/predict", methods=["POST"])
def predict():
    # initialize the data dictionary that will be returned from the
    # view
    response = {"success": False}

    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        if flask.request.files.get("file"):

            # read the sound file
            status, sound_file = save_file(flask.request)

            prediction = get_prediction(sound_file)
            response["predictions"] = prediction

            # indicate that the request was a success
            response["success"] = True

    # return the data dictionary as a JSON response
    return flask.jsonify(response)


# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
           "please wait until server has fully started"))
    app.run()
