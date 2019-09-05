import datetime
import os
import flask

# initialize our Flask application and the Keras model
from werkzeug.utils import secure_filename

from utils.sound import SoundUtils
from webserver.loader import predict_class_audio

app = flask.Flask(__name__)
app.config['UPLOAD_FOLDER'] = "../webserver/uploads"


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

    if file:
        filename = now + '_' + file.filename
        filename = secure_filename(filename)
        full_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
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
            mfcc = SoundUtils.segment_request_file(sound_file)
            prediction = predict_class_audio(mfcc)

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
    app.run()
