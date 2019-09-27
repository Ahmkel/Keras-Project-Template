from waitress import serve
import webserver.endpoints as api
from utils.utils import from_env

PORT = int(from_env('PORT', '8080'))
HOST = from_env('HOST', "0.0.0.0")

# serve the web application
serve(api.app, host=HOST, port=PORT)
