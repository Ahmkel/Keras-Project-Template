from waitress import serve
import webserver.endpoints as api


PORT = 8080
HOST = "0.0.0.0"

# serve the web application
serve(api.app, host=HOST, port=PORT)
