from waitress import serve
import webserver.endpoints as api

serve(api.app, host='0.0.0.0', port=8080)
