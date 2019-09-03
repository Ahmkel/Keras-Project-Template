from waitress import serve
import web.endpoints as api

serve(api.app, host='0.0.0.0', port=8080)
