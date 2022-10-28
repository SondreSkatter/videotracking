#!/bin/sh
export FLASK_APP=face_api
export FLASK_RUN_HOST=0.0.0.0
export APP_CONFIG_FILE="$(pwd)/face_api/config/local.py"
exec gunicorn --workers "$(sysctl -n hw.ncpu)" --bind 0.0.0.0:8080 -m 007 wsgi