FROM python:3.7 AS base
RUN apt-get update
# install apt-utils first so that package configuration isn't delayed
RUN apt-get install apt-utils --yes
# cmake is required for dlib
RUN apt-get install cmake --yes
# install dlib before hand to make rebuilds faster
RUN pip install dlib
#install required packages
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
ENV FLASK_APP face_api
ENV FLASK_RUN_HOST 0.0.0.0
WORKDIR /usr/src/app
CMD ["flask", "run"]

#development docker for local tests
FROM base AS dev
ENV FLASK_ENV development
ENV APP_CONFIG_FILE /usr/src/app/face_api/config/development.py
VOLUME /usr/src/app

#production docker
FROM base AS prod
ENV FLASK_ENV production
ENV APP_CONFIG_FILE /usr/src/app/face_api/config/production.py
RUN apt-get update
COPY wsgi.py  /usr/src/app/wsgi.py
COPY face_api /usr/src/app/face_api
COPY models /usr/src/app/models
COPY frontend/src /usr/src/app/frontend/src
EXPOSE 80/tcp
CMD exec gunicorn --log-file=- --workers "$(nproc --all)" --bind 0.0.0.0:80 -m 007 wsgi