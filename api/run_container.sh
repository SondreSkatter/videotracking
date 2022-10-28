#!/bin/bash

docker container run \
  -v "$(pwd):/usr/src/app" \
  -e 5000 -P \
  --net host \
  face_api \
  flask run
