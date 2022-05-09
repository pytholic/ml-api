#!/bin/bash

# clean up
./docker_clear.sh

# build image
docker build -t ml-api .

# run
docker run --name ml-api -d -p 8000:80 ml-api
