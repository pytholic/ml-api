#!/bin/bash

# stop and remove the old container
docker stop ml-api
docker rm ml-api

# remove the image
docker rmi ml-api
