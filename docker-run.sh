#!/bin/bash

# Define constant for container name
CONTAINER_NAME="whisper"

docker stop $CONTAINER_NAME
docker rm $(docker ps -aqf "name=$CONTAINER_NAME")

docker run -d --restart unless-stopped --name $CONTAINER_NAME \
    -p 9876:9876 \
    -v whisper-cache:/app/whisper-cache \
    --gpus all \
    rkilchmn/remote-faster-whisper:latest