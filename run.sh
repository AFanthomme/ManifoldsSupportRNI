docker run  --gpus all --mount type=bind,src=./out,dst=/app/out \
-it $(docker build --build-arg UID=$(id -u) --build-arg GID=$(id -g) -q .) "$@" 