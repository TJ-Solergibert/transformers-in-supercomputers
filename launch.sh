# Script que utilizo para lanzar el container, montando el directorio del proyecto dentro del container
# en /workspace/transformers-in-supercomputers
docker run --gpus all --ipc=host --ulimit memlock=-1 \
    --ulimit stack=67108864 -it --rm \
    -v /home/tj/DEV/transformers-in-supercomputers:/workspace/transformers-in-supercomputers \
    d6779f3e7f3f