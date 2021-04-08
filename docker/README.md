# Developing with Docker

## Build Docker image for GPU machine.

Run from root repository directory

    $ docker build -f docker/Dockerfile.gpu -t low-n-gpu .

## Build CPU-only Docker image

    $ docker build -f docker/Dockerfile.cpu -t low-n-cpu .

# Start Docker machine
From repository root, run `bash docker/run_cpu_docker.sh`for CPU docker
For GPU docker run `bash docker/run_gpu_docker.sh`
