# low-N-protein-engineering

Code to reproduce analyses in Biswas, Khimulya, Alley et al (2021). Nature Methods.

## 1. Set up a Docker environment

We use docker to provide a consistent environment and facilitate reprodubility. First build an image:

To use GPU acceleration:

    $ docker build -f docker/Dockerfile.gpu -t low-n-gpu .

To use CPU-only: 

    $ docker build -f docker/Dockerfile.cpu -t low-n-cpu .


Note you will need a GPU in order to completely reproduce the results, especially where UniRep inference is required.
The above has been tested on `p3.2xlarge` (which have a NVIDIA V100) AWS instances running `Deep Learning AMI (Ubuntu 18.04) Version 30.0 - ami-029510cec6d69f121`. 

## 2. Dig in!

Here's the repository layout: 

1. In `analysis` you'll find the code needed to reproduce the analyses in the paper. Every sub-directory within `analysis` contains a README for what analyses and figures are covered. Sorry in advance for the sometimes cryptic subdirectory names! We have a few instances of (deeply) hardcoded paths in the code. We didn't change these in order to minimize changes to the original code we wrote, figuring this would help with reproducibility.

2. `docker` contains files needed to build the docker environment (more instructions below) needed to reproduce analyses and figures.

3. `requirements` contains the requirements that are pulled into the docker environment. 



 




