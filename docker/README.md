Running Lamtram via Docker
==========================

[Docker](https://www.docker.com/) makes running software with layers of dependencies significantly easier, especially in non-standard environments like research clusters.  To get started with Docker on GPUs, install the following:
* CUDA (7.5 or later): [download page](https://developer.nvidia.com/cuda-downloads)
* nvidia-docker: [github repository](https://github.com/NVIDIA/nvidia-docker)

Software can then be built automatically and deterministically in Docker containers.

Quick Start
-----------

Once CUDA and nvidia-docker are installed, run:

    ./docker/build

Then run:

    ./docker/run lamtram-train --help

Other programs and scripts are also on the default path.  If you make changes to Lamtram, simply re-run the build script to recompile/rebuild.

For more details on what Docker is doing, read on.

Building the Lamtram Image
--------------------------

To build a Docker image of the current commit of Lamtram, just run the build script:

    ./docker/build

This will start a new container based on Ubuntu Linux with CUDA and download/build all of Lamtram's dependencies.  Then the current version of Lamtram (the one in this checkout) will be copied into the container and built.

The container will be saved as an image named "lamtram" (dependencies are cached as "lamtram-deps" for future builds).  You can change the name by editing the file `docker/config`.  You can also edit properties of the build in `docker/Dockerfile.lamtram-deps` and `docker/Dockerfile.lamtram`.  For instance, if you make changes to Lamtram that require a newer version of DyNet, you can update `DYNET_VERSION`.

To see current Docker images available on your system, you can run `docker images` which will return something like this:

    REPOSITORY          TAG           IMAGE ID            CREATED            SIZE
    lamtram             latest        d325acb19524        3 hours ago        3.569 GB
    lamtram-deps        latest        7a113876af1f        3 hours ago        3.262 GB

You can remove images with `docker rmi`:

    docker rmi -f lamtram:latest

See the [Docker documentation](https://docs.docker.com/engine/reference/commandline/rmi/) for more information on managing images.

Running Lamtram via Docker
--------------------------

Once the image is built, you can run Lamtram using the run script:

    ./docker/run lamtram-train --help

Lamtram programs (`lamtram`, `lamtram-train`), scripts (`convert-cond.pl`, `unk-single.pl`), and support programs (`fast_align`) are all available on the default path.  Docker containers are isolated from the host operating system, so only specific volumes (directories) are accessible.  By default, the run script mounts the current working directory and anything in `VOLUME_PATHS` (edit in the script to fit your setup).  Otherwise, programs can be run normally.  To run any example in the Lamtram documentation, simply add `/path/to/lamtram/docker/run` before the command.
