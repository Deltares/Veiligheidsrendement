# To build this docker (from the root checkout) run:
# `docker build -t vrtool_dev .devcontainer`
# To run the docker for pytest:
# `docker run -v {your_vrtool_checkout_dir}:/usr/src/app/ -v {vrtool_env_dir}:/usr/src/.env -t vrtool_dev`
# `pixi install -e dev`
# `pixi run -e dev pytest -m {your_pytest_arguments}`

# FROM condaforge/miniforge3:latest
FROM containers.deltares.nl/gfs-dev/vrtool_externals:latest AS vrtool_externals-env
FROM ghcr.io/prefix-dev/pixi:latest AS pixi-base

# This image overlay was the best way I found to run (linux) .Net applications within
# a miniforge container.

ARG SRC_ROOT="/usr/src"
ARG SRC_EXTERNALS="${SRC_ROOT}/test_externals"
COPY --from=vrtool_externals-env $SRC_ROOT/test_externals ${SRC_EXTERNALS}

# Set PIXI_HOME to a location inside the source root
ENV PIXI_HOME=$SRC_ROOT/.pixi

# This location needs to be mounted with your vrtool checkout:
# -v {your_vrtool_checkout_dir}:/usr/src/app
WORKDIR $SRC_ROOT/app
RUN apt-get update
RUN apt-get install -y dotnet-sdk-8.0
# Define the endpoint
CMD [ "/bin/bash" ]