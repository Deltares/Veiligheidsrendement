# To build this docker (from the root checkout) run:
# `docker build -t vrtool_dev .devcontainer`
# To run the docker for pytest:
# `docker run -v {your_vrtool_checkout_dir}:/usr/src/app/ -v {vrtool_env_dir}:/usr/src/.env -t vrtool_dev`
# `poetry install`
# `poetry run pytest -m {your_pytest_arguments}`

# FROM condaforge/miniforge3:latest
FROM containers.deltares.nl/gfs-dev/vrtool_externals:latest AS vrtool_externals-env
FROM mcr.microsoft.com/dotnet/sdk:8.0 AS dotnet-env
# This image overlay was the best way I found to run (linux) .Net applications within
# a miniforge container.

ARG SRC_ROOT="/usr/src"
ARG SRC_EXTERNALS="${SRC_ROOT}/test_externals"

# Get Miniforge and make it the main Python interpreter
ENV CONDA_DIR=/opt/conda
COPY --from=vrtool_externals-env /opt/conda ${CONDA_DIR}
COPY --from=vrtool_externals-env ${SRC_EXTERNALS} ${SRC_EXTERNALS}
ENV PATH=${CONDA_DIR}/bin:${PATH}

RUN echo ". ${CONDA_DIR}/etc/profile.d/conda.sh && conda activate base" >> /etc/skel/.bashrc
RUN echo ". ${CONDA_DIR}/etc/profile.d/conda.sh && conda activate base" >> ~/.bashr

# Install additional dependencies.
ENV TZ=Europe/Amsterdam \
    DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install libgl1 -y 
# Unable to locate libicu66
# && apt-get install libicu66 -y


# For persisting the environment remember to mount it as well:
# -v {your_local_tmp_dir}:/usr/src/.env
ARG CONDA_ENV="${SRC_ROOT}/.env"

# Install conda environment
# This location needs to be mounted with your vrtool checkout:
# -v {your_vrtool_checkout_dir}:/usr/src/app
WORKDIR $SRC_ROOT/app
COPY environment.yml $SRC_ROOT/.devcontainer/environment.yml
# COPY externals $SRC_ROOT/test_externals

# Give execution rights to DStability
# RUN chmod a+x "${SRC_ROOT}/test_externals/DStabilityConsole/D-Stability Console"

# Set conda environment
RUN conda init bash \
    && . ~/.bashrc \
    && conda env create -f $SRC_ROOT/.devcontainer/environment.yml -p $CONDA_ENV

   
# Activate conda environment
ENV PATH=$CONDA_ENV/bin:$PATH
RUN conda init & conda activate ${CONDA_ENV} & poetry config virtualenvs.create false
RUN echo conda init && echo "conda activate ${CONDA_ENV}" >>  ~/.bashrc
# This ensures it's available also for `sh` instead of `bash`.
RUN echo "conda activate ${CONDA_ENV}" > ~/.profile

# Define the endpoint
CMD [ "/bin/bash" ]