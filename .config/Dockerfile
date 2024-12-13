# To build this docker run:
# `docker build -t vrtool_core_dev`
# To run the docker for pytest:
# `docker run -v {your_vrtool_checkout_dir}:/usr/src/app/ -v {vrtool_env_dir}:/usr/src/.env -t vrtool_core_dev`
# `poetry install`
# `poetry run pytest -m {your_pytest_arguments}`

FROM condaforge/miniforge3:latest

ENV TZ=Europe/Amsterdam \
    DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install libgl1 -y && apt-get install libicu66 -y


ARG SRC_ROOT="/usr/src"

# For persisting the environment remember to mount it as well:
# -v {your_local_tmp_dir}:/usr/src/.env
ARG CONDA_ENV="${SRC_ROOT}/.env"

# Install conda environment
# This location needs to be mounted with your vrtool checkout:
# -v {your_vrtool_checkout_dir}:/usr/src/app
WORKDIR $SRC_ROOT/app
COPY environment.yml $SRC_ROOT/.config/environment.yml
COPY externals $SRC_ROOT/test_externals

# Give execution rights to DStability
# RUN chmod a+x "${SRC_ROOT}/test_externals/DStabilityConsole/D-Stability Console.exe"

RUN conda init bash \
    && . ~/.bashrc \
    && conda env create -f $SRC_ROOT/.config/environment.yml -p $CONDA_ENV

# Activate conda environment
ENV PATH $CONDA_ENV/bin:$PATH
RUN conda activate ${CONDA_ENV} & poetry config virtualenvs.create false
RUN echo conda init && echo "conda activate ${CONDA_ENV}" >>  ~/.bashrc
# This ensures it's available also for `sh` instead of `bash`.
RUN echo "conda activate ${CONDA_ENV}" > ~/.profile

# Define the endpoint
CMD [ "/bin/bash" ]