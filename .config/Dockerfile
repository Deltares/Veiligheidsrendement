FROM python:3.10

RUN apt-get update && apt-get install libgl1 -y

ARG SRC_ROOT="/usr/src"

# Install conda environment
# This location needs to be mounted with your vrtool checout:
# -v {your_vrtool_checkout_dir}:/usr/src/app
WORKDIR $SRC_ROOT/app

RUN pip install poetry

# Define the endpoint
CMD [ "/bin/bash" ]