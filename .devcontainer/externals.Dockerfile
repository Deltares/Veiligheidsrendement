# To build this docker (from the root checkout) run:
# `docker build -t vrtool_externals -f .devcontainer/externals.Dockerfile`

# We use a miniforge3 image for consistency with the other development image.
FROM ubuntu:latest

ARG SRC_ROOT="/usr/src"

WORKDIR $SRC_ROOT/app
COPY externals $SRC_ROOT/test_externals
RUN chmod a+x "${SRC_ROOT}/test_externals/DStabilityConsole/D-Stability Console"


# Define the endpoint
CMD [ "/bin/bash" ]