# To build this docker run:
# `docker build -t vrtool`

FROM python:3.12

RUN apt-get update

# Copy the directories with the local vrtool.
WORKDIR /vrtool_src
COPY README.md LICENSE pyproject.toml poetry.lock /vrtool_src/
COPY vrtool /vrtool_src/vrtool

# Install the required packages
RUN pip install poetry
RUN poetry config virtualenvs.create false
RUN poetry install --without dev,test
RUN apt-get clean autoclean

# Define the endpoint
CMD ["python3"]