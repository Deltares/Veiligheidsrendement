# To build this docker run:
# `docker build -t vrtool`

FROM python:3.12

RUN apt-get update

# Copy the directories with the local vrtool.
WORKDIR /app
COPY README.md LICENSE pyproject.toml /app/
COPY vrtool /app/vrtool

# Install koswat and its dependencies.
RUN pip install --upgrade pip && pip install /app

# Set the entrypoint to run vrtool as a module.
ENTRYPOINT ["python", "-m", "vrtool"]