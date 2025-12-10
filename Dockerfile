# To build this docker run:
# `docker build -t vrtool`
# To run this docker:
# `docker run -it 
# -v <local_path_to_externals>:.
# --rm vrtool run_full
# <config_file_name> --externals /app/externals`
# This will mount the local externals directory to /app/externals in the container.
FROM containers.deltares.nl/gfs-dev/vrtool_externals:latest as externals

WORKDIR /usr/src/

FROM python:3.12

RUN apt-get update

# Copy the directories with the local vrtool.
WORKDIR /app
COPY README.md LICENSE pyproject.toml /app/
COPY vrtool /app/vrtool
COPY scripts /app/scripts

# Install koswat and its dependencies.
RUN pip install --upgrade pip && pip install /app

COPY --from=externals /usr/src/test_externals /app/externals

# Set the entrypoint to run vrtool as a module.
ENTRYPOINT ["python", "-m", "vrtool", "--externals", "/app/externals"]