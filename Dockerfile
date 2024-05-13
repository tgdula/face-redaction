# FROM python:3.11-slim

#FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04
FROM nvcr.io/nvidia/ai-workbench/python-cuda117:1.0.3
ENV DEBIAN_FRONTEND=noninteractive

# the following line is needed in order for the build to succeed due to some outdated stuff in the docker image
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub

# Update package lists and install necessary tools
RUN apt-get update && \
    apt-get install -y software-properties-common

# Add deadsnakes PPA to get Python 3.11
RUN add-apt-repository ppa:deadsnakes/ppa

RUN apt update --fix-missing && \
        apt install python3.11 python3.11-distutils -y && \
        apt install python3-pip -y && \
        ln -sf /usr/bin/python3.11 /usr/bin/python && \
        ln -sf /usr/bin/pip3 /usr/bin/pip

# install latest pip
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11

RUN apt-get update -y && apt-get install -y --no-install-recommends build-essential gcc \
                                        libsndfile1



RUN pip install poetry==1.8.0

RUN poetry config virtualenvs.create false

WORKDIR /code

COPY ./pyproject.toml ./README.md ./poetry.lock* ./

# RUN apt-get update && apt-get install -y build-essential cmake
# from: https://github.com/maeharin/docker-python3-dlib-opencv/blob/master/Dockerfile
RUN apt-get -y update
# for dlib
RUN apt-get install -y build-essential cmake
# for opencv
RUN apt-get install -y libopencv-dev

# HINT: doesn't work
#RUN poetry install  --no-interaction --no-ansi
RUN pip install --upgrade pip

# Install dependencies
RUN poetry config virtualenvs.create false
RUN poetry install --no-interaction --no-dev --no-ansi

COPY ./face_redaction ./face_redaction

RUN poetry install --no-interaction --no-ansi

ENTRYPOINT ["poetry", "run", "redact"]