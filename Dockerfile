FROM python:3

RUN apt update && apt install --no-install-recommends -y \
    protobuf-compiler \
    && rm -rf /var/lib/apt/lists/*

RUN pip install -U --pre tensorflow

WORKDIR /usr/src/app

RUN git clone --depth 1 https://github.com/hcmcaic/ai-challenge-2020.git

WORKDIR /usr/src/app/ai-challenge-2020

RUN git clone --depth 1 https://github.com/tensorflow/models.git

WORKDIR /usr/src/app/ai-challenge-2020/models/research

RUN protoc object_detection/protos/*.proto --python_out=.
RUN cp object_detection/packages/tf2/setup.py .
RUN pip install .
