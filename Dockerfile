FROM python:3

RUN apt update && apt install --no-install-recommends -y \
    protobuf-compiler \
    wget \
    tar \
    && rm -rf /var/lib/apt/lists/*

RUN pip install -U --pre tensorflow

WORKDIR /usr/src/app

RUN git clone --depth 1 https://github.com/hcmcaic/ai-challenge-2020.git

WORKDIR /usr/src/app/ai-challenge-2020/solution_baseline

# Clone the tensorflow models repository
RUN git clone --depth 1 https://github.com/tensorflow/models.git

# Install the Object Detection API
WORKDIR /usr/src/app/ai-challenge-2020/solution_baseline/models/research

RUN protoc object_detection/protos/*.proto --python_out=.
RUN cp object_detection/packages/tf2/setup.py .
RUN pip install .

# Install the requirements package for SORT source code
RUN pip install filterpy scikit-image lap

WORKDIR /usr/src/app/ai-challenge-2020/solution_baseline
COPY app.py .
RUN wget http://download.tensorflow.org/models/object_detection/tf2/20200711/centernet_hg104_512x512_coco17_tpu-8.tar.gz
RUN tar -xf centernet_hg104_512x512_coco17_tpu-8.tar.gz

RUN python app.py
