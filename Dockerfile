FROM tensorflow/tensorflow:2.3.1-gpu AS base

# update
RUN python --version && pip --version && pip list
RUN apt update -y && apt install git wget unzip -y

# install requirements
WORKDIR /build
COPY requirements.txt /build
RUN pip install -r requirements.txt

# download track maps
WORKDIR /build/src/racecar-gym/models/scenes
RUN wget https://github.com/axelbr/racecar_gym/releases/download/tracks-v1.0.0/all.zip && unzip all.zip

WORKDIR /dreamer
COPY . /build

# run dreamer
ENTRYPOINT ["/usr/bin/python3", "dreamer.py"]