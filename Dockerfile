# Base Image on DRL image
FROM mitdrl/ubuntu:latest

# Timezone
RUN apt-get update && DEBIAN_FRONTEND="noninteractive" apt-get install -y tzdata
ENV TZ=America/New_York
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install dependencies
RUN apt-get install -y libsndfile1
RUN conda install python=3.6 tensorflow-gpu=2 tqdm
RUN pip install keras-ncp

# Update something to the bashrc (/etc/bashrc_skipper) to customize your shell
RUN pip install pyfiglet
RUN echo -e "alias py='python'" >> /etc/bashrc_skipper

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

# WORKDIR /dreamer

# run dreamer
# ENTRYPOINT ["/usr/bin/python3", "dreamer.py"]

# Switch to src directory
WORKDIR /src

# Copy your code into the docker that is assumed to live in . (on machine)
COPY ./ /src