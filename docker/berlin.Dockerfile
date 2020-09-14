FROM tensorflow/tensorflow:2.3.0-gpu AS base
RUN python --version && pip --version && pip list
RUN apt update -y && apt install git xvfb python3.7 -y
WORKDIR dreamer
COPY . .
RUN python3.7 -m pip install --upgrade pip
RUN python3.7 -m pip install -r requirements.txt
ENV task racecar_f1tenth-berlin-two-gui-v0
CMD python3.7 dreamer.py --logdir /logs/2/${task} --task ${task}