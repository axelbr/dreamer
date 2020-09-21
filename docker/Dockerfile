FROM tensorflow/tensorflow:2.3.0-gpu AS base
RUN python --version && pip --version && pip list
RUN apt update -y && apt install git xvfb -y
WORKDIR dreamer
COPY . .
RUN pip -m pip install -r requirements.txt
ENTRYPOINT ['xvfb-run -s "-screen 0 1400x900x24" python dreamer.py --logdir /logs/${task} --task ${task}']
CMD /bin/bash