# Dream to Control

Fast and simple implementation of the Dreamer agent in TensorFlow 2.

<img width="100%" src="https://imgur.com/x4NUHXl.gif">

If you find this code useful, please reference in your paper:

```
@article{hafner2019dreamer,
  title={Dream to Control: Learning Behaviors by Latent Imagination},
  author={Hafner, Danijar and Lillicrap, Timothy and Ba, Jimmy and Norouzi, Mohammad},
  journal={arXiv preprint arXiv:1912.01603},
  year={2019}
}
```

## Method

![Dreamer](https://imgur.com/JrXC4rh.png)

Dreamer learns a world model that predicts ahead in a compact feature space.
From imagined feature sequences, it learns a policy and state-value function.
The value gradients are backpropagated through the multi-step predictions to
efficiently learn a long-horizon policy.

- [Project website][website]
- [Research paper][paper]
- [Official implementation][code] (TensorFlow 1)

[website]: https://danijar.com/dreamer
[paper]: https://arxiv.org/pdf/1912.01603.pdf
[code]: https://github.com/google-research/dreamer

## Run with Docker

The image is based on `tensorflow:2.3.1-gpu`, you need `nvidia-docker` to run them.
See ![here](https://github.com/NVIDIA/nvidia-docker) for more details.

To build the image:
```  
docker build -t dreamer:1.0 .
```

To run the container:
```
docker run
    -v $(pwd)/logs:/dreamer/logs
    -v $(pwd)/scenarios:/dreamer/scenarios
    --name dreamer
    --runtime=nvidia
    --rm
    dreamer:1.0 
    --task racecar_columbia --logdir logs/prova_docker --prefill 100 --steps 1000
```

**Note:** 
- You can attach the dreamer's input parameter at the end of the command.
- The mount point `logs` allows you to write the results on the host machine.
- The host can monitor the experiment by running `tensorboard --logdir logs/`
- The mount point `scenarios` allows the user to fast-prototype new race scenarios. 

## Instructions

Get dependencies:

```
pip3 install --user tensorflow-gpu==2.1.0
pip3 install --user tensorflow_probability
pip3 install --user git+git://github.com/deepmind/dm_control.git
pip3 install --user pandas
pip3 install --user matplotlib
```

Train the agent:

```
python3 dreamer.py --logdir ./logdir/dmc_walker_walk/dreamer/1 --task dmc_walker_walk
```

Generate plots:

```
python3 plotting.py --indir ./logdir --outdir ./plots --xaxis step --yaxis test/return --bins 3e4
```

Graphs and GIFs:

```
tensorboard --logdir ./logdir
```
