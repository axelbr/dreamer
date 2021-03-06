# Dreamer for Autonomous Racing Cars

Implementation of Dreamer for Autonomous Racing Cars.

We propose to learn a policy directly from raw Lidar observations.
The resulting policy has been evaluated on F1tenth tracks and then transfered to real cars.

![Dreamer Austria](doc/austria_view.gif)

The implementation extends the original implementation of [Dreamer](https://github.com/danijar/dreamer).

## Method

We invite the reader to refer to the [Dreamer website](https://danijar.com/project/dreamer/) for the details on the algorithm.

![Dreamer](https://imgur.com/JrXC4rh.png)

From the author:

>Dreamer learns a world model that predicts ahead in a compact feature space.
From imagined feature sequences, it learns a policy and state-value function.
The value gradients are backpropagated through the multi-step predictions to
efficiently learn a long-horizon policy.


## Instructions

This code has been tested on `Ubuntu 18.04` with `Python 3.7`.
 
Get dependencies:

```
pip install --user -r requirements.txt
```

### Training

Train the agent with Lidar reconstruction:

```
python dreamer.py --track columbia --obs_type lidar
```

Train the agent with Occupancy Map reconstruction:
```
python dreamer.py --track columbia --obs_type lidar_occupancy
```

Please, refer to `dreamer.py` for the other command-line arguments.

### Offline Evaluation
The evaluation module runs offline testing of a trained agent (Dreamer, D4PG, MPO, PPO, SAC).

To run evaluation:
```
python evaluations/run_evaluation.py --agent dreamer \
                                     --trained_on austria \
                                     --obs_type lidar \
                                     --checkpoint_dir logs/checkpoints \
                                     --outdir logs/evaluations \
                                     --eval_episodes 10 \
                                     --tracks columbia barcelona \
```
The script will look for all the checkpoints with pattern `logs/checkpoints/austria_dreamer_lidar_*`
The checkpoint format depends on the saving procedure (`pkl`, `zip` or directory).

The results are stored as tensorflow logs.

## Plotting
The plotting module containes several scripts to visualize the results, usually aggregated over multiple experiments.

To plot the learning curves:
```
python plotting/plot_training_curves.py --indir logs/experiments \
                                        --outdir plots/learning_curves \
                                        --methods dreamer mpo \
                                        --tracks austria columbia treitlstrasse_v2 \
                                        --legend
```
It will produce the comparison between Dreamer and MPO on the tracks Austria, Columbia, Treitlstrasse_v2.

To plot the evaluation results:
```
python plotting/plot_test_evaluation.py --indir logs/evaluations \
                                        --outdir plots/evaluation_charts \
                                        --methods dreamer mpo \
                                        --vis_tracks austria columbia treitlstrasse_v2 \
                                        --legend
```
It will produce the bar charts comparing Dreamer and MPO evaluated in Austria, Columbia, Treitlstrasse_v2.


## Instructions with Docker

We also provide an docker image based on `tensorflow:2.3.1-gpu`.
You need `nvidia-docker` to run them, see ![here](https://github.com/NVIDIA/nvidia-docker) for more details.

To build the image:
```  
docker build -t dreamer .
```

To train Dreamer within the container:
```
docker run -u $(id -u):$(id -g) -v $(pwd):/src --gpus all --rm dreamer python dreamer.py --track columbia --steps 1000000
```

