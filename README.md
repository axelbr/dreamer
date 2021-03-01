# Dreamer for Autonomous Racing Cars

Implementation of Dreamer for Autonomous Racing Cars.

We propose to learn a policy directly from raw Lidar observations.
The resulting policy has been evaluated on F1tenth tracks and then transfered to real cars.

![Dreamer Austria](doc/austria_view.gif)

The implementation extends the original implementation of [Dreamer](https://github.com/danijar/dreamer).

## Method

We invite the reader to refer to the [Dreamer website](https://danijar.com/project/dreamer/) for the details on methodology.

![Dreamer](https://imgur.com/JrXC4rh.png)

From the author:

>Dreamer learns a world model that predicts ahead in a compact feature space.
From imagined feature sequences, it learns a policy and state-value function.
The value gradients are backpropagated through the multi-step predictions to
efficiently learn a long-horizon policy.


## Instructions

Get dependencies:

```
pip3 install --user -r requirements.txt
```

### Training

Train the agent with Lidar reconstruction:

```
python3 dreamer.py --track columbia --obs_type lidar
```

Train the agent with Occupancy Map reconstruction:
```
python3 dreamer.py --track columbia --obs_type lidar_occupancy
```

Please, refer to `dreamer.py` for the other command-line arguments.

### Offline Evaluation
The evaluation module runs offline testing of a trained agent (Dreamer, D4PG, MPO, PPO, SAC) or a programmed agent (Follow The Gap).

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

The results are stored as tensorflow logs.

# Plotting
TODO


## Instructions with Docker

We also provide an docker image based on `tensorflow:2.3.1-gpu`.
You need `nvidia-docker` to run them, see ![here](https://github.com/NVIDIA/nvidia-docker) for more details.

To build the image:
```  
docker build -t dreamer .
```

To train Dreamer within the container:
```
docker run    
    -u $(id -u):$(id -g) 
    -v $(pwd):/src    
    --gpus all
    --rm dreamer
    python dreamer.py --track columbia --steps 1000000
```

