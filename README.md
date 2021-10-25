# Reinforcement Learning Story

Repository with homework assignment for the lecture about applying reinforcement learning in production.
> Reinforcement Learning for a Walk-Through Trajectory in a Dynamic Environment

Homework task to train agent for exploring a multi-room zone with a limited view sensor.

## Solution

<p align="center">
    <img src="https://github.com/SpirinEgor/rl_story/blob/master/examples/example-0.gif" alt="Example of work"/>
</p>

Solution based on the [PPO](https://arxiv.org/abs/1707.06347) algorithm.
It is a classical RL approach that is often chosen as baseline.

Reward for this task is a combination of explored cells and walking to a new cell:

<p align="center">
<img src="https://latex.codecogs.com/png.latex?\dpi{100}&space;\fn_phv&space;R&space;=&space;\frac{\text{explored}}{\text{viewed&space;cells}}&space;&plus;&space;1_{\text{is&space;new&space;cell}}," title="R = \frac{\text{explored}}{\text{viewed cells}} + 1_{\text{is new cell}}," />
</p>

where `explored` is a number of new explored cells,
`viewed cells` is a total number of already viewed cells, and
`is new cell` is an indicator, whether agent steps on a new cell.

Therefore, this reward not only directs the agent to increase viewed cells on each step,
but enforces it to visit new cells.
This may help an agent to plan the route,
e.g. move directly to a new place rather than small steps in an already known area with a small fraction of explored cells.

## Replication

To replicate solution, first, install all necessary dependencies in your virtual environment.
It is better to use `Python==3.9` and [install torch manually](https://pytorch.org/get-started/locally/) with GPU support.
```shell
pip install -r requirements.txt
```

### Training

To configure training specify YAML config,
examples of possible configurations are stored in the [config directory](configs).

To start training simply use [`train.py`](src/train.py) script:
```shell
PYTHONPATH="." python src/train.py $PATH_TO_YAML
```

It may ask to log in into [wandb.ai](https://wandb.ai/) account for proper logging during training.
It will store training config as well as all checkpoints, intermediate metrics, and evaluation gifs.

### Evaluation

To evaluate agent and create gifs with its exploration use [`evaluate.py`](src/evaluate.py) script:
```shell
PYTHONPAHT="." python src/evaluate.py $PATH_TO_YAML $PATH_TO_CHECKPOINT $PATH_TO_OUTPUT_DIR --n-samples 5
```

## Examples

Train logs are stored in [WandB project](https://wandb.ai/voudy/RL-vacuum).
All examples obtained from `checkpoint-20` in [`dainty-frog-2` run](https://wandb.ai/voudy/RL-vacuum/runs/1o8b1ott).

<p align="center">
    <img src="https://github.com/SpirinEgor/rl_story/blob/master/examples/example-0.gif" alt="Example 0"/>
    <br>
    <img src="https://github.com/SpirinEgor/rl_story/blob/master/examples/example-1.gif" alt="Example 1"/>
    <br>
    <img src="https://github.com/SpirinEgor/rl_story/blob/master/examples/example-2.gif" alt="Example 2"/>
    <br>
    <img src="https://github.com/SpirinEgor/rl_story/blob/master/examples/example-3.gif" alt="Example 3"/>
    <br>
    <img src="https://github.com/SpirinEgor/rl_story/blob/master/examples/example-4.gif" alt="Example 4"/>
</p>

## Future plans

Possible future work:
- Improve code base by adding metrics, i.e. repetition rate
- Benchmark other RL algorithms, e.g. [DDQN](https://arxiv.org/abs/1509.06461)
- Research multi-agent solution
