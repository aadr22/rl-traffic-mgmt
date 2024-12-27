<img src="docs/_static/logo.png" align="right" width="30%"/>

[![DOI](https://zenodo.org/badge/161216111.svg)](https://zenodo.org/doi/10.5281/zenodo.10869789)
[![tests](https://github.com/LucasAlegre/sumo-rl/actions/workflows/linux-test.yml/badge.svg)](https://github.com/LucasAlegre/sumo-rl/actions/workflows/linux-test.yml)
[![PyPI version](https://badge.fury.io/py/sumo-rl.svg)](https://badge.fury.io/py/sumo-rl)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License](http://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat)](https://github.com/LucasAlegre/sumo-rl/blob/main/LICENSE)

# SUMO-RL

<!-- start intro -->

SUMO-RL provides a simple interface to instantiate Reinforcement Learning (RL) environments with [SUMO](https://github.com/eclipse/sumo) for Traffic Signal Control.

Goals of this repository:
- Provide a simple interface to work with Reinforcement Learning for Traffic Signal Control using SUMO
- Compatibility with gymnasium.Env and popular RL libraries such as [stable-baselines3](https://github.com/DLR-RM/stable-baselines3) and [RLlib](https://docs.ray.io/en/main/rllib.html)
- Easy customisation: state and reward definitions are easily modifiable


# Reinforcement Learning based Smart Traffic Management System using RFID AND Image Detection
Changes from original repository ((https://github.com/LucasAlegre/sumo-rl):
- To compare the efficacy of Q-learning/DQN/SARSA Reinforcement Learning agents on 2-way-single intersection
- To simulate the best performing reinforcement learning algorithm on a 2-way single intersection using Image Detection and RFID-like monitoring systems

The main class is [SumoEnvironment](https://github.com/LucasAlegre/sumo-rl/blob/main/sumo_rl/environment/env.py).
If instantiated with parameter 'single-agent=True', it behaves like a regular [Gymnasium Env](https://github.com/Farama-Foundation/Gymnasium).
For multiagent environments, use [env](https://github.com/LucasAlegre/sumo-rl/blob/main/sumo_rl/environment/env.py) or [parallel_env](https://github.com/LucasAlegre/sumo-rl/blob/main/sumo_rl/environment/env.py) to instantiate a [PettingZoo](https://github.com/PettingZoo-Team/PettingZoo) environment with AEC or Parallel API, respectively.
[TrafficSignal](https://github.com/LucasAlegre/sumo-rl/blob/main/sumo_rl/environment/traffic_signal.py) is responsible for retrieving information and actuating on traffic lights using [TraCI](https://sumo.dlr.de/wiki/TraCI) API.

For more details, check the [documentation online](https://lucasalegre.github.io/sumo-rl/).

<!-- end intro -->

## Install

<!-- start install -->

### Install SUMO latest version:

```bash
sudo add-apt-repository ppa:sumo/stable
sudo apt-get update
sudo apt-get install sumo sumo-tools sumo-doc
```
Don't forget to set SUMO_HOME variable (default sumo installation path is /usr/share/sumo)
```bash
echo 'export SUMO_HOME="/usr/share/sumo"' >> ~/.bashrc
source ~/.bashrc
```
Important: for a huge performance boost (~8x) with Libsumo, you can declare the variable:
```bash
export LIBSUMO_AS_TRACI=1
```
Notice that you will not be able to run with sumo-gui or with multiple simulations in parallel if this is active ([more details](https://sumo.dlr.de/docs/Libsumo.html)).

### Install SUMO-RL, Stable Baselines 3, Gymnasium to run project 

<!--begin install-->

Stable release version is available through pip
```bash
pip install sumo-rl stable_baselines3 gymnasium
```

<!-- end install -->

### Setup and activate virtual environment to run simulation:
<!--begin setup-->
```bash
python -m venv env
# On Windows
env\Scripts\activate
# On Unix/MacOS
source env/bin/activate
```
<!-- end setup -->

## MDP - Observations, Actions and Rewards

### Observation

<!-- start observation -->

The default observation for each traffic signal agent is a vector:
```python
    obs = [phase_one_hot, min_green, lane_1_density,...,lane_n_density, lane_1_queue,...,lane_n_queue]
```
- ```phase_one_hot``` is a one-hot encoded vector indicating the current active green phase
- ```min_green``` is a binary variable indicating whether min_green seconds have already passed in the current phase
- ```lane_i_density``` is the number of vehicles in incoming lane i dividided by the total capacity of the lane
- ```lane_i_queue```is the number of queued (speed below 0.1 m/s) vehicles in incoming lane i divided by the total capacity of the lane

You can define your own observation by implementing a class that inherits from [ObservationFunction](https://github.com/LucasAlegre/sumo-rl/blob/main/sumo_rl/environment/observations.py) and passing it to the environment constructor.

<!-- end observation -->

### Action

<!-- start action -->

The action space is discrete.
Every 'delta_time' seconds, each traffic signal agent can choose the next green phase configuration.

E.g.: In the [2-way single intersection](https://github.com/LucasAlegre/sumo-rl/blob/main/experiments/dqn_2way-single-intersection.py) there are |A| = 4 discrete actions, corresponding to the following green phase configurations:

<p align="center">
<img src="docs/_static/actions.png" align="center" width="75%"/>
</p>

Important: every time a phase change occurs, the next phase is preeceded by a yellow phase lasting ```yellow_time``` seconds.

<!-- end action -->

### Rewards

<!-- start reward -->

The default reward function is the change in cumulative vehicle delay:

<p align="center">
<img src="docs/_static/reward.png" align="center" width="25%"/>
</p>

That is, the reward is how much the total delay (sum of the waiting times of all approaching vehicles) changed in relation to the previous time-step.

You can choose a different reward function (see the ones implemented in [TrafficSignal](https://github.com/LucasAlegre/sumo-rl/blob/main/sumo_rl/environment/traffic_signal.py)) with the parameter `reward_fn` in the [SumoEnvironment](https://github.com/LucasAlegre/sumo-rl/blob/main/sumo_rl/environment/env.py) constructor.

It is also possible to implement your own reward function:

```python
def my_reward_fn(traffic_signal):
    return traffic_signal.get_average_speed()

env = SumoEnvironment(..., reward_fn=my_reward_fn)
```

<!-- end reward -->


### RESCO Benchmarks

In the folder [nets/RESCO](https://github.com/LucasAlegre/sumo-rl/tree/main/sumo_rl/nets/RESCO) you can find the network and route files from [RESCO](https://github.com/jault/RESCO) (Reinforcement Learning Benchmarks for Traffic Signal Control), which was built on top of SUMO-RL. See their [paper](https://people.engr.tamu.edu/guni/Papers/NeurIPS-signals.pdf) for results.

<p align="center">
<img src="sumo_rl/nets/RESCO/maps.png" align="center" width="60%"/>
</p>

### Experiments

Check [experiments](https://github.com/LucasAlegre/sumo-rl/tree/main/experiments) for examples on how to instantiate an environment and train your RL agent.

### [Q-learning algorithm](https://github.com/LucasAlegre/sumo-rl/blob/main/agents/ql_agent.py) in a two-way intersection:
```bash
python experiments/ql_single-intersection.py
```

### [stable-baselines3 DQN algorithm](https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/dqn/dqn.py) in a 2-way single intersection:
Obs: you need to install stable-baselines3 with ```pip install "stable_baselines3[extra]>=2.0.0a9"``` for [Gymnasium compatibility](https://stable-baselines3.readthedocs.io/en/master/guide/install.html).
```bash
python experiments/dqn_2way-single-intersection.py
```

### [SARSA algorithm](https://github.com/LucasAlegre/sumo-rl/blob/main/agents/ql_agent.py) in a two-way single intersection:
```bash
python experiments/sarsa_2way-single-intersection.py
```
### [Q-learning algorithm + simulation with dummy RFID and Image Detection monitoring] in a two-way single intersection:
```bash
python experiments/ql_rfid_monitoring.py -gui -s 100000
```

