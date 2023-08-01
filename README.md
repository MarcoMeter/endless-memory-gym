[[Paper](https://openreview.net/forum?id=jHc8dCx6DDr)] [[Installation](#installation)]  [[Usage](#usage)] [[Mortar Mayhem](#mortar-mayhem)] [[Endless Mortar Mayhem](#endless-mortar-mayhem)] [[Mystery Path](#mystery-path)] [[Endless Mystery Path](#enndless-mystery-path)] [[Searing Spotlights](#searing-spotlights)] [[Endless Searing Spotlights](#endless-searing-spotlights)] [[Training](#training)]

# Memory Gym: Partially Observable Challenges to Memory-Based Agents in Endless Episodes

<style>
  table {
    border-collapse: collapse;
    margin: 0 auto; /* Added margin for center alignment */
  }
  td {
    text-align: center;
    vertical-align: middle;
    padding: 5px;
    border: none;
  }
</style>

<table align="center">
  <tr>
    <td></td>
    <td>Endless Mortar Mayhem</td>
    <td>Endless Mystery Path</td>
    <td>Endless Searing Spotlights</td>
  </tr>
  <tr>
    <td>Agent Observation</td>
    <td><img src="docs/assets/emm_0.gif" width=180></td>
    <td><img src="docs/assets/emp_0.gif" width=180></td>
    <td><img src="docs/assets/ess_0.gif" width=180></td>
  </tr>
  <tr>
    <td>Ground Truth</td>
    <td><img src="docs/assets/emm_0_gt.gif" width=180></td>
    <td><img src="docs/assets/emp_0.gif" width=180></td>
    <td><img src="docs/assets/ess_0.gif" width=180></td>
  </tr>
</table>


Memory Gym features the environments **Mortar Mayhem**, **Mystery Path**, and **Searing Spotlights** that are inspired by some mini games of [Pummel Party](http://rebuiltgames.com/). These environments shall benchmark an agent's memory to
- memorize events across long sequences,
- generalize,
- and be robust to noise.

Especially, these environments feature endless task variants (see the gifs above). As the agent's policy improves, the task goes on. The traveling game "I packed my bag ..." inspired this dynamic concept, which allows for examining levels of effectinvess instead of just sample efficiency.

## Citation

```bibtex
@inproceedings{pleines2023memory,
title={Memory Gym: Partially Observable Challenges to Memory-Based Agents},
author={Marco Pleines and Matthias Pallasch and Frank Zimmer and Mike Preuss},
booktitle={International Conference on Learning Representations},
year={2023},
url={https://openreview.net/forum?id=jHc8dCx6DDr}
}
```

## Installation

Major dependencies:
- gymnasium==0.29.0
- PyGame==2.4.0

```console
conda create -n memory-gym python=3.11 --yes
conda activate memory-gym
pip install memory-gym
```

or

```console
conda create -n memory-gym python=3.11 --yes
conda activate memory-gym
git clone https://github.com/MarcoMeter/drl-memory-gym.git
cd drl-memory-gym
pip install -e .
```


## Usage
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1LjlUOEer8vjGrz0rLM8pP5UyeNCsURkY?usp=sharing)

Executing the environment using random actions:
```python
import memory_gym
import gymnasium as gym

env = gym.make("Endless-SearingSpotlights-v0")
# env = gym.make("SearingSpotlights-v0")
# env = gym.make("Endless-MortarMayhem-v0")
# env = gym.make("MortarMayhem-v0")
# env = gym.make("MortarMayhem-Grid-v0")
# env = gym.make("MortarMayhemB-v0")
# env = gym.make("MortarMayhemB-Grid-v0")
# env = gym.make("Endless-MysteryPath-v0")
# env = gym.make("MysteryPath-v0")
# env = gym.make("MysteryPath-Grid-v0")

# Pass reset parameters to the environment
options = {"agent_scale": 0.25}

obs, info = env.reset(seed=1, options=options)
done = False
while not done:
    obs, reward, done, truncation, info = env.step(env.action_space.sample())

print(info)
```

Manually play the environments using the console scripts (works only using an anaconda environment):
```console
mortar_mayhem
# MMAct
mortar_mayhem_b
# MMGrid
mortar_mayhem_grid
# MMAct Grid
mortar_mayhem_b_grid
mystery_path
mystery_path_grid
searing_spotlights

# Endless Environments
endless_mortar_mayhem
endless_mystery_path
endless_searing_spotlights
```

You can also execute the python scripts directly, for example:
```
python ./memory_gym/mortar_mayhem.py
```

Controls:
- WASD or Arrow Keys to move or rotate
- Page Up / Page Down to increment / decrement environment seeds

## Mortar Mayhem

<table align="center">
  <tr>
    <td>Agent Observation</td>
    <td>Ground Truth</td>
  </tr>
  <tr>
    <td><img src="docs/assets/mortar_mayhem_0.gif" width=180></td>
    <td><img src="docs/assets/mortar_mayhem_0_gt.gif" width=180></td>
  </tr>
</table>

Mortar Mayhem challenges the agent with a sequence of commands that the agent has to memorize and execute in the right order. During the beginning of the episode, each command is visualized one by one. Mortar Mayhem can be reduced to solely executing commands. In this case, the command sequence is always available as vector observation (one-hot encoded) and, therefore, is not visualized.

The max length of an episode can be calculated as follows:

```
max episode length = (command_show_duration + command_show_delay) * command_count + (explosion_delay + explosion_duration) * command_count - 2
```

![Mortar Mayhem Environment](/docs/assets/mm.jpg)

### Reset Parameters

| Parameter              | Default | Description                                                                                                                                       |
|------------------------|--------:|---------------------------------------------------------------------------------------------------------------------------------------------------|
| agent_scale            |    0.25 | The dimensions of the agent.                                                                                                                      |
| agent_speed            |     2.5 | The speed of the agent.                                                                                                                           |
| arena_size             |       5 | The grid dimension of the arena (min: 2, max: 6)                                                                                                  |
| allowed_commands       |       9 | Available commands: right, down, left, up, stay, right down, right up, left down, left up. If set to five, the first five commands are available. |
| command_count          |     [5] | The number of commands that are asked to be executed by the agent. This is a list that the environment samples from.                              |
| command_show_duration  |     [3] | The number of steps that one command is shown. This is a list that the environment samples from.                                                  |
| command_show_delay     |     [1] | The number of steps between showing one command. This is a list that the environment samples from.                                                |
| explosion_duration     |     [6] | The number of steps that an agent has to stay on the commanded tile. This is a list that the environment samples form.                            |
| explosion_delay        |    [18] | The entire duration in steps that the agent has to execute the current command. This is a list that the environments samples from.                |
| visual_feedback        |    True | Whether to turn off the visualization of the feedback. Upon command evaluation, the wrong tiles are rendered red.                                 |
| reward_command_failure |     0.0 | What reward to signal upon failing at the current command.                                                                                        |
| reward_command_success |     0.1 | What reward to signal upon succeeding at the current command.                                                                                       |
| reward_episode_success |     0.0 | What reward to signal if the entire command sequence is successfully solved by the agent.                                                         |

## Endless Mortar Mayhem

## Mystery Path

<table align="center">
  <tr>
    <td>Agent Observation</td>
    <td>Ground Truth</td>
  </tr>
  <tr>
    <td><img src="docs/assets/mystery_path_0.gif" width=180></td>
    <td><img src="docs/assets/mystery_path_0_gt.gif" width=180></td>
  </tr>
</table>

Mystery Path procedurally generates an invisible path for the agent to cross from the origin to the goal. Per default, only the origin of the path is visible. Upon falling off the path, the agent has to restart from the origin. Note that the episode is not terminated by falling off. Hence, the agent has to memorize where it fell off and where it did not.

![Mystery Path Environment](/docs/assets/mp.jpg)

### Reset Parameters

| Parameter              |      Default | Explanation                                                                                                                 |
|------------------------|-------------:|-----------------------------------------------------------------------------------------------------------------------------|
| max_steps              |          512 | The maximum number of steps for the agent to play one episode.                                                              |
| agent_scale            |         0.25 | The dimensions of the agent.                                                                                                |
| agent_speed            |          2.5 | The speed of the agent.                                                                                                     |
| cardinal_origin_choice | [0, 1, 2, 3] | Allowed cardinal directions for the path generation to place the origin. This is a list that the environment samples from.  |
| show_origin            |         True | Whether to hide or show the origin tile of the generated path.                                                              |
| show_goal              |        False | Whether to hide or show the goal tile of the generated path.                                                                |
| visual_feedback        |         True | Whether to visualize that the agent is off the path. A red cross is rendered on top of the agent.                           |
| reward_goal            |          1.0 | What reward to signal when reaching the goal tile.                                                                          |
| reward_fall_off        |          0.0 | What reward to signal when falling off.                                                                                     |
| reward_path_progress   |          0.0 | What reward to signal when making progress on the path. This is only signaled for reaching another tile for the first time. |
| reward_step            |          0.0 | What reward to signal for each step.                                                                                        |

## Endless Mystery Path

<p align=center>
<img src="docs/assets/emp_path.png" width=420>
</p>

## Searing Spotlights

<table align="center">
  <tr>
    <td>Agent Observation</td>
    <td>Ground Truth</td>
  </tr>
  <tr>
    <td><img src="docs/assets/searing_spotlights_0.gif" width=180></td>
    <td><img src="docs/assets/searing_spotlights_0_gt.gif" width=180></td>
  </tr>
</table>

Searing Spotlights is a pitch black surrounding to the agent. The environment is initially fully observable but the light is dimmed untill off during the first few frames. Only randomly moving spotlights unveil information on the environment's ground truth, while posing a threat to the agent. If spotted by spotlight, the agent looses health points. While the agent must avoid closing in spotlights, it further has to collect coins. After collecting all coins, the agent has to take the environment's exit.

![Searing Spotlights Environment](/docs/assets/spots.jpg)

### Reset Parameters

| Parameter                | Default | Explanation                                                                                                     |
|--------------------------|--------:|-----------------------------------------------------------------------------------------------------------------|
| max_steps                |     512 | The maximum number of steps for the agent to play one episode.                                                  |
| agent_scale              |    0.25 | The dimensions of the agent.                                                                                    |
| agent_speed              |     2.5 | The speed of the agent.                                                                                         |
| agent_health             |     100 | The initial health points of the agent.                                                                         |
| agent_visible            |   False | Whether to make the agent permanently visible.                                                                  |
| sample_agent_position    |    True | Whether to hide or show the goal tile of the generated path.                                                    |
| num_coins                |     [1] | The number of coins that are spawned. This is a list that the environment samples from.                         |
| coin_scale               |   0.375 | The scale of the coins.                                                                                         |
| coins_visible            |   False | Whether to make the coins permanently visible.                                                                  |
| use_exit                 |    True | Whether to spawn and use the exit task. The exit is accessible by the agent after collecting all coins.         |
| exit_scale               |     0.0 | The scale of the exit.                                                                                          |
| exit_visible             | False   | Whether to make the exit permanently visible.                                                                   |
| initial_spawns           | 4       | The number of spotlights that are initially spawned.                                                            |
| num_spawns               | 30      | The number of spotlights that are to be spawned.                                                                |
| initial_spawn_interval   | 30      | The number of steps until the next spotlight is spawned.                                                        |
| spawn_interval_threshold | 10      | The spawn interval is decayed until reaching this lower threshold.                                              |
| spawn_interval_decay     | 0.95    | The decay rate of the spotlight spawn interval.                                                                 |
| spot_min_radius          | 7.5     | The minimum radius of the spotlights. The radius is sampled from the range min to max.                          |
| spot_max_radius          | 13.75   | The maximum radius of the spotlights. The radius is sampled from the range min to max.                          |
| spot_min_speed           | 0.0025  | The minimum speed of the spotlights. The speed is sampled from the range min to max.                            |
| spot_max_speed           | 0.0075  | The maximum speed of the spotlights. The speed is sampled from the range min to max.                            |
| spot_damage              | 1.0     | Damage per step while the agent is spotted by one spotlight.                                                    |
| light_dim_off_duration   | 6       | The number of steps to dim off the global light.                                                                |
| light_threshold          | 255     | The threshold for dimming the global light. A value of 255 indicates that the light will dimmed of completely.  |
| visual_feedback          | True    | Whether to render the tiled background red if the agent is spotted.                                             |
| black_background         | False   | Whether to render the environments background black, while the spotlights are rendered as white circumferences. |
| hide_chessboard          | False   | Whether to hide the chessboard background. This renders the background of the environment white.                           |
| reward_inside_spotlight  | 0.0     | What reward to signal for each step while being inside a spotlight.                                             |
| reward_outside_spotlight | 0.0     | What reward to signal for each step while being outside of a spotlight.                                         |
| reward_death             | 0.0     | What reward to signal upon losing all health points.                                                            |
| reward_exit              | 1.0     | What reward to signal after successfully using the exit.                                                        |
| reward_max_steps         | 0.0     | What reward to signal if max steps is reached.                                                                  |
| reward_coin              | 0.25    | What reward to signal upon collecting one coin.                                                                 |

## Endless Searing Spotlights


## Training

Baseline results are avaible via these repositories.

[Recurrence + PPO](https://github.com/MarcoMeter/recurrent-ppo-truncated-bptt)

[Gated TransformerXL + PPO](https://github.com/MarcoMeter/episodic-transformer-memory-ppo)

## Changelog

v1.0.0

Improvements
- All environment concepts are extrapolated to endless episodes!
    - Endless Mortar Mayhem
    - Endless Mystery Path
    - Endless Searing Spotlights
- Improved simulation speed by using already rotated sprites and not rotating the character's surface every frame
- Mystery Path: A* obstacle walls are also placed now on the environments boundary to mitigate trivial paths
- All endless environments feature a ground truth space. As specified by this space ground truth information is added to the info dictionary
- Searing Spotlights may also visualize whether a positive reward was signaled on the previous frame

Breaking Changes
- Refactored the info key "exit_success" in Searing Spotlights to "success"

Bug Fixes
- Fixed the speed of character controller, because moving downwards was slower than moving upwards due to float truncation