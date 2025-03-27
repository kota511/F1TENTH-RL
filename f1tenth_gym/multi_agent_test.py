import gym
import yaml
import numpy as np
import torch
from argparse import Namespace
from stable_baselines3 import PPO, DDPG, DQN

def preprocess(obs, agent_idx):
    lidar = obs["scans"][agent_idx]
    pose = np.array([
        obs["poses_x"][agent_idx],
        obs["poses_y"][agent_idx],
        obs["poses_theta"][agent_idx]
    ])
    velocity = obs["linear_vels_x"][agent_idx]
    state = np.concatenate([
        np.array([velocity / 20.0]),
        pose / 100.0,
        lidar / 30.0
    ])
    return state.astype(np.float32)

# Make environment
with open("../configs/config_Spielberg_map.yaml") as file:
    conf_dict = yaml.safe_load(file)
conf = Namespace(**conf_dict)
env = gym.make("f110_gym:f110-v0",
               map=conf.map_path,
               map_ext=conf.map_ext,
               num_agents=3)

# Load trained models
ppo_model = PPO.load("../models/ppo_model")
ddpg_model = DDPG.load("../models/ddpg_model")
dqn_model = DQN.load("../models/dqn_model")

# Discrete actions for DQN
discrete_actions = [
    np.array([0.0, 1.0]),
    np.array([-0.2, 1.0]),
    np.array([0.2, 1.0]),
    np.array([0.0, 2.0]),
    np.array([-0.2, 2.0]),
    np.array([0.2, 2.0]),
]

# Initial positions for 3 agents
initial_poses = np.array([[0.0, 0.0, 0.0],
                          [0.0, -1.0, 0.0],
                          [0.0, -2.0, 0.0]])
obs, _, done, _ = env.reset(poses=initial_pose)
done = False

while not done:
    # PPO Agent (agent 0)
    state_ppo = preprocess(obs, 0)
    action_ppo, _ = ppo_model.predict(state_ppo, deterministic=True)

    # DDPG Agent (agent 1)
    state_ddpg = preprocess(obs, 1)
    action_ddpg, _ = ddpg_model.predict(state_ddpg, deterministic=True)

    # DQN Agent (agent 2)
    state_dqn = preprocess(obs, 2)
    action_dqn, _ = dqn_model.predict(state_dqn, deterministic=True)
    action_dqn_continuous = discrete_actions[int(action_dqn)]

    # combine all actions
    actions = np.array([
        action_ppo,
        action_ddpg,
        action_dqn_continuous
    ])

    obs, reward, done, info = env.step(actions)

    # multi-agent environments return done as an array
    if isinstance(done, np.ndarray):
        done = np.any(done)  # end if any agent is done

env.close()
