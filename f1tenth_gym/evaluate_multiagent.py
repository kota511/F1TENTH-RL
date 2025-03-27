import gym
import yaml
import numpy as np
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

# Load map config
with open("../configs/config_Spielberg_map.yaml") as file:
    conf_dict = yaml.safe_load(file)
conf = Namespace(**conf_dict)

# Environment initialization
num_agents = 3
env = gym.make("f110_gym:f110-v0",
               map=conf.map_path,
               map_ext=conf.map_ext,
               num_agents=3)

# Load pre-trained models explicitly
ppo_model = PPO.load("../models/ppo_model")
ddpg_model = DDPG.load("../models/ddpg_model")
dqn_model = DQN.load("../models/dqn_model")

# Define discrete actions for DQN clearly
discrete_actions = [
    np.array([0.0, 1.0]),
    np.array([-0.2, 1.0]),
    np.array([0.2, 1.0]),
    np.array([0.0, 2.0]),
    np.array([-0.2, 2.0]),
    np.array([0.2, 2.0]),
]

initial_pose = np.array([
    [conf.sx, conf.sy, conf.stheta],
    [conf.sx, conf.sy - 1.0, conf.stheta],
    [conf.sx, conf.sy - 2.0, conf.stheta]
])

obs, _, done, _ = env.reset(poses=initial_pose)
done = [False] * 3

while not any(done):
    # PPO (agent 0)
    action_ppo, _ = ppo_model.predict(preprocess(obs, 0), deterministic=True)

    # DDPG Agent (agent 1)
    state_ddpg = preprocess(obs, 1)
    action_ddpg, _ = ddpg_model.predict(state_ddpg, deterministic=True)

    # DQN Agent (agent 2)
    state_dqn = preprocess(obs, 2)
    action_dqn, _ = dqn_model.predict(state_dqn, deterministic=True)
    action_dqn_continuous = discrete_actions[int(action_dqn)]

    # Combine all actions explicitly
    actions = np.array([
        action_ppo,
        action_ddpg,
        action_dqn_continuous
    ])

    # Execute all actions
    obs, rewards, done, info = env.step(actions)

    # Check if any agent is done
    if any(done):
        break

env.close()
