import gym
import yaml
import numpy as np
from argparse import Namespace
from stable_baselines3 import DQN

class F110EnvWrapper(gym.Env):
    def __init__(self, env, conf):
        super(F110EnvWrapper, self).__init__()
        self.env = env
        self.conf = conf

        self.action_mapping = {
            0: np.array([0.0, 5.0]),   # straight, low speed
            1: np.array([-0.2, 5.0]),  # left, low speed
            2: np.array([0.2, 5.0]),   # right, low speed
            3: np.array([0.0, 10.0]),   # straight, high speed
            4: np.array([-0.2, 10.0]),  # left, high speed
            5: np.array([0.2, 10.0]),   # right, high speed
        }

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(1084,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(len(self.action_mapping))

    def reset(self):
        initial_pose = np.array([[self.conf.sx, self.conf.sy, self.conf.stheta]])
        obs, _, _, _ = self.env.reset(poses=initial_pose)
        return self.preprocess(obs)

    def step(self, action):
        continuous_action = np.array([self.action_mapping[action]])
        obs, reward, done, info = self.env.step(continuous_action)
        return self.preprocess(obs), reward, done, info

    def preprocess(self, obs):
        lidar = obs["scans"][0]
        pose = np.array([
            obs["poses_x"][0], obs["poses_y"][0], obs["poses_theta"][0]
        ])
        velocity = obs["linear_vels_x"][0]
        state = np.concatenate([
            np.array([velocity / 20.0]),
            pose / 100.0,
            lidar / 30.0
        ])
        return state.astype(np.float32)

def make_env():
    with open("../configs/config_Spielberg_map.yaml") as file:
        conf_dict = yaml.safe_load(file)
    conf = Namespace(**conf_dict)

    base_env = gym.make(
        "f110_gym:f110-v0",
        map=conf.map_path,
        map_ext=conf.map_ext,
        num_agents=1
    )
    wrapped_env = F110EnvWrapper(base_env, conf)
    return wrapped_env

# Create environment and load trained model
env = make_env()
model = DQN.load("../models/dqn_model")

# Run simulation visually
obs = env.reset()
done = False
total_reward = 0.0

while not done:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(int(action))
    total_reward += reward
    env.env.render()  # This opens the visualization window

print("Total reward:", total_reward)
