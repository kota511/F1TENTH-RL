import gym
import yaml
import numpy as np
from argparse import Namespace
from stable_baselines3 import DDPG
from stable_baselines3.common.vec_env import DummyVecEnv

class F110EnvWrapper(gym.Env):
    def __init__(self, env, conf):
        super(F110EnvWrapper, self).__init__()
        self.env = env
        self.conf = conf
        
        # Correctly defined observation space
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(1084,), dtype=np.float32
        )
        # Correctly defined action space for continuous control
        self.action_space = gym.spaces.Box(
            low=np.array([-0.4, 0.0]),
            high=np.array([0.4, 5.0]),
            dtype=np.float32
        )

    def reset(self):
        initial_pose = np.array([[self.conf.sx, self.conf.sy, self.conf.stheta]])
        obs, _, _, _ = self.env.reset(poses=initial_pose)
        return self.preprocess(obs)

    def step(self, action):
        action = np.array([action])  # Ensure correct shape (1,2)
        obs, reward, done, info = self.env.step(action)
        return self.preprocess(obs), reward, done, {}

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

env = DummyVecEnv([make_env])

# Define noise for exploration explicitly
from stable_baselines3.common.noise import NormalActionNoise
action_noise = NormalActionNoise(mean=np.zeros(2), sigma=0.1 * np.ones(2))

model = DDPG(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log="./ddpg_f1tenth_logs/",
    batch_size=64,
    gamma=0.995,
    learning_rate=0.0003,
    action_noise=action_noise,
    buffer_size=50_000,
)

model.learn(total_timesteps=500_000)
model.save("../models/ddpg_model")
