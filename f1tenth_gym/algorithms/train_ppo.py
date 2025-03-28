import os
import time
import yaml
import numpy as np
from argparse import Namespace
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from f110_gym.envs.base_classes import Integrator
from purepursuit import PurePursuitPlanner
from callbacks import SurvivalEvalCallback
import gym
import torch

# Constants for saving models and logs
MODEL_DIR = "models"
LOG_DIR = "logs"
TRACK = "example"
BEST_MODEL_DIR = f"./PPO_train_{TRACK}/"  # Best model checkpoint path
WHEELBASE = 0.17145 + 0.15875

# Make directories if they don't exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(BEST_MODEL_DIR, exist_ok=True)

def make_env():
    """
    Creates and returns the wrapped F1TENTH environment with PurePursuit planner.
    """
    # Load YAML config
    with open(f"../f1tenth_racetracks/{TRACK}/config_{TRACK}_map.yaml") as file:
        conf_dict = yaml.safe_load(file)
    conf = Namespace(**conf_dict)
    conf.map_path = f"../f1tenth_racetracks/{TRACK}/{TRACK}_map"
    conf.wpt_path = f"../f1tenth_racetracks/{TRACK}/{TRACK}_waypoints.csv"

    # Create base F1TENTH gym environment
    base_env = gym.make(
        "f110_gym:f110-v0",
        map=conf.map_path,
        map_ext=conf.map_ext,
        num_agents=1,
        timestep=0.01,
        integrator=Integrator.RK4
    )

    # Instantiate the PurePursuit planner
    planner = PurePursuitPlanner(conf, WHEELBASE)

    # Wrap the environment to interface with PPO
    wrapped_env = F110RLWrapper(base_env, planner, conf)
    return wrapped_env

class F110RLWrapper(gym.Wrapper):
    def __init__(self, env, planner, conf, max_steps=5000):
        super().__init__(env)
        self.env = env
        self.planner = planner
        self.conf = conf
        self.max_steps = max_steps
        self.step_count = 0
        self.obs = None
        self.angle_error = 0.0  # cache angle error for reward
        self.prev_lap_count = 0

        # Updated observation space: [x, y, theta, speed, angle_error]
        self.observation_space = gym.spaces.Box(
            low=np.array([0.0, 0.0, -np.pi, 0.0, -np.pi]),
            high=np.array([np.inf, np.inf, np.pi, np.inf, np.pi]),
            shape=(5,),
            dtype=np.float32
        )

        # PPO only controls steering (normalized)
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0]),
            high=np.array([1.0]),
            shape=(1,),
            dtype=np.float32
        )

    def reset(self):
        self.step_count = 0
        initial_pose = np.array([[self.conf.sx, self.conf.sy, self.conf.stheta]])
        self.obs, *_ = self.env.reset(poses=initial_pose)
        return self._preprocess_obs(self.obs)

    def step(self, action):
        sim_action = self._preprocess_action(action)
        self.obs, _, done, info = self.env.step(sim_action)
        self.step_count += 1

        reward = self._compute_reward()

        self.prev_lap_count = self.obs['lap_counts'][0]

        done = done or self.step_count >= self.max_steps
        return self._preprocess_obs(self.obs), reward, done, info

    def _preprocess_obs(self, obs):
        pose_x = obs['poses_x'][0]
        pose_y = obs['poses_y'][0]
        pose_theta = obs['poses_theta'][0]
        linear_vel = obs['linear_vels_x'][0]

        # Compute angle error to lookahead point
        position = np.array([pose_x, pose_y])
        lookahead_point = self.planner._get_current_waypoint(
            self.planner.waypoints, self.conf.tlad, position, pose_theta
        )
        if lookahead_point is not None:
            dx = lookahead_point[0] - pose_x
            dy = lookahead_point[1] - pose_y
            desired_heading = np.arctan2(dy, dx)
            self.angle_error = np.arctan2(
                np.sin(desired_heading - pose_theta),
                np.cos(desired_heading - pose_theta)
            )
        else:
            self.angle_error = 0.0  # fallback

        return np.array([pose_x, pose_y, pose_theta, linear_vel, self.angle_error], dtype=np.float32)

    def _preprocess_action(self, action):
        MAX_STEER = 1.0
        steer = float(action[0]) * MAX_STEER

        pose_x = self.obs['poses_x'][0]
        pose_y = self.obs['poses_y'][0]
        pose_theta = self.obs['poses_theta'][0]

        MAX_SPEED = 10.0
        speed, _ = self.planner.plan(pose_x, pose_y, pose_theta, self.conf.tlad, self.conf.vgain)
        speed = np.clip(speed, 0.0, MAX_SPEED)

        return np.array([[steer, speed]], dtype=np.float32)

    def _compute_reward(self):
        speed = 0.1 * self.obs['linear_vels_x'][0]
        crash_penalty = -50.0 if self.step_count < self.max_steps and self.obs['collisions'][0] > 0 else 0.0
        angle_penalty = -1.0 * abs(self.angle_error) #best result with -2 so far
        alive_bonus = 2 #best result with 1 so far

        lap_count = self.obs['lap_counts'][0]
        lap_bonus = 0.0
        if lap_count > 0.5:
            print(f"Lap completed! Lap count: {lap_count}")
            lap_bonus = 50.0

        return speed + alive_bonus + angle_penalty + crash_penalty + lap_bonus

if __name__ == "__main__":
    # Create vectorized environment for PPO
    train_env = DummyVecEnv([make_env])
    eval_env = DummyVecEnv([make_env])

    # Define PPO model
    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        learning_rate=0.0003,
        tensorboard_log=LOG_DIR,
        # device="mps"
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # Callback to log and save best model during training
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=BEST_MODEL_DIR,
        log_path=BEST_MODEL_DIR,
        eval_freq=5000,
        deterministic=True,
        render=False
    )

    # Start training
    model.learn(
        total_timesteps=5_000_000,
        callback=eval_callback,
        progress_bar=True
    )

    # Save final model
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    model.save(f"{MODEL_DIR}/ppo_model_{timestamp}")
