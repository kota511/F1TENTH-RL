import gym
import yaml
import numpy as np
from argparse import Namespace
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from purepursuit import PurePursuitPlanner
import time

TRACK = "example"
WHEELBASE = 0.17145 + 0.15875

class F110RLWrapper(gym.Wrapper):
    def __init__(self, env, planner, conf, max_steps=5000):
        super().__init__(env)
        self.env = env
        self.planner = planner
        self.conf = conf
        self.max_steps = max_steps
        self.step_count = 0
        self.obs = None
        self.angle_error = 0.0

        self.observation_space = gym.spaces.Box(
            low=np.array([0.0, 0.0, -np.pi, 0.0, -np.pi]),
            high=np.array([np.inf, np.inf, np.pi, np.inf, np.pi]),
            dtype=np.float32
        )

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
        self.obs, reward, done, info = self.env.step(sim_action)
        self.step_count += 1

        # End episode after 2 laps or max steps
        lap_count = self.obs['lap_counts'][0]
        done = done or lap_count >= 2 or self.step_count >= self.max_steps

        return self._preprocess_obs(self.obs), reward, done, info

    def _preprocess_obs(self, obs):
        pose_x = obs['poses_x'][0]
        pose_y = obs['poses_y'][0]
        pose_theta = obs['poses_theta'][0]
        linear_vel = obs['linear_vels_x'][0]

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
            self.angle_error = 0.0

        return np.array([pose_x, pose_y, pose_theta, linear_vel, self.angle_error], dtype=np.float32)

    def _preprocess_action(self, action):
        MAX_STEER = 1.0
        steer = float(action[0]) * MAX_STEER
        pose_x = self.obs['poses_x'][0]
        pose_y = self.obs['poses_y'][0]
        pose_theta = self.obs['poses_theta'][0]
        speed, _ = self.planner.plan(pose_x, pose_y, pose_theta, self.conf.tlad, self.conf.vgain)
        return np.array([[steer, speed]], dtype=np.float32)

def make_env():
    with open(f"../f1tenth_racetracks/{TRACK}/config_{TRACK}_map.yaml") as file:
        conf_dict = yaml.safe_load(file)
    conf = Namespace(**conf_dict)
    conf.map_path = f"../f1tenth_racetracks/{TRACK}/{TRACK}_map"
    conf.wpt_path = f"../f1tenth_racetracks/{TRACK}/{TRACK}_waypoints.csv"

    base_env = gym.make(
        "f110_gym:f110-v0",
        map=conf.map_path,
        map_ext=conf.map_ext,
        num_agents=1,
        timestep=0.01
    )

    planner = PurePursuitPlanner(conf, WHEELBASE)
    wrapped_env = F110RLWrapper(base_env, planner, conf)
    return wrapped_env

if __name__ == "__main__":
    env = DummyVecEnv([make_env])
    model = PPO.load(f"./PPO_train_{TRACK}/best_model")

    obs = env.reset()
    done = False
    total_reward = 0.0
    start = time.time()

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        raw_obs = env.envs[0].obs  # access the raw obs dictionary inside your wrapper
        lap_count = raw_obs['lap_counts'][0]
        lap_time = raw_obs['lap_times'][0]
        env.envs[0].env.render()

    real_elapsed_time = time.time() - start

    print(f"\nEvaluation Summary:")
    print(f"Total reward: {float(total_reward):.2f}")
    print(f"Lap count: {lap_count}")
    print(f"Lap time (last completed lap): {lap_time:.2f} sec")
    print(f"Wall-clock time: {real_elapsed_time:.2f} sec")
