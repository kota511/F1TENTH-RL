import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import gym
import yaml
from argparse import Namespace
from torch.utils.tensorboard import SummaryWriter
import time
import os
import argparse
import cma  # CMA-ES for hyperparameter optimisation

os.environ['PYGLET_PLATFORM'] = 'macos'

# =============================================================================
# Memory Buffer
# =============================================================================
class PPOMemory:
    def __init__(self, buffer_size, input_dims, n_actions, device):
        self.buffer_size = buffer_size
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []
        self.ptr = 0
        self.device = device
        self.buffer_size = buffer_size
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.device = device
        self.states = torch.zeros((buffer_size, input_dims), dtype=torch.float32, device=device)
        self.actions = torch.zeros((buffer_size, n_actions), dtype=torch.float32, device=device)
        self.probs = torch.zeros(buffer_size, dtype=torch.float32, device=device)
        self.vals = torch.zeros(buffer_size, dtype=torch.float32, device=device)
        self.rewards = torch.zeros(buffer_size, dtype=torch.float32, device=device)
        self.dones = torch.zeros(buffer_size, dtype=torch.float32, device=device)
        self.ptr = 0

    def store_memory(self, state, action, probs, vals, reward, done):
        if self.ptr < self.buffer_size:
            self.states[self.ptr] = torch.as_tensor(state, dtype=torch.float32, device=self.device)
            self.actions[self.ptr] = torch.as_tensor(action, dtype=torch.float32, device=self.device)
            self.probs[self.ptr] = torch.as_tensor(probs, dtype=torch.float32, device=self.device)
            self.vals[self.ptr] = torch.as_tensor(vals, dtype=torch.float32, device=self.device)
            self.rewards[self.ptr] = torch.as_tensor(reward, dtype=torch.float32, device=self.device)
            self.dones[self.ptr] = torch.as_tensor(done, dtype=torch.float32, device=self.device)
            self.ptr += 1
    
    def generate_batches(self, batch_size):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i: i + batch_size] for i in batch_start]
        return (np.array(self.states),
                np.array(self.actions),
                np.array(self.probs),
                np.array(self.vals),
                np.array(self.rewards),
                np.array(self.dones),
                batches)

    def clear_memory(self):
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []


# =============================================================================
# Actor and Critic Networks
# =============================================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ActorNetwork(nn.Module):
    def __init__(self, input_dims, n_actions, alpha=0.0003):
        super(ActorNetwork, self).__init__()
        self.device = device
        self.actor = nn.Sequential(
            nn.Linear(input_dims, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions),
            nn.Tanh()
        )
        self.log_std = nn.Parameter(torch.zeros(n_actions))
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.to(self.device)

    def forward(self, state):
        state = state.to(self.device)
        mean = self.actor(state)
        std = self.log_std.exp()
        return Normal(mean, std)

class CriticNetwork(nn.Module):
    def __init__(self, input_dims, alpha=0.0003):
        super(CriticNetwork, self).__init__()
        self.device = device
        self.critic = nn.Sequential(
            nn.Linear(input_dims, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.to(self.device)

    def forward(self, state):
        state = state.to(self.device)
        return self.critic(state)

# =============================================================================
# PPO Agent
# =============================================================================
class PPOAgent:
    def __init__(self,
                 input_dims,
                 n_actions,
                 gamma=0.99,
                 alpha=0.0003,
                 gae_lambda=0.95,
                 policy_clip=0.2,
                 batch_size=128,
                 n_epochs=10,
                 entropy_coeff=0.01,
                 buffer_size=2048):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        self.batch_size = batch_size
        self.entropy_coeff = entropy_coeff

        self.actor = ActorNetwork(input_dims, n_actions, alpha)
        self.critic = CriticNetwork(input_dims, alpha)
        self.buffer_size = buffer_size
        self.memory = PPOMemory(buffer_size, input_dims, n_actions, self.actor.device)

    def store_transition(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def choose_action(self, observation):
        state = torch.FloatTensor(np.array([observation])).to(self.actor.device)
        dist = self.actor(state)
        action = dist.sample()
        log_probs = dist.log_prob(action).sum()
        value = self.critic(state)
        return action.detach().numpy()[0], log_probs.detach().numpy(), value.item()

    def learn(self):
        state_arr = torch.stack(self.memory.states).to(self.actor.device)
        action_arr = torch.stack(self.memory.actions).to(self.actor.device)
        old_prob_arr = torch.stack(self.memory.probs).to(self.actor.device)
        vals_arr = torch.stack(self.memory.vals).to(self.actor.device)
        reward_arr = torch.stack(self.memory.rewards).to(self.actor.device)
        dones_arr = torch.stack(self.memory.dones).to(self.actor.device)

        for _ in range(self.n_epochs):
            (state_arr, action_arr, old_prob_arr, vals_arr,
             reward_arr, dones_arr, batches) = self.memory.generate_batches(self.batch_size)
            values = vals_arr
            returns = np.zeros_like(reward_arr, dtype=np.float32)
            advantages = np.zeros_like(reward_arr, dtype=np.float32)

            with torch.no_grad():
                values = torch.tensor(vals_arr, device=self.actor.device)
                rewards = torch.tensor(reward_arr, device=self.actor.device)
                dones = torch.tensor(dones_arr, device=self.actor.device)

                advantages = torch.zeros_like(rewards)
                last_gae = 0.0
                for t in reversed(range(len(rewards))):
                    next_value = values[t+1] if t < len(rewards)-1 else 0.0
                    next_non_terminal = 1.0 - dones[t]
                    delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
                    last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
                    advantages[t] = last_gae
                returns = advantages + values

            advantages = torch.tensor(advantages, dtype=torch.float32)
            if advantages.numel() > 1:
                adv_std = advantages.std()
                if adv_std < 1e-10:
                    adv_std = 1e-10
                advantages = (advantages - advantages.mean()) / adv_std

            for batch in batches:
                states = torch.FloatTensor(state_arr[batch])
                old_probs = torch.FloatTensor(old_prob_arr[batch])
                actions = torch.FloatTensor(action_arr[batch])
                returns_batch = torch.FloatTensor(returns[batch])
                adv_batch = advantages[batch]

                dist = self.actor(states)
                critic_value = self.critic(states).squeeze()
                new_log_probs = dist.log_prob(actions).sum(dim=1)

                prob_ratio = torch.exp(new_log_probs - old_probs)
                weighted_probs = adv_batch * prob_ratio
                clipped_probs = torch.clamp(prob_ratio, 1 - self.policy_clip, 1 + self.policy_clip) * adv_batch
                actor_loss = -torch.min(weighted_probs, clipped_probs).mean()

                critic_loss = 0.5 * (returns_batch - critic_value).pow(2).mean()
                entropy_loss = dist.entropy().mean()

                total_loss = actor_loss + critic_loss - self.entropy_coeff * entropy_loss

                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.actor.optimizer.step()
                self.critic.optimizer.step()
        

        self.memory.clear_memory()

    def save_models(self, actor_path, critic_path):
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)

    def load_models(self, actor_path, critic_path):
        self.actor.load_state_dict(torch.load(actor_path))
        self.critic.load_state_dict(torch.load(critic_path))


# =============================================================================
# Preprocessing and Reward Functions
# =============================================================================
def preprocess_state(obs):
    lidar = obs["scans"][0]
    pose = np.array([obs["poses_x"][0], obs["poses_y"][0], obs["poses_theta"][0]])
    velocity = obs["linear_vels_x"][0]
    state = np.concatenate([np.array([velocity / 20.0]), pose / 100.0, lidar / 30.0])
    return state


def compute_reward(obs, action, done):
    base_reward = 1.0
    survival_bonus = 0.05  # bonus per timestep for staying alive
    if done:
        return -5.0  # penalty for crashing
    lidar = obs["scans"][0]
    velocity = obs["linear_vels_x"][0]
    min_distance = min(lidar)
    distance_penalty = -2.0 if min_distance < 0.5 else 0.0
    speed_reward = velocity * 0.8
    return base_reward + distance_penalty + speed_reward + survival_bonus


# =============================================================================
# Evaluation Function: run n_eval episodes with the current policy (deterministically)
# =============================================================================
def evaluate_policy(agent, conf, n_eval=5):
    scores = []
    env = gym.make("f110_gym:f110-v0", map=conf.map_path, map_ext=conf.map_ext, num_agents=1)
    for _ in range(n_eval):
        obs, _, done, _ = env.reset(np.array([[conf.sx, conf.sy, conf.stheta]]))
        score = 0.0
        while not done:
            state = preprocess_state(obs)
            # Deterministic action: the mean of the actor's distribution
            state_tensor = torch.FloatTensor(np.array([state]))
            with torch.no_grad():
                dist = agent.actor(state_tensor)
                action = dist.mean.numpy()[0]
            steering = np.clip(action[0], -0.4, 0.4)
            speed = np.clip((action[1] + 1) * 2.0 + 1.0, 1.0, 3.0)
            obs, step_reward, done, _ = env.step(np.array([[steering, speed]]))
            score += step_reward
        scores.append(score)
    return np.mean(scores), np.std(scores)


# =============================================================================
# Training Function with Curriculum Learning and Periodic Evaluation
# =============================================================================
def train(conf, ppo_override=None, n_episodes=1000, writer_log=True):
    with open("config_Spielberg_map.yaml") as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)
    # Overwrite config values with passed configuration
    for key, val in conf.__dict__.items():
        if key in conf_dict:
            conf_dict[key] = val
    conf = Namespace(**conf_dict)
    # Convert nested ppo_params to a Namespace only if it's a dict.
    if isinstance(conf.ppo_params, dict):
        conf.ppo_params = Namespace(**conf.ppo_params)

    if writer_log:
        writer = SummaryWriter(log_dir="ppo_f1tenth_logs")
    else:
        writer = None

    env = gym.make("f110_gym:f110-v0", map=conf.map_path, map_ext=conf.map_ext, num_agents=1)

    input_dims = 1084
    n_actions = 2

    # Use overridden PPO hyperparameters if provided, else use defaults from config.
    alpha = ppo_override.get('alpha', conf.ppo_params.alpha) if ppo_override else conf.ppo_params.alpha
    gamma = ppo_override.get('gamma', conf.ppo_params.gamma) if ppo_override else conf.ppo_params.gamma
    gae_lambda = ppo_override.get('gae_lambda', conf.ppo_params.gae_lambda) if ppo_override else conf.ppo_params.gae_lambda
    policy_clip = ppo_override.get('policy_clip', conf.ppo_params.policy_clip) if ppo_override else conf.ppo_params.policy_clip
    entropy_coef = ppo_override.get('entropy_coef', conf.ppo_params.entropy_coef) if ppo_override else conf.ppo_params.entropy_coef

    agent = PPOAgent(
        input_dims=input_dims,
        n_actions=n_actions,
        batch_size=conf.ppo_params.batch_size,
        alpha=alpha,
        gamma=gamma,
        gae_lambda=gae_lambda,
        policy_clip=policy_clip,
        n_epochs=conf.ppo_params.n_epochs,
        entropy_coeff=entropy_coef
    )

    initial_max_steps = conf.ppo_params.initial_max_steps
    final_max_steps = conf.ppo_params.final_max_steps
    curriculum_episodes = conf.ppo_params.curriculum_episodes

    best_eval = -np.inf
    eval_interval = 50  # Evaluate every 50 episodes
    n_eval = 5          # Number of evaluation episodes

    for episode in range(n_episodes):
        # Linear curriculum for maximum steps.
        curr_max_steps = initial_max_steps + (final_max_steps - initial_max_steps) * min(episode, curriculum_episodes) / curriculum_episodes

        obs, _, done, _ = env.reset(np.array([[conf.sx, conf.sy, conf.stheta]]))
        score = 0.0
        steps = 0

        while not done and steps < curr_max_steps:
            state = preprocess_state(obs)
            action, prob, val = agent.choose_action(state)
            steering = np.clip(action[0], -0.4, 0.4)
            speed = np.clip((action[1] + 1) * 2.0 + 1.0, 1.0, 3.0)
            next_obs, _, done, _ = env.step(np.array([[steering, speed]]))
            reward = compute_reward(next_obs, action, done)
            agent.store_transition(state, action, prob, val, reward, done)

            if len(agent.memory.states) >= agent.batch_size:
                agent.learn()

            obs = next_obs
            score += reward
            steps += 1

        if writer:
            writer.add_scalar("Score", score, episode)
        print(f"Episode {episode}, Score: {score}, Steps: {steps} (Max allowed: {curr_max_steps})")

        if episode % eval_interval == 0 and episode > 0:
            eval_mean, eval_std = evaluate_policy(agent, conf, n_eval=n_eval)
            print(f"Evaluation at episode {episode}: Average Score = {eval_mean:.2f} ± {eval_std:.2f}")
            if eval_mean > best_eval:
                best_eval = eval_mean
                agent.save_models("ppo_best_actor.pth", "ppo_best_critic.pth")
                print(f"New best model saved at episode {episode} with average score {eval_mean:.2f}")

    if writer:
        writer.close()
    agent.save_models("ppo_actor.pth", "ppo_critic.pth")
    print("Final models saved to 'ppo_actor.pth' and 'ppo_critic.pth'.")
    return agent


# =============================================================================
# Simulator/ Evaluation Driver Using the Trained PPO Actor
# =============================================================================
class RLDriver:
    def __init__(self, actor_model_path, input_dims, n_actions):
        self.actor = ActorNetwork(input_dims, n_actions)
        self.actor.load_state_dict(torch.load(actor_model_path, map_location=torch.device('cpu')))
        self.actor.eval()

    def choose_action(self, observation):
        state = torch.FloatTensor(np.array([observation]))
        with torch.no_grad():
            dist = self.actor(state)
            action = dist.mean  # deterministic action selection
        action_np = action.numpy()[0]
        steering = np.clip(action_np[0], -0.4, 0.4)
        speed = np.clip((action_np[1] + 1) * 2.0 + 1.0, 1.0, 3.0)
        return [steering, speed]


class GymRunner:
    def __init__(self, racetrack, drivers):
        self.racetrack = racetrack
        self.drivers = drivers

    def run(self, render=True):
        with open('config_Spielberg_map.yaml') as file:
            conf_dict = yaml.load(file, Loader=yaml.FullLoader)
        conf = Namespace(**conf_dict)

        env = gym.make('f110_gym:f110-v0', map=conf.map_path, map_ext=conf.map_ext, num_agents=1)
        poses = np.array([[0.8007017, -0.2753365, 4.1421595]])
        obs, step_reward, done, info = env.reset(poses=poses)
        if render:
            env.render()
        laptime = 0.0
        start = time.time()

        while not done:
            actions = []
            state = preprocess_state(obs)
            for driver in self.drivers:
                action = driver.choose_action(state)
                actions.append(action)
            actions = np.array(actions)
            obs, step_reward, done, info = env.step(actions)
            laptime += step_reward
            if render:
                env.render(mode='human')

        elapsed = time.time() - start
        print('Sim elapsed time:', laptime, 'Real elapsed time:', elapsed)
        return laptime


def evaluate(n_eval=10):
    input_dims = 1084
    n_actions = 2
    driver = RLDriver("ppo_actor.pth", input_dims, n_actions)
    drivers = [driver]
    RACETRACK = 'Spielberg'
    runner = GymRunner(RACETRACK, drivers)
    scores = []
    for i in range(n_eval):
        print(f"Evaluating episode {i} ...")
        score = runner.run(render=False)
        scores.append(score)
    avg_score = np.mean(scores)
    std_score = np.std(scores)
    print(f"Average Score over {n_eval} episodes: {avg_score:.2f} ± {std_score:.2f}")
    return avg_score


# =============================================================================
# Hyperparameter Optimisation via CMA-ES
# =============================================================================
def run_training_eval(conf, hyperparams, n_train=50, n_eval=5):
    """
    Train for a small number of episodes and then evaluate.
    hyperparams is a dict containing: alpha, gamma, gae_lambda, policy_clip, entropy_coef.
    """
    print("Training with hyperparameters:", hyperparams)
    # Train without Tensorboard logging to reduce overhead
    train(conf, ppo_override=hyperparams, n_episodes=n_train, writer_log=False)
    # Create a dummy agent to evaluate using the saved model
    input_dims = 1084
    n_actions = 2
    agent = PPOAgent(input_dims, n_actions)  # new agent instance
    agent.load_models("ppo_actor.pth", "ppo_critic.pth")
    eval_mean, _ = evaluate_policy(agent, conf, n_eval=n_eval)
    print("Evaluation score:", eval_mean)
    return eval_mean


def optimise_hyperparams(conf):
    # Define the objective function for CMA-ES.
    # We are tuning [alpha, gamma, gae_lambda, policy_clip, entropy_coef].
    def objective(x):
        # Clip to sensible ranges.
        alpha = np.clip(x[0], 1e-5, 1e-2)
        gamma = np.clip(x[1], 0.90, 0.999)
        gae_lambda = np.clip(x[2], 0.8, 0.99)
        policy_clip = np.clip(x[3], 0.1, 0.4)
        entropy_coef = np.clip(x[4], 0.0, 0.1)
        hyperparams = {
            'alpha': alpha,
            'gamma': gamma,
            'gae_lambda': gae_lambda,
            'policy_clip': policy_clip,
            'entropy_coef': entropy_coef
        }
        score = run_training_eval(conf, hyperparams, n_train=50, n_eval=5)
        # CMA minimises the objective, so return the negative score.
        return -score

    # Initial guess: use the default parameters from your config.
    with open("config_Spielberg_map.yaml") as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)
    conf = Namespace(**conf_dict)
    conf.ppo_params = Namespace(**conf.ppo_params)
    x0 = [conf.ppo_params.alpha,
          conf.ppo_params.gamma,
          conf.ppo_params.gae_lambda,
          conf.ppo_params.policy_clip,
          conf.ppo_params.entropy_coef]
    sigma = 0.1  # initial standard deviation
    opts = {
        'popsize': conf_dict.get('popsize', 100),
        'maxfevals': conf_dict.get('budget', 1000)
    }
    print("Starting CMA optimisation...")
    res = cma.fmin(objective, x0, sigma, options=opts)
    best_params = res[0]
    best_hyperparams = {
        'alpha': np.clip(best_params[0], 1e-5, 1e-2),
        'gamma': np.clip(best_params[1], 0.90, 0.999),
        'gae_lambda': np.clip(best_params[2], 0.8, 0.99),
        'policy_clip': np.clip(best_params[3], 0.1, 0.4),
        'entropy_coef': np.clip(best_params[4], 0.0, 0.1)
    }
    print("Optimisation completed. Best hyperparameters found:")
    print(best_hyperparams)
    return best_hyperparams


# =============================================================================
# Main Function with Mode Selection
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'simulate', 'evaluate', 'optimise'],
                        help='Select mode: "train" to train and save models, "simulate" to run one simulation, "evaluate" to run multiple episodes and average performance, or "optimise" to tune hyperparameters via CMA.')
    args = parser.parse_args()

    # Load config from YAML.
    with open("config_Spielberg_map.yaml") as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)
    conf = Namespace(**conf_dict)
    conf.ppo_params = Namespace(**conf.ppo_params)

    if args.mode == 'train':
        train(conf)
    elif args.mode == 'simulate':
        input_dims = 1084
        n_actions = 2
        driver = RLDriver("ppo_actor.pth", input_dims, n_actions)
        drivers = [driver]
        RACETRACK = 'Spielberg'
        runner = GymRunner(RACETRACK, drivers)
        runner.run(render=True)
    elif args.mode == 'evaluate':
        evaluate(n_eval=10)
    elif args.mode == 'optimise':
        best_hyperparams = optimise_hyperparams(conf)
        # Optionally, retrain a final model with the best hyperparameters found:
        print("Retraining final model with optimised hyperparameters...")
        train(conf, ppo_override=best_hyperparams)
        evaluate(n_eval=10)

if __name__ == '__main__':
    main()
