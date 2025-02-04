import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import gym
import yaml
from argparse import Namespace
from torch.utils.tensorboard import SummaryWriter

class PPOMemory:
    """
    Memory buffer to store states, actions, probabilities, values, rewards, and dones.
    Used for generating batches for training the PPO agent.
    """
    def __init__(self):
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []

    def generate_batches(self, batch_size):
        """
        Generate shuffled batches from memory for training.
        
        Args:
            batch_size (int): Size of each batch.
        
        Returns:
            Tuple of arrays: states, actions, probabilities, values, rewards, dones, and batch indices.
        """
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i : i + batch_size] for i in batch_start]

        return (
            np.array(self.states),
            np.array(self.actions),
            np.array(self.probs),
            np.array(self.vals),
            np.array(self.rewards),
            np.array(self.dones),
            batches
        )

    def store_memory(self, state, action, probs, vals, reward, done):
        """
        Store a single transition in memory.
        
        Args:
            state (np.array): Current state.
            action (np.array): Action taken.
            probs (float): Probability of the action.
            vals (float): Value estimated by the critic.
            reward (float): Reward received.
            done (bool): Whether the episode is done.
        """
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        """Clear all stored transitions from memory."""
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []

class ActorNetwork(nn.Module):
    """
    Neural network for the actor (policy network).
    Outputs a probability distribution over actions.
    """
    def __init__(self, input_dims, n_actions, alpha=0.0003):
        super(ActorNetwork, self).__init__()

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

    def forward(self, state):
        """
        Forward pass for the actor network.
        
        Args:
            state (torch.Tensor): Input state.
        
        Returns:
            torch.distributions.Normal: Probability distribution over actions.
        """
        mean = self.actor(state)
        std = self.log_std.exp()  # ensures standard deviation is always positive
        dist = Normal(mean, std)
        return dist

class CriticNetwork(nn.Module):
    """
    Neural network for the critic (value network).
    Outputs a single value representing the state value.
    """
    def __init__(self, input_dims, alpha=0.0003):
        super(CriticNetwork, self).__init__()

        self.critic = nn.Sequential(
            nn.Linear(input_dims, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)

    def forward(self, state):
        """
        Forward pass for the critic network.
        
        Args:
            state (torch.Tensor): Input state.
        
        Returns:
            torch.Tensor: State value.
        """
        return self.critic(state)

class PPOAgent:
    """
    Proximal Policy Optimization (PPO) agent.
    Contains the actor and critic networks and performs training steps.
    """
    def __init__(
        self,
        input_dims,
        n_actions,
        gamma=0.99,
        alpha=0.0003,
        gae_lambda=0.95,
        policy_clip=0.2,
        batch_size=128,
        n_epochs=10,
        entropy_coeff=0.01
    ):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        self.batch_size = batch_size
        self.entropy_coeff = entropy_coeff

        self.actor = ActorNetwork(input_dims, n_actions, alpha)
        self.critic = CriticNetwork(input_dims, alpha)
        self.memory = PPOMemory()

    def store_transition(self, state, action, probs, vals, reward, done):
        """
        Store a transition in the memory.
        
        Args:
            state (np.array): Current state.
            action (np.array): Action taken.
            probs (float): Probability of the action.
            vals (float): Value estimated by the critic.
            reward (float): Reward received.
            done (bool): Whether the episode is done.
        """
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def choose_action(self, observation):
        """
        Choose an action based on the current policy.
        
        Args:
            observation (np.array): Current state observation.
        
        Returns:
            tuple: Action, log probability of the action, and state value.
        """
        state = torch.FloatTensor(np.array([observation]))
        dist = self.actor(state)
        action = dist.sample()
        log_probs = dist.log_prob(action).sum()
        value = self.critic(state)

        return action.detach().numpy()[0], log_probs.detach().numpy(), value.detach().numpy()[0]

    def learn(self):
        """
        Perform the training step if the memory has enough data to form a batch.
        We also add an extra check to handle the case when advantage has zero variance.
        """
        for _ in range(self.n_epochs):
            (
                state_arr,
                action_arr,
                old_prob_arr,
                vals_arr,
                reward_arr,
                dones_arr,
                batches
            ) = self.memory.generate_batches(self.batch_size)

            values = vals_arr
            returns = np.zeros_like(reward_arr, dtype=np.float32)
            advantages = np.zeros_like(reward_arr, dtype=np.float32)

            last_gae = 0.0
            for t in reversed(range(len(reward_arr))):
                next_value = values[t + 1] if t < len(reward_arr) - 1 else 0.0
                next_non_terminal = 1.0 - dones_arr[t]
                delta = (
                    reward_arr[t]
                    + self.gamma * next_value * next_non_terminal
                    - values[t]
                )
                last_gae = (
                    delta
                    + self.gamma * self.gae_lambda * next_non_terminal * last_gae
                )
                advantages[t] = last_gae
                returns[t] = advantages[t] + values[t]

            advantages = torch.tensor(advantages, dtype=torch.float32)
            # safeguard: only normalise if there's more than one advantage in the batch
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
                clipped_probs = torch.clamp(
                    prob_ratio, 1 - self.policy_clip, 1 + self.policy_clip
                ) * adv_batch
                actor_loss = -torch.min(weighted_probs, clipped_probs).mean()

                critic_loss = 0.5 * (returns_batch - critic_value).pow(2).mean()
                entropy_loss = dist.entropy().mean()

                total_loss = actor_loss + critic_loss - self.entropy_coeff * entropy_loss

                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()

                # clip gradients just in case
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)

                self.actor.optimizer.step()
                self.critic.optimizer.step()

        self.memory.clear_memory()

def preprocess_state(obs):
    """
    Preprocess the raw observation from the environment into a normalised state.
    
    Args:
        obs (dict): Raw observation containing LiDAR scans, pose, and velocity.
    
    Returns:
        np.array: Normalised state vector.
    """
    lidar = obs["scans"][0]
    pose = np.array([obs["poses_x"][0], obs["poses_y"][0], obs["poses_theta"][0]])
    velocity = obs["linear_vels_x"][0]
    state = np.concatenate(
        [np.array([velocity / 20.0]), pose / 100.0, lidar / 30.0]
    )
    return state

def compute_reward(obs, action, done):
    """
    Compute the reward for the current step.
    
    Args:
        obs (dict): Current observation.
        action (np.array): Action taken by the agent.
        done (bool): Whether the episode is done.
    
    Returns:
        float: Reward for the current step.
    """
    base_reward = 1.0
    if done:
        return -5.0  # penalty for crashing

    lidar = obs["scans"][0]
    velocity = obs["linear_vels_x"][0]
    min_distance = min(lidar)

    # penalty for being too close to obstacles/sides
    distance_penalty = -2.0 if min_distance < 0.5 else 0.0

    # reward for higher velocity
    speed_reward = velocity * 0.8

    return base_reward + distance_penalty + speed_reward

def main():
    """
    Main training loop for the PPO agent in the F1TENTH environment.
    """
    with open("config_Spielberg_map.yaml") as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)
    conf = Namespace(**conf_dict)

    writer = SummaryWriter(log_dir="ppo_f1tenth_logs")

    env = gym.make("f110_gym:f110-v0", map=conf.map_path, map_ext=conf.map_ext, num_agents=1)
    input_dims = 1084
    n_actions = 2
    agent = PPOAgent(
        input_dims=input_dims,
        n_actions=n_actions,
        batch_size=128,
        alpha=0.0003,
        gamma=0.995,
        gae_lambda=0.95,
        policy_clip=0.1,
        n_epochs=4,
        entropy_coeff=0.001
    )

    n_episodes = 1000

    for episode in range(n_episodes):
        obs, _, done, _ = env.reset(
            np.array([[conf.sx, conf.sy, conf.stheta]])
        )
        score = 0.0
        steps = 0

        while not done and steps < 1000:
            state = preprocess_state(obs)
            action, prob, val = agent.choose_action(state)
            steering = np.clip(action[0], -0.4, 0.4)
            speed = np.clip((action[1] + 1) * 2.0 + 1.0, 1.0, 3.0)
            next_obs, _, done, _ = env.step(np.array([[steering, speed]]))
            reward = compute_reward(next_obs, action, done)

            agent.store_transition(state, action, prob, val, reward, done)

            # only learn if we've at least got one full batch in memory
            if len(agent.memory.states) >= agent.batch_size:
                agent.learn()

            obs = next_obs
            score += reward
            steps += 1

        writer.add_scalar("Score", score, episode)
        print(f"Episode {episode}, Score: {score}, Steps: {steps}")

    writer.close()

if __name__ == "__main__":
    main()
