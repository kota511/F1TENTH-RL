import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gym
import yaml
from argparse import Namespace
from torch.utils.tensorboard import SummaryWriter

class ReplayBuffer:
    """
    Replay buffer to store transitions for training the DDPG agent.
    """
    def __init__(self, max_size, input_dims, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.action_memory = np.zeros((self.mem_size, n_actions), dtype=np.float32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, states_, dones

class ActorNetwork(nn.Module):
    """
    Actor network for DDPG.
    Maps states to actions.
    """
    def __init__(self, input_dims, n_actions, alpha, fc1_dims=400, fc2_dims=300):
        super(ActorNetwork, self).__init__()

        self.fc1 = nn.Linear(*input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.fc3 = nn.Linear(fc2_dims, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))

        return x

class CriticNetwork(nn.Module):
    """
    Critic network for DDPG.
    Maps states and actions to Q-values.
    """
    def __init__(self, input_dims, n_actions, alpha, fc1_dims=400, fc2_dims=300):
        super(CriticNetwork, self).__init__()

        self.fc1 = nn.Linear(*input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims + n_actions, fc2_dims)
        self.fc3 = nn.Linear(fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, action):
        x = torch.relu(self.fc1(state))
        x = torch.cat([x, action], dim=1)
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)

        return x

class DDPGAgent:
    """
    Deep Deterministic Policy Gradient (DDPG) agent.
    """
    def __init__(self, alpha, beta, input_dims, tau, env, gamma=0.99, 
                 n_actions=2, max_size=1_000_000, fc1_dims=400, fc2_dims=300, 
                 batch_size=64):
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.env = env

        self.actor = ActorNetwork(input_dims, n_actions, alpha, fc1_dims, fc2_dims)
        self.target_actor = ActorNetwork(input_dims, n_actions, alpha, fc1_dims, fc2_dims)
        self.critic = CriticNetwork(input_dims, n_actions, beta, fc1_dims, fc2_dims)
        self.target_critic = CriticNetwork(input_dims, n_actions, beta, fc1_dims, fc2_dims)

        self.memory = ReplayBuffer(max_size, input_dims, n_actions)

        self.update_network_parameters(tau=1)

    def choose_action(self, observation):
        """
        Choose action based on the current policy.
        """
        state = torch.tensor([observation], dtype=torch.float32).to(self.actor.device)
        action = self.actor(state).cpu().detach().numpy()[0]
        return action

    def store_transition(self, state, action, reward, state_, done):
        """
        Store a transition in the replay buffer.
        """
        self.memory.store_transition(state, action, reward, state_, done)

    def update_network_parameters(self, tau=None):
        """
        Perform soft update of target network parameters.
        """
        if tau is None:
            tau = self.tau

        actor_params = dict(self.actor.named_parameters())
        target_actor_params = dict(self.target_actor.named_parameters())
        for name in actor_params:
            target_actor_params[name].data.copy_(tau * actor_params[name].data + (1 - tau) * target_actor_params[name].data)

        critic_params = dict(self.critic.named_parameters())
        target_critic_params = dict(self.target_critic.named_parameters())
        for name in critic_params:
            target_critic_params[name].data.copy_(tau * critic_params[name].data + (1 - tau) * target_critic_params[name].data)

    def learn(self):
        """
        Sample a batch of transitions and update the networks.
        """
        if self.memory.mem_cntr < self.batch_size:
            return

        states, actions, rewards, states_, dones = self.memory.sample_buffer(self.batch_size)

        states = torch.tensor(states, dtype=torch.float32).to(self.critic.device)
        actions = torch.tensor(actions, dtype=torch.float32).to(self.critic.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.critic.device)
        states_ = torch.tensor(states_, dtype=torch.float32).to(self.critic.device)
        dones = torch.tensor(dones, dtype=torch.bool).to(self.critic.device)

        target_actions = self.target_actor(states_)
        target_critic_value = self.target_critic(states_, target_actions)
        target_critic_value[dones] = 0.0
        target_critic_value = target_critic_value.view(-1)

        critic_value = self.critic(states, actions).view(-1)
        target = rewards + self.gamma * target_critic_value

        critic_loss = nn.MSELoss()(critic_value, target)
        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        self.critic.optimizer.step()

        actor_loss = -self.critic(states, self.actor(states)).mean()
        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_network_parameters()

def preprocess_state(obs):
    """
    Preprocess observation for the DDPG agent.
    """
    lidar = obs['scans'][0]
    pose = np.array([obs['poses_x'][0], obs['poses_y'][0], obs['poses_theta'][0]])
    velocity = obs['linear_vels_x'][0]

    state = np.concatenate([
        np.array([velocity / 20.0]), pose / 100.0, lidar / 30.0
    ])

    return state

def compute_reward(obs, action, done):
    """
    Compute reward based on the current state.
    """
    base_reward = 1.0
    if done:
        return -10.0

    lidar = obs['scans'][0]
    velocity = obs['linear_vels_x'][0]
    min_distance = min(lidar)

    distance_penalty = -5.0 if min_distance < 0.5 else 0.0
    speed_reward = velocity * 0.5

    return base_reward + distance_penalty + speed_reward

def main():
    with open('config_Spielberg_map.yaml') as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)
    conf = Namespace(**conf_dict)

    writer = SummaryWriter(log_dir='ddpg_f1tenth_logs')

    env = gym.make('f110_gym:f110-v0', map=conf.map_path, map_ext=conf.map_ext, num_agents=1)
    input_dims = [1084]
    n_actions = 2

    agent = DDPGAgent(
        alpha=conf.ddpg_params['actor_lr'],
        beta=conf.ddpg_params['critic_lr'],
        input_dims=input_dims,
        tau=conf.ddpg_params['tau'],
        env=env,
        gamma=conf.ddpg_params['gamma'],
        n_actions=n_actions,
        max_size=conf.ddpg_params['replay_buffer_size'],
        batch_size=conf.ddpg_params['batch_size']
    )


    n_episodes = 1000

    for episode in range(n_episodes):
        obs, _, _, _ = env.reset(np.array([[conf.sx, conf.sy, conf.stheta]]))
        score = 0
        done = False
        steps = 0

        while not done and steps < 1000:
            state = preprocess_state(obs)
            action = agent.choose_action(state)
            steering = np.clip(action[0], -0.4, 0.4)
            speed = np.clip((action[1] + 1) * 2.0 + 1.0, 1.0, 3.0)

            next_obs, reward, done, _ = env.step(np.array([[steering, speed]]))
            reward = compute_reward(next_obs, action, done)

            next_state = preprocess_state(next_obs)
            agent.store_transition(state, action, reward, next_state, done)
            agent.learn()

            obs = next_obs
            score += reward
            steps += 1

        writer.add_scalar('Score', score, episode)
        print(f'Episode {episode}, Score: {score}, Steps: {steps}')

    writer.close()

if __name__ == '__main__':
    main()
