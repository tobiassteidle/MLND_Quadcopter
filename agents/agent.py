import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from models.model import Actor, Critic
from buffer.replay_buffer import ReplayBuffer

# Hyperparameters
gamma = 0.99                # discount for future rewards
tau = 1e-3                  # update parameter (1-tau)
policy_noise = 0.1          # target policy smoothing noise
noise_clip = 0.5
policy_delay = 2            # delayed policy updates parameter
lr_actor = 1e-4
lr_critic = 1e-4

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class TD3:
    def __init__(self, task):
        self.task = task
        self.state_dim = task.state_size
        self.action_dim = task.action_size
        self.action_low = task.action_low
        self.action_high = task.action_high

        self.actor = Actor(self.state_dim, self.action_dim, self.action_high).to(device)
        self.actor_target = Actor(self.state_dim, self.action_dim, self.action_high).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)

        self.critic_1 = Critic(self.state_dim, self.action_dim).to(device)
        self.critic_1_target = Critic(self.state_dim, self.action_dim).to(device)
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters(), lr=lr_critic)

        self.critic_2 = Critic(self.state_dim, self.action_dim).to(device)
        self.critic_2_target = Critic(self.state_dim, self.action_dim).to(device)
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters(), lr=lr_critic)

        # Replay memory
        self.buffer_size = 100000
        self.batch_size = 100
        self.memory = ReplayBuffer(self.buffer_size)

        self.step_count = 0

    def reset_episode(self):
        #self.noise.reset()
        state = self.task.reset()
        self.last_state = state
        return state

    def step(self, action, reward, next_state, done):
        # Save experience / reward
        self.memory.add((self.last_state, next_state, action, reward, float(done)))

        # Learn, if enough samples are available in memory
        if len(self.memory) > self.batch_size:
            self.learn()

        # Roll over last state and action
        self.last_state = next_state

    def act(self, state):
        state = torch.cuda.FloatTensor(state.reshape(1, -1)).to(device)
        action = self.actor(state).cpu().data.numpy().flatten()
        return action

    def learn(self):
        self.step_count+=1

        # Sample a batch of transitions from replay buffer:
        x, y, u, r, d = self.memory.sample(self.batch_size)
        state = torch.cuda.FloatTensor(x).to(device)
        action = torch.cuda.FloatTensor(u).to(device)
        next_state = torch.cuda.FloatTensor(y).to(device)
        done = torch.cuda.FloatTensor(1 - d).to(device)
        reward = torch.cuda.FloatTensor(r).to(device)

        # Select action according to policy and add clipped noise
        noise = torch.cuda.FloatTensor(u).data.normal_(0, policy_noise).to(device)
        noise = noise.clamp(-noise_clip, noise_clip)
        next_action = (self.actor_target(next_state) + noise).clamp(self.action_low, self.action_high)

        # Compute target Q-value:
        target_Q1 = self.critic_1_target(next_state, next_action)
        target_Q2 = self.critic_2_target(next_state, next_action)
        target_Q = torch.min(target_Q1, target_Q2)
        target_Q = reward + ((1-done) * gamma * target_Q).detach()

        # Optimize Critic 1:
        current_Q1 = self.critic_1(state, action)
        loss_Q1 = F.mse_loss(current_Q1, target_Q)
        self.critic_1_optimizer.zero_grad()
        loss_Q1.backward()
        self.critic_1_optimizer.step()

        # Optimize Critic 2:
        current_Q2 = self.critic_2(state, action)
        loss_Q2 = F.mse_loss(current_Q2, target_Q)
        self.critic_2_optimizer.zero_grad()
        loss_Q2.backward()
        self.critic_2_optimizer.step()

        # Delayed policy updates:
        if self.step_count % policy_delay == 0:
            # Compute actor loss:
            actor_loss = -self.critic_1(state, self.actor(state)).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # update:
            self.soft_update(self.actor, self.actor_target, tau)
            self.soft_update(self.critic_1, self.critic_1_target, tau)
            self.soft_update(self.critic_2, self.critic_2_target, tau)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        ?_target = t*?_local + (1 - t)*?_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
