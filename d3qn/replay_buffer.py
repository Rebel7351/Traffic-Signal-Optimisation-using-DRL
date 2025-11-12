from collections import deque
import torch
import random
from device import device

class ReplayBuffer:
    """Experience replay buffer for Dueling DQN"""
    def __init__(self, capacity, state_size):
        self.capacity = capacity
        self.state_size = state_size
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        # Ensure states are numpy arrays
        if isinstance(state, torch.Tensor):
            state = state.cpu().numpy()
        if isinstance(next_state, torch.Tensor):
            next_state = next_state.cpu().numpy()
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states = torch.FloatTensor([e[0] for e in batch]).to(device)
        actions = torch.LongTensor([e[1] for e in batch]).to(device)
        rewards = torch.FloatTensor([e[2] for e in batch]).to(device)
        next_states = torch.FloatTensor([e[3] for e in batch]).to(device)
        dones = torch.BoolTensor([e[4] for e in batch]).to(device)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)
