import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from network import QNetwork
from replay_buffer import ReplayBuffer
from device import device

class DQNAgent:
    """Deep Q-Network Agent for traffic signal control"""

    def __init__(self, state_size, action_size, agent_id, learning_rate=0.001, 
                 gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995,
                 buffer_size=3600, batch_size=32, target_update_freq=100,
                 hidden_layers=[128, 128]):

        self.state_size = state_size
        self.action_size = action_size
        self.agent_id = agent_id
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        # Neural networks
        self.q_network = QNetwork(state_size, action_size, hidden_layers).to(device)
        self.target_network = QNetwork(state_size, action_size, hidden_layers).to(device)

        # Initialize target network with same weights as main network
        self.update_target_network()

        # Optimizer and loss function
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()

        # Experience replay buffer
        self.memory = ReplayBuffer(buffer_size, state_size)

        # Training metrics
        self.training_step = 0
        self.losses = []

    def preprocess_state(self, state):
        """Preprocess state observation"""
        if isinstance(state, dict):
            state = np.array(list(state.values())).flatten()
        elif isinstance(state, (list, tuple)):
            state = np.array(state).flatten()
        elif isinstance(state, np.ndarray):
            state = state.flatten()
        else:
            state = np.array([state]).flatten()

        if len(state) != self.state_size:
            if len(state) < self.state_size:
                state = np.pad(state, (0, self.state_size - len(state)), 'constant')
            else:
                state = state[:self.state_size]

        return state.astype(np.float32)

    def act(self, state, training=True):
        """Select action using epsilon-greedy policy"""
        state = self.preprocess_state(state)

        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.action_size)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            q_values = self.q_network(state_tensor)
            action = q_values.argmax().item()

        return action

    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        state = self.preprocess_state(state)
        next_state = self.preprocess_state(next_state)
        self.memory.push(state, action, reward, next_state, done)

    def replay(self):
        """Train the network on a batch of experiences"""
        if len(self.memory) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))

        with torch.no_grad():

            # Evaluate selected actions using target network
            max_next_q_values = self.target_network(next_states).max(1)[0].unsqueeze(1)
            
            target_q_values = rewards.unsqueeze(1) + (self.gamma * max_next_q_values * (~dones).unsqueeze(1))

        loss = self.loss_fn(current_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.losses.append(loss.item())
        self.training_step += 1

        # if self.epsilon > self.epsilon_min:
        #     self.epsilon *= self.epsilon_decay

        if self.training_step % self.target_update_freq == 0:
            self.update_target_network()
            self.epsilon =max(self.epsilon_min  ,self.epsilon*self.epsilon_decay)

    def update_target_network(self):
        """Copy weights from main network to target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def save_model(self, filepath):
        """Save the trained model"""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_step': self.training_step
        }, filepath)

    def load_model(self, filepath):
        """Load a trained model"""
        checkpoint = torch.load(filepath, map_location=device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.training_step = checkpoint['training_step']

