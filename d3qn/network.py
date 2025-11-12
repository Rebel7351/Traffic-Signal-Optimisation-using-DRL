import numpy as np
import torch
import torch.nn as nn
from device import device

class DuelingQNetwork(nn.Module):
    """Dueling DQN Network for traffic signal control."""
    def __init__(self, state_size, action_size, hidden_layers=[128, 128]):
        super(DuelingQNetwork, self).__init__()
        self.state_size = state_size
        self.action_size = action_size

        # Shared layers
        layers = []
        input_size = state_size
        for hidden_size in hidden_layers:
            layers.extend([
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            input_size = hidden_size
        self.feature = nn.Sequential(*layers)

        # Value and advantage streams
        self.value = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.advantage = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)

    def forward(self, state):
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).to(device)
        elif not isinstance(state, torch.Tensor):
            state = torch.FloatTensor([state]).to(device)
        if state.dim() == 1:
            state = state.unsqueeze(0)
        x = self.feature(state)
        value = self.value(x)
        advantage = self.advantage(x)
        qvals = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return qvals
