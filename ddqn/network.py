import numpy as np
import torch
import torch.nn as nn
import random
from device import device
class QNetwork(nn.Module):
    """Deep Q-Network for traffic signal control"""
    
    def __init__(self, state_size, action_size, hidden_layers=[128, 128]):
        super(QNetwork, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        # Build network layers
        layers = []
        input_size = state_size
        
        for hidden_size in hidden_layers:
            layers.extend([
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            input_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(input_size, action_size))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier uniform initialization"""
        for module in self.network:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, state):
        """Forward pass through the network"""
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).to(device)
        elif not isinstance(state, torch.Tensor):
            state = torch.FloatTensor([state]).to(device)
        
        if state.dim() == 1:
            state = state.unsqueeze(0)
            
        return self.network(state)
