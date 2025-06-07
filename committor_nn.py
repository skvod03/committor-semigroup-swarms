import torch
import torch.nn as nn

class CommittorNN(nn.Module):
    """
    Neural network for committor function. Predicts the probability of a configuration being in Basin B.
    """
    def __init__(self, input_size=2):
        """
        Initializes the committor network.
        
        Args:
            input_size (int): Dimension of input space (default is 2 for 2D coordinates).
        """
        super(CommittorNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 100)  # 2D input, output 100 features
        self.fc2 = nn.Linear(100, 1)  # Output a single value between 0 and 1

    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (tensor): Input coordinates (2D tensor).
        
        Returns:
            tensor: Predicted committor value between 0 and 1.
        """
        x = torch.tanh(self.fc1(x))  # Apply tanh activation to first layer
        x = torch.sigmoid(self.fc2(x))  # Apply sigmoid activation to output layer
        return x.squeeze(-1)  # Return a scalar for each input point

