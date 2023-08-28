
""" Model definition """
from torch.nn import (
    Dropout,
    Linear,
    LSTM,
    Module,
    ReLU,
    Softmax
)

class LSTMClassifier(Module):
    """
    A PyTorch LSTM implementation.

    Methods:
        __init__(self, input_dim, hidden_dim, output_dim):
            Initializes the neural network.

        forward(self, x):
            Defines the forward pass of the neural network.
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        Initialize the neural network module.

        Args:
            input_dim (int): Number of input features.
            hidden_dim (int): Number of hidden nodes in the neural network.
            output_dim (int): Number of output nodes in the neural network.
        """
        super(LSTMClassifier, self).__init__()
        self.lstm = LSTM(input_dim, hidden_dim, batch_first=True)
        self.relu = ReLU()
        self.dropout = Dropout(0.2)
        self.fc = Linear(hidden_dim, output_dim)
        self.softmax = Softmax(dim=1)

    def forward(self, x):
        """
        Forward pass of the PyTorch module.

        Args:
            x (torch.Tensor): Input tensor to the module.

        Returns:
            torch.Tensor: Output tensor after applying the forward computation.
        """
        lstm_output, _ = self.lstm(x)
        x = lstm_output[:, -1, :] # only take the last timestamp's state (since LSTM is recursive)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.softmax(x)
        return x
