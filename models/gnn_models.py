from torch import nn


class SimpleAntiFraudGNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super(SimpleAntiFraudGNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x