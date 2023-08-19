from torch import nn


# class SimpleAntiFraudGNN(nn.Module):
#     def __init__(self, input_dim: int, hidden_dim: int):
#         super(SimpleAntiFraudGNN, self).__init__()
#         self.fc1 = nn.Linear(input_dim, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, 1)
#         self.relu = nn.ReLU()
#
#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.relu(x)
#         x = self.fc2(x)
#         return x


# class SimpleAntiFraudGNN(nn.Module):
#     def __init__(self):
#         super(SimpleAntiFraudGNN, self).__init__()
#         self.fc1 = nn.Linear(10, 16)
#         self.fc2 = nn.Linear(16, 2)
#         self.relu = nn.ReLU()
#
#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.relu(x)
#         x = self.fc2(x)
#         return x

class SimpleAntiFraudGNN(nn.Module):
    def __init__(self):
        super(SimpleAntiFraudGNN, self).__init__()
        self.fc1 = nn.Linear(10, 16)
        self.fc2 = nn.Linear(16, 1)
        self.relu = nn.ReLU()
        # self.sigmoid = nn.Sigmoid()
        # self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        # x = self.sigmoid(x)
        # x = self.softmax(x)
        return x.squeeze()
