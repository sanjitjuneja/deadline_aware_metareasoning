import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForwardNN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(FeedForwardNN, self).__init__()

        self.layer1 = nn.Linear(in_dim, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, out_dim)

    def forward(self, obs):
        # Ensure obs is a torch tensor
        if not isinstance(obs, torch.Tensor):
            device = torch.device("cpu")
            obs = torch.tensor(obs, dtype=torch.float).to(device)

        activation1 = F.relu(self.layer1(obs))
        activation2 = F.relu(self.layer2(activation1))
        output = self.layer3(activation2)

        output = F.softmax(output, dim=0)

        return output
