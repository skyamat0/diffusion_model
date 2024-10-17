import torch
import torch.nn as nn
import torch.nn.functional as F

class ScoreNet(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.layer1 = nn.Linear(in_dim, 512)
        self.layer2 = nn.Linear(512, 256)
        self.layer3 = nn.Linear(256, 128)
        self.layer4 = nn.Linear(128, 64)
        self.output = nn.Linear(64, in_dim)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = F.relu(self.layer4(x))
        return self.output(x)

def dsm(s_theta, target):
    return ((s_theta + target) ** 2).sum(dim=1).mean()


if __name__ == "__main__":
    in_dim = 784
    sigma = 0.1
    x = torch.rand(10, in_dim)
    x_tilde = x + torch.randn_like(x) * sigma
    target = -(x_tilde - x)/(sigma**2)
    s_theta = ScoreNet(in_dim)
    print(s_theta(x))
    print(dsm(s_theta(x), target))
    
        