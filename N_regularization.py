import torch
import torch.nn as nn

class N(nn.Module):
    def __init__(self):
        super(N, self).__init__()

        # 4 encoding steps pytorch [batch, channel, x,y,z...]
        # our data is stored as [batch, z, y, x]
        self.network_layers = nn.Sequential(
                                nn.Conv3d(4, 16, 3, 2, 1),
                                nn.Conv3d(16, 16, 3, 2, 1),
                                nn.ReLU(),
                                nn.Conv3d(16, 16, 3, 2, 1),
                                nn.ReLU()
        )

    def forward(self, input):
        x = self.network_layers(input)

        return x.mean()
