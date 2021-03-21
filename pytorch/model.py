import torch.nn as nn
import torch
import torch.nn.functional as F

class MyModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.block1 = nn.Sequential(nn.Conv2d(in_channels = 3, out_channels = 16,
                                              kernel_size = 7),
                                    nn.MaxPool2d(2, 2),
                                    nn.ReLU())

        self.block2 = nn.Sequential(nn.Conv2d(in_channels = 16, out_channels = 8,
                                              kernel_size = 5),
                                    nn.MaxPool2d(2, 2),
                                    nn.ReLU())

        # block2 gives us the tensor with a shape [batch_size, 8, 71, 71]
        self.fc1 = nn.Linear(8 * 71 * 71, 2)

    def forward(self, x: torch.Tensor):

        x = self.block1(x)
        x = self.block2(x)
        x = x.view(-1, 8 * 71 * 71)
        x = self.fc1(x)
        x = torch.softmax(x, dim = 0)

        return x
