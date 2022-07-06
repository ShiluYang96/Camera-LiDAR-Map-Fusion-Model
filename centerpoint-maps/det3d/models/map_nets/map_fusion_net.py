import logging

import torch
import torch.nn as nn

from ..registry import MAP_NETS


# class Identity(nn.Module):
#     def __init__(self):
#         super(Identity, self).__init__()

#     def forward(self, x):
#         return x
@MAP_NETS.register_module
class MapFusionNet(nn.Module):
    """
    Simple network that returns the image data as is.
    """
    def __init__(self):
        super().__init__()
        # print("Inside BasicMapNet")
        self.name = "MapFusionNet"
        self.model = nn.Sequential(
            torch.nn.Conv2d(3, 16, (3, 3), padding=1),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 16, (3, 3), padding=1),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 32, (3, 3), padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, (3, 3), padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, (3, 3), padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, (3, 3), padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
        )

    def forward(self, data):

        # Assume that we get inputs as RGB  - (3, 1024, 1024)
        # Basic Map Net simply returns the map image as is (we assume the image is stores as 128x128)
        # print("Inside BasicMapNet.forward")
        data = torch.Tensor(data)
        data = data.to(torch.device("cuda"))
        data = data.view(-1, 3, 128, 128)
        # print("shape of data: ", data.shape)

        # output should be 128 channels of shape (128 x 128)
        output = self.model(data)
        return output


