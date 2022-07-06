import logging

import torch
import torch.nn as nn

from ..registry import MAP_NETS


@MAP_NETS.register_module
class BasicMapNet(nn.Module):
    """
    Simple network that returns the image data as is.
    """
    def __init__(self):
        super().__init__()
        # print("Inside BasicMapNet")
        self.name = "BasicMapNet"
        

    def forward(self, data):

        # Basic Map Net simply returns the map image as is (we assume the image is stores as 128x128)
        # print("Inside BasicMapNet.forward")
        data = torch.Tensor(data)
        data = data.to(torch.device("cuda"))
        data = data.view(-1, 3, 128, 128)
        # print("shape of data: ", data.shape)
        return data

