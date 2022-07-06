import logging

import torch
import torch.nn as nn


import segmentation_models_pytorch as smp

from ..registry import MAP_NETS

# class Identity(nn.Module):
#     def __init__(self):
#         super(Identity, self).__init__()
        
#     def forward(self, x):
#         return x

@MAP_NETS.register_module
class UnetMapNet(nn.Module):
    """
    Simple network that returns the image data as is.
    """
    def __init__(self):
        super().__init__()
        # print("Inside BasicMapNet")
        self.name = "UnetMapNet"
        self.model = smp.Unet(
            encoder_name = 'resnet18',
            encoder_depth = 5,
            encoder_weights = 'imagenet',
            decoder_use_batchnorm = True,
            decoder_channels = (256, 128, 64, 32, 16),
            in_channels = 3, 
        )
        self.model.decoder.blocks = self.model.decoder.blocks[:2]
        self.model.segmentation_head = nn.Identity()
        

    def forward(self, data):

        # Assume that we get inputs as RGB  - (3, 1024, 1024)
        # Basic Map Net simply returns the map image as is (we assume the image is stores as 128x128)
        # print("Inside BasicMapNet.forward")
        data = torch.Tensor(data)
        data = data.to(torch.device("cuda"))
        data = data.view(-1, 3, 1024, 1024)
        # print("shape of data: ", data.shape)

        # output should be 128 channels of shape (128 x 128)
        output = self.model(data)
        
        return output

