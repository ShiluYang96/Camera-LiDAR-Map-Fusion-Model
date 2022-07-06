import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import numpy as np

image_dir = '/home/foj-sy/Documents/datasets/Doors/images'

data_transform = transforms.Compose([
    transforms.ToTensor(),
])

testdataset = datasets.ImageFolder(
    root=image_dir, transform=data_transform)

dataloader = torch.utils.data.DataLoader(
    testdataset, batch_size=1, shuffle=True, num_workers=2)
mean = torch.zeros(3)
std = torch.zeros(3)
print('==> Computing mean and std..')
for inputs, targets in dataloader:
    for i in range(3):
        mean[i] += inputs[:, i, :, :].mean()
        std[i] += inputs[:, i, :, :].std()
mean.div_(len(testdataset))
std.div_(len(testdataset))
print(mean)
print(std)