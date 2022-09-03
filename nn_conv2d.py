import os
import shutil

import torch
import torchvision
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("../data", train=False,
                                       transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataloader = DataLoader(dataset, batch_size=64)


class Heino(nn.Module):
    def __init__(self):
        super(Heino, self).__init__()
        self.conv1 = Conv2d(in_channels=3, out_channels=6,
                            kernel_size=3, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        return x


heino = Heino()

logs_path = "logs"
if os.path.exists(logs_path):
    shutil.rmtree(logs_path)
writer = SummaryWriter(logs_path)
step = 0

# print(heino)
for data in dataloader:
    imgs, targets = data
    opt = heino(imgs)
    print(imgs.shape)
    print(opt.shape)
    # torch.Size([64, 3, 32, 32])
    # torch.Size([64, 6, 30, 30])  ->  [xx, 3, 30, 30]
    opt = torch.reshape(opt, (-1, 3, 30, 30))
    writer.add_images("opt", opt, step)
    step += 1
