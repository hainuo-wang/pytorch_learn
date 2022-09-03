import os
import shutil

import torch
import torchvision.datasets
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("datasetCIFAR10", train=False,
                                       transform=torchvision.transforms.ToTensor(), download=True)
dataloader = DataLoader(dataset, batch_size=64)
# ipt = torch.tensor([[1, 2, 0, 3, 1],
#                     [0, 1, 2, 3, 1],
#                     [1, 2, 1, 0, 0],
#                     [5, 2, 3, 1, 1],
#                     [2, 1, 0, 1, 1]], dtype=torch.float32)
# ipt = torch.reshape(ipt, (-1, 1, 5, 5))
# print(ipt.shape)


class Heino(nn.Module):
    def __init__(self):
        super(Heino, self).__init__()
        self.maxpool1 = MaxPool2d(kernel_size=3, ceil_mode=True)

    def forward(self, ipt):
        opt = self.maxpool1(ipt)
        return opt


heino = Heino()
# result = heino(ipt)
# print(result)
logs_path = "logs_maxpool"
if os.path.exists(logs_path):
    shutil.rmtree(logs_path)
writer = SummaryWriter(logs_path)
step = 0
for data in dataloader:
    imgs, target = data
    writer.add_images("input_maxpool", imgs, step)
    opt = heino(imgs)
    writer.add_images("output_maxpool", opt, step)
    step += 1
writer.close()

