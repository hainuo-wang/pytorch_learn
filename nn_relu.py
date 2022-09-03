import os
import shutil

import torch
import torchvision
from torch import nn
from torch.nn import ReLU, Sigmoid
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

ipt = torch.tensor([[1, -0.5],
                    [-1, 3]])
ipt = torch.reshape(ipt, (-1, 1, 2, 2))
print(ipt.shape)

dataset = torchvision.datasets.CIFAR10("./data_relu", train=False,
                                       transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataloader = DataLoader(dataset, batch_size=64)

class Heino(nn.Module):
    def __init__(self):
        super(Heino, self).__init__()
        self.relu1 = ReLU(inplace=False)
        self.sigmod1 = Sigmoid()

    def forward(self, ipt):
        opt = self.sigmod1(ipt)
        return opt


heino = Heino()
# opt = heino(ipt)
# print(opt)
logs_path = "logs_relu"
if os.path.exists(logs_path):
    shutil.rmtree(logs_path)
writer = SummaryWriter(logs_path)
step = 0
for data in dataloader:
    imgs, targets = data
    writer.add_images("input_sigmoid", imgs, step)
    opt = heino(imgs)
    writer.add_images("output_sigmoid", opt, step)
    step += 1

writer.close()
