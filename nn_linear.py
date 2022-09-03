import torch
import torchvision
from torch import nn
from torch.nn import Linear
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10("./data_linear", train=False,
                                       transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataloader = DataLoader(dataset, batch_size=64, drop_last=True)


class Heino(nn.Module):
    def __init__(self):
        super(Heino, self).__init__()
        self.linear1 = Linear(196608, 10)

    def forward(self):
        opt = self.linear1(ipt)
        return opt


heino = Heino()

for data in dataloader:
    imgs, targets = data
    print(imgs.shape)
    # opt = torch.reshape(imgs, (1, 1, 1, -1))
    # print(opt.shape)
    opt = torch.flatten(imgs)
    result = heino(opt)
    print(result.shape)
