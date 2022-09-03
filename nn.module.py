import torch
from torch import nn


class Heino(nn.Module):
    def __init__(self):
        super(Heino, self).__init__()  # 初始化

    def forward(self, ipt):
        opt = ipt + 1
        return opt


heino = Heino()
x = torch.tensor(1.0)
result = heino(x)
print(result)
