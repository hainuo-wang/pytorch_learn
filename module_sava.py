import torch
import torchvision
from torch import nn
from torch.nn import Conv2d

vgg16 = torchvision.models.vgg16(pretrained=False)
# method1(模型结构+模型参数)
torch.save(vgg16, "vgg16_method1.pth")

# method2(模型参数)(官方推荐)
torch.save(vgg16.state_dict(), "vgg16_method2.pth")


class Heino(nn.Module):
    def __init__(self):
        super(Heino, self).__init__()
        self.conv1 = Conv2d(in_channels=3, out_channels=64, kernel_size=3)

    def forward(self, x):
        x = self.conv1(x)
        return x


heino = Heino()
torch.save(heino, "heino.method1.pth")


# 陷阱1
model = torch.load("heino.method1.pth")
print(model)  # ??
