import torchvision
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10("./data_linear", train=False,
                                       transform=torchvision.transforms.ToTensor(),
                                       download=True)
dataloader = DataLoader(dataset, batch_size=64)


class Heino(nn.Module):
    def __init__(self):
        super(Heino, self).__init__()
        self.module1 = Sequential(
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self, x):
        x = self.module1(x)
        return x


loss = nn.CrossEntropyLoss()
heino = Heino()
for data in dataloader:
    imgs, targets = data
    opt = heino(imgs)
    # print(opt)
    # print(targets)
    result_loss = loss(opt, targets)
    print(result_loss)
    result_loss.backward()
    print("ok")
