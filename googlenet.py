import sys

import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torchvision import transforms  # 对图像进行处理的工具
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F  # 使用激活函数relu()的包
import torch.optim as optim  # 优化器的包
from tqdm import tqdm

batch_size = 64
# 对图像进行预处理，将图像转换为
transform = transforms.Compose([
    # 将原始图像PIL变为张量tensor(H*W*C),再将[0,255]区间转换为[0.1,1.0]
    transforms.ToTensor(),
    # 使用均值和标准差对张量图像进行归一化
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='dataset/mnist/', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)

test_dataset = datasets.MNIST(root='dataset/mnist/', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)


class InceptionA(nn.Module):
    def __init__(self, in_channels):
        super(InceptionA, self).__init__()
        # 第一个通道，输入通道为in_channels,输出通道为16，卷积盒的大小为1*1的卷积层
        self.branch1x1 = nn.Conv2d(in_channels, 16, kernel_size=1)

        # 第二个通道
        self.branch5x5_1 = nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch5x5_2 = nn.Conv2d(16, 24, kernel_size=5, padding=2)

        # 第三个通道
        self.branch3x3_1 = nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch3x3_2 = nn.Conv2d(16, 24, kernel_size=3, padding=1)
        self.branch3x3_3 = nn.Conv2d(24, 24, kernel_size=3, padding=1)

        # 第四个通道
        self.branch_pool = nn.Conv2d(in_channels, 24, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)
        branch3x3 = self.branch3x3_3(branch3x3)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        # 拼接
        outputs = [branch1x1, branch5x5, branch3x3, branch_pool]
        return torch.cat(outputs, dim=1)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(88, 20, kernel_size=5)

        self.incep1 = InceptionA(in_channels=10)
        self.incep2 = InceptionA(in_channels=20)

        self.mp = nn.MaxPool2d(2)
        self.fc = nn.Linear(1408, 10)

    def forward(self, x):
        in_size = x.size(0)
        x = F.relu(self.mp(self.conv1(x)))
        x = self.incep1(x)
        x = F.relu(self.mp(self.conv2(x)))
        x = self.incep2(x)
        x = x.view(in_size, -1)
        x = self.fc(x)
        return x


model = Net()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
# 交叉熵损失函数
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)  # momentum动量


def train(epoch, train_loader):
    running_loss = 0.0
    # 返回了数据下标和数据
    times = 0
    train_loader = tqdm(train_loader, file=sys.stdout, colour="Green")
    for batch_idx, data in enumerate(train_loader, 0):
        # 送入两个张量，一个张量是64个图像的特征，一个张量图片对应的数字
        inputs, target = data
        # 把输入输出迁入GPU
        inputs, target = inputs.to(device), target.to(device)
        # 梯度归零
        optimizer.zero_grad()

        # forward+backward+update
        outputs = model(inputs)
        # 计算损失，用的交叉熵损失函数
        loss = criterion(outputs, target)
        # 反馈
        loss.backward()
        # 随机梯度下降更新
        optimizer.step()

        running_loss += loss.item()
        times += 1

    print('epoch:%2d  loss:%.3f' % (epoch + 1, running_loss / times))


def test():
    correct = 0
    total = 0
    # 不会计算梯度
    with torch.no_grad():
        for data in test_loader:  # 拿数据
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)  # 预测
            # outputs.data是一个矩阵，每一行10个量，最大值的下标就是预测值
            _, predicted = torch.max(outputs.data, dim=1)  # 沿着第一维度，找最大值的下标，返回最大值和下标
            total += labels.size(0)  # labels.size(0)=64 每个都是64个元素，就可以计算总的元素
            # (predicted == labels).sum()这个是张量，而加了item()变为一个数字，即相等的数量
            correct += (predicted == labels).sum().item()
    print('Accuracy on test set:%d %%' % (100 * correct / total))  # 正确的数量除以总数
    return 100 * correct / total


if __name__ == '__main__':
    total_accuracy = []
    for epoch in range(15):
        train(epoch, train_loader)
        single_accuracy = test()
        total_accuracy.append(single_accuracy)
    figure = plt.figure(figsize=(8, 8))
    plt.title("GoogLeNet")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.grid(visible=True)
    plt.plot(range(15), total_accuracy)
    plt.show()
