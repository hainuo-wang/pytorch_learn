import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms  # 对图像进行处理的工具
from torchvision import datasets
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F  # 使用激活函数relu()的包
import torch.optim as optim  # 优化器的包


class DiabetesDataset(Dataset):
    def __init__(self, filepath):
        xy = np.loadtxt(filepath, delimiter=',', dtype=np.float32)
        self.len = xy.shape[0]  # shape(多少行，多少列)
        self.x_data = torch.from_numpy(xy[:, :-1])
        self.y_data = torch.from_numpy(xy[:, [-1]])

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


dataset = DiabetesDataset('diabetes.csv')
length = len(dataset)
train_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True, num_workers=1)  # num_workers 多线程
test_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True, num_workers=1)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.channels = channels
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        y = F.relu(self.conv1(x))
        y = self.conv2(y)
        return F.relu(x + y)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)
        self.mp = nn.MaxPool2d(2)

        self.rblock1 = ResidualBlock(16)
        self.rblock2 = ResidualBlock(32)

        self.fc = nn.Linear(512, 10)

    def forward(self, x):
        in_size = x.size(0)
        x = self.mp(F.relu(self.conv1(x)))
        x = self.rblock1(x)
        x = self.mp(F.relu(self.conv2(x)))
        x = self.rblock2(x)
        x = x.view(in_size, -1)
        x = self.fc(x)
        return x


model = Net()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
# 交叉熵损失函数
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)  # momentum动量


def train(epoch):
    running_loss = 0.0
    # 返回了数据下标和数据
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

        # 每300次输出一次
        running_loss += loss.item()
        if batch_idx % 100 == 99:
            print('[%d,%5d] loss:%.3f' % (epoch + 1, batch_idx + 1, running_loss / 300))
            running_loss = 0.0


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
        train(epoch)
        single_accuracy = test()
        total_accuracy.append(single_accuracy)
    figure = plt.figure(figsize=(8, 8))
    plt.title("RestNet")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.grid(visible=True)
    plt.plot(range(15), total_accuracy)
    plt.show()
    plt.show()
