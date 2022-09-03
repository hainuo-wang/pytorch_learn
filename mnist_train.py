import os
import shutil
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

# 定义训练设备
from tqdm import tqdm

device = torch.device("cuda")
# 准备训练集、测试集
train_dataset = datasets.MNIST(root="dataset/mnist/", train=True, download=True,
                               transform=transforms.ToTensor())
test_dataset = datasets.MNIST(root="dataset/mnist/", train=False, download=True,
                              transform=transforms.ToTensor())
# len
train_data_size = len(train_dataset)
test_data_size = len(test_dataset)
# 用dataloader加载数据
train_dataloader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=True)


# 搭建网络模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Sequential(torch.nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
                                         torch.nn.ReLU(),
                                         torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                                         torch.nn.ReLU(),
                                         torch.nn.MaxPool2d(stride=2, kernel_size=2))
        self.dense = torch.nn.Sequential(torch.nn.Linear(14 * 14 * 128, 1024),
                                         torch.nn.ReLU(),
                                         torch.nn.Dropout(p=0.5),
                                         torch.nn.Linear(1024, 10))

    def forward(self, x):
        x = self.conv1(x)
        x = x.view(-1, 14 * 14 * 128)
        x = self.dense(x)
        return x


# 创建网络模型
net = Net()
net = net.to(device)
# 定义损失函数
loss_func = nn.CrossEntropyLoss()
loss_func = loss_func.to(device)
# 定义优化器
learning_rate = 1e-3
optimizer = torch.optim.Adam(params=net.parameters(), lr=learning_rate)
# 设置训练网络的一些参数
# 记录训练次数
total_train_step = 0
# 记录测试次数
total_test_step = 0
# 训练轮数
epoch = 5
# 添加tensorboard
logs_path = "logs_mnist"
if os.path.exists(logs_path):
    shutil.rmtree(logs_path)
writer = SummaryWriter(logs_path)

for i in range(epoch):
    running_loss = 0.0
    times = 0
    # 训练步骤开始
    net.train()
    train_dataloader = tqdm(train_dataloader, desc="train", file=sys.stdout, colour="Green")
    net.train()
    for data in train_dataloader:
        images, targets = data
        images = images.to(device)
        targets = targets.to(device)
        opt = net(images)
        loss = loss_func(opt, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 一轮训练结束
        total_train_step += 1
        running_loss += loss.item()
        times += 1
    print('epoch:%2d  loss:%.3f' % (i + 1, running_loss / times))
    net.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        test_dataloader = tqdm(test_dataloader, desc="test ", file=sys.stdout, colour="red")
        for data in test_dataloader:
            images, targets = data
            images = images.to(device)
            targets = targets.to(device)
            opt = net(images)
            loss = loss_func(opt, targets)
            total_test_loss += loss
            accuracy = (opt.argmax(1) == targets).sum()
            total_accuracy += accuracy
        print('Accuracy on test set:%d %%' % (100 * total_accuracy / test_data_size))
        writer.add_scalar("test_loss", total_test_loss, total_train_step)
        writer.add_scalar("test_accuracy", total_accuracy / test_data_size, total_test_step)
        if (total_accuracy / test_data_size) >= 0.99:
            torch.save(net, "mnist.pth")
        total_test_step += 1
