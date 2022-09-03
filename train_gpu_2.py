import sys

import torch.optim
import torchvision.datasets
from torch.utils.tensorboard import SummaryWriter

# 准备数据集
from torch import nn
from torch.utils.data import DataLoader
# 定义训练的设备
from tqdm import tqdm

device = torch.device("cuda")
train_data = torchvision.datasets.CIFAR10(root="CIFAR10", train=True,
                                          transform=torchvision.transforms.ToTensor(),
                                          download=True)
test_data = torchvision.datasets.CIFAR10(root="CIFAR10", train=False,
                                         transform=torchvision.transforms.ToTensor(),
                                         download=True)
# length
train_data_size = len(train_data)
test_data_size = len(test_data)
print("训练的数据集的长度为：{}".format(train_data_size))
# print(f"训练的数据集的长度为：{train_data_size}")
print("测试的数据集的长度为：{}".format(test_data_size))

# 利用dataloader加载数据集
train_dataloader = DataLoader(dataset=train_data, batch_size=64)
test_dataloader = DataLoader(dataset=test_data, batch_size=64)

# 搭建神经网络


class Heino(nn.Module):
    def __init__(self):
        super(Heino, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(in_features=64 * 4 * 4, out_features=64),
            nn.Linear(in_features=64, out_features=10)
        )

    def forward(self, x):
        x = self.model(x)
        return x


# 创建网络模型
heino = Heino()
heino = heino.to(device)
# 损失函数
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)
# 优化器
learning_rate = 1e-3
optimizer = torch.optim.SGD(params=heino.parameters(), lr=learning_rate)

# 设置训练网络的一些参数
# 记录训练次数
total_train_step = 0
# 记录测试次数
total_test_step = 0
# 训练轮数
epoch = 100

# 添加tensorboard
writer = SummaryWriter("logs_train")

for i in range(epoch):
    running_loss = 0.0
    times = 0
    # 训练步骤开始
    heino.train()
    train_dataloader = tqdm(train_dataloader, desc="train", file=sys.stdout, colour="Green")
    for data in train_dataloader:
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        opts = heino(imgs)
        loss = loss_fn(opts, targets)  # 损失函数，看损失值
        # 优化器优化模型
        optimizer.zero_grad()  # 梯度清零
        loss.backward()  # 反向传播，得到每一个参数结点的梯度
        optimizer.step()  # 优化，一次训练结束

        total_train_step += 1
        running_loss += loss.item()
        times += 1
    print('epoch:%2d  loss:%.3f' % (i + 1, running_loss / times))
    # 测试步骤开始
    heino.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        test_dataloader = tqdm(test_dataloader, desc="test ", file=sys.stdout, colour="red")
        for data in test_dataloader:
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            opts = heino(imgs)
            loss = loss_fn(opts, targets)
            total_test_loss += loss
            accuracy = (opts.argmax(1) == targets).sum()
            total_accuracy += accuracy
    print('Accuracy on test set:%d %%' % (100 * total_accuracy / test_data_size))
    writer.add_scalar("test_loss", total_test_loss, total_train_step)
    writer.add_scalar("test_accuracy", total_accuracy / test_data_size, total_test_step)
    total_test_step += 1
    # torch.save(heino, "heino_{}.pth".format(i))
    # torch.save(heino.state+dict(), "heino_{}.pth".format(i))
    # print("模型已保存")
writer.close()
