import torch.optim
import torchvision.datasets
from torch.utils.tensorboard import SummaryWriter

from model import *
# 准备数据集
from torch import nn
from torch.utils.data import DataLoader

train_data = torchvision.datasets.CIFAR10(root="./data1", train=True,
                                          transform=torchvision.transforms.ToTensor(),
                                          download=True)
test_data = torchvision.datasets.CIFAR10(root="./data1", train=False,
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

# 创建网络模型
heino = Heino()
# 损失函数
loss_fn = nn.CrossEntropyLoss()
# 优化器
learning_rate = 1e-2
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
    print("------第{}轮训练开始-------".format(i + 1))
    # 训练步骤开始
    heino.train()
    for data in train_dataloader:
        imgs, targets = data
        opts = heino(imgs)
        loss = loss_fn(opts, targets)  # 损失函数，看损失值
        # 优化器优化模型
        optimizer.zero_grad()  # 梯度清零
        loss.backward()  # 反向传播，得到每一个参数结点的梯度
        optimizer.step()  # 优化，一次训练结束

        total_train_step += 1
        if total_train_step % 100 == 0:
            print("训练次数：{}，loss：{}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)
    # 测试步骤开始
    heino.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            opts = heino(imgs)
            loss = loss_fn(opts, targets)
            total_test_loss += loss
            accuracy = (opts.argmax(1) == targets).sum()
            total_accuracy += accuracy
    print("整体测试集上的loss：{}".format(total_test_loss.item()))
    print("整体测试集上的正确率：{}".format(total_accuracy / test_data_size))
    writer.add_scalar("test_loss", total_test_loss, total_train_step)
    writer.add_scalar("test_accuracy", total_accuracy / test_data_size, total_test_step)
    total_test_step += 1
    torch.save(heino, "heino_{}.pth".format(i))
    # torch.save(heino.state+dict(), "heino_{}.pth".format(i))
    print("模型已保存")
writer.close()
