import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt

torch.manual_seed(1)
# 设置超参数
epoches = 2
batch_size = 50
learning_rate = 0.001


# 搭建CNN
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()  # 继承__init__功能
        ## 第一层卷积
        self.conv1 = nn.Sequential(
            # 输入[1,28,28]
            nn.Conv2d(
                in_channels=1,  # 输入图片的高度
                out_channels=16,  # 输出图片的高度
                kernel_size=5,  # 5x5的卷积核，相当于过滤器
                stride=1,  # 卷积核在图上滑动，每隔一个扫一次
                padding=2,  # 给图外边补上0
            ),
            # 经过卷积层 输出[16,28,28] 传入池化层
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # 经过池化 输出[16,14,14] 传入下一个卷积
        )
        ## 第二层卷积
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,  # 同上
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            # 经过卷积 输出[32, 14, 14] 传入池化层
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # 经过池化 输出[32,7,7] 传入输出层
        )
        ## 输出层
        self.output = nn.Linear(in_features=32 * 7 * 7, out_features=10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)  # [batch, 32,7,7]
        x = x.view(x.size(0), -1)  # 保留batch, 将后面的乘到一起 [batch, 32*7*7]
        output = self.output(x)  # 输出[50,10]
        return output


# 下载MNist数据集
train_data = torchvision.datasets.MNIST(
    root="./mnist/",  # 训练数据保存路径
    train=True,
    transform=torchvision.transforms.ToTensor(),  # 数据范围已从(0-255)压缩到(0,1)
    download=False,  # 是否需要下载
)
print(train_data.train_data.size())   # [60000,28,28]
print(train_data.train_labels.size())  # [60000]
# plt.imshow(train_data.train_data[0].numpy())
# plt.show()

test_data = torchvision.datasets.MNIST(
    root="./mnist/",  # 训练数据保存路径
    train=False,
    transform=torchvision.transforms.ToTensor(),  # 数据范围已从(0-255)压缩到(0,1)
    download=False,  # 是否需要下载
)
print(test_data.test_data.size())  # [10000, 28, 28]
# print(test_data.test_labels.size())  # [10000, 28, 28]
test_x = torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor)[:2000] / 255
test_y = test_data.test_labels[:2000]

# 装入Loader中
train_loader = Data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=3)


def main():
    # cnn 实例化
    cnn = CNN()
    print(cnn)

    # 定义优化器和损失函数
    optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)
    loss_function = nn.CrossEntropyLoss()

    # 开始训练
    for epoch in range(epoches):
        print("进行第{}个epoch".format(epoch))
        for step, (batch_x, batch_y) in enumerate(train_loader):
            output = cnn(batch_x)  # batch_x=[50,1,28,28]
            # output = output[0]
            loss = loss_function(output, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 50 == 0:
                test_output = cnn(test_x)  # [10000 ,10]
                pred_y = torch.max(test_output, 1)[1].data.numpy()
                # accuracy = sum(pred_y==test_y)/test_y.size(0)
                accuracy = ((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
                print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)

    test_output = cnn(test_x[:10])
    pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
    print(pred_y)
    print(test_y[:10])


if __name__ == "__main__":
    main()
