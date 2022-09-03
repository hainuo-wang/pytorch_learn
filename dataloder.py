import os
import shutil

import torchvision.datasets

# 准备的测试数据集
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

test_data = torchvision.datasets.CIFAR10("datasetCIFAR10", train=False, transform=torchvision.transforms.ToTensor())

test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=True, num_workers=0, drop_last=False)
# 测试数据集中第一张图片及target
img, target = test_data[0]
print(img.shape)
print(target)

logs_path = "dataloder"
if os.path.exists(logs_path):
    shutil.rmtree(logs_path)
writer = SummaryWriter(logs_path)

step = 0
for data in test_loader:
    imgs, targets = data
    # print(imgs.shape)
    # print(targets)
    writer.add_images("test_data", imgs, step)
    step = step + 1
writer.close()
