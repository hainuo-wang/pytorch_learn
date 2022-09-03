import os
import shutil

import torchvision
from torch.utils.tensorboard import SummaryWriter

dataset_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])
train_set = torchvision.datasets.CIFAR10(root="./datasetCIFAR10", train=True, transform=dataset_transform, download=True)
test_set = torchvision.datasets.CIFAR10(root="./datasetCIFAR10", train=False, transform=dataset_transform, download=True)

# print(test_set[0])
# img, target = test_set[0]
# print(img)
# print(target)
# print(test_set.classes[target])
# img.show()
# print(test_set[0])
logs_path = "p10"
if os.path.exists(logs_path):
    shutil.rmtree(logs_path)
writer = SummaryWriter(logs_path)

for i in range(10):
    img, target = test_set[i]
    print(test_set.classes[target])
    writer.add_image("test_set", img, i)
writer.close()
