import torchvision

# train_data = torchvision.datasets.ImageNet("./imagenet_data", split="train",
#                                            transform=torchvision.transforms.ToTensor())
from torch import nn

vgg16_false = torchvision.models.vgg16(pretrained=False)
vgg16_true = torchvision.models.vgg16(pretrained=True)
# print(vgg16_true)

train_data = torchvision.datasets.CIFAR10("./data", train=True,
                                          transform=torchvision.transforms.ToTensor(),
                                          download=True)

vgg16_true.add_module("add_linear", nn.Linear(in_features=1000, out_features=10))  # 最后添加
print(vgg16_true)
vgg16_true.classifier.add_module("add_linear", nn.Linear(in_features=1000, out_features=10))  # 中间添加(classifier)
print(vgg16_true)

vgg16_false.classifier[6] = nn.Linear(4096, 10)  # 修改
print(vgg16_false)
