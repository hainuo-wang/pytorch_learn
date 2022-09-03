import torch
import torchvision
from PIL import Image
from torch import nn

image_path = "image/手写数字3.jpg"
image = Image.open(image_path)
image = image.convert("RGB")
transform = torchvision.transforms.Compose([torchvision.transforms.Resize((28, 28)),
                                            torchvision.transforms.ToTensor()])
image = transform(image)


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


model = torch.load("mnist.pth", map_location=torch.device("cpu"))
image = torch.reshape(image, (3, 1, 28, 28))
model.eval()
with torch.no_grad():
    opt = model(image)
# print(opt)
# print(opt.argmax(1))
print("\n---识别到的数字是：{}---".format(max(opt.argmax(1))))
