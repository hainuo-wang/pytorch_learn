import torch
import torchvision
from PIL import Image
from torch import nn

img_path = "image/ULPS-Dog-swim-AdobeStock_184507027.jpg"
image = Image.open(img_path)
print(image)
image = image.convert("RGB")
transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                            torchvision.transforms.ToTensor()])
image = transform(image)
print(image.shape)


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


model = torch.load("heino_0.pth", map_location=torch.device("cpu"))
print(model)

image = torch.reshape(image, (1, 3, 32, 32))
model.eval()
with torch.no_grad():
    opt = model(image)
print(opt)
print(opt.argmax(1))
