import torch

# method1
import torchvision.models

model1 = torch.load("vgg16_method1.pth")
model2 = torch.load("vgg16_method2.pth")
print(model1)
print(model2)

# method2
vgg16 = torchvision.models.vgg16(pretrained=False)
model3 = vgg16.load_state_dict(torch.load("vgg16_method2.pth"))
print(model3)
