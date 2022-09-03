from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

writer = SummaryWriter("logs")
img = Image.open("hymenoptera_data/train/ants/998118368_6ac1d91f81.jpg")
print(img)
# ToTensor
trans_totensor = transforms.ToTensor()
img_tensor = trans_totensor(img)
writer.add_image("ToTensor", img_tensor)

# Normalize
print(img_tensor[0][0][0])
trans_norm = transforms.Normalize([1, 3, 5], [3, 2, 1])
img_norm = trans_norm(img_tensor)
print(img_norm[0][0][0])
writer.add_image("Normalize", img_norm, 2)

# Resize
print(img.size)
trans_resize = transforms.Resize((512, 512))
img_resize = trans_resize(img)
# PIL->tensor
img_resize = trans_totensor(img_resize)
writer.add_image("Resize", img_resize)
print(img_resize)

# Compose
trans_resize_2 = transforms.Resize(512)
# PIL -> PIL-> tensor
trans_compose = transforms.Compose([trans_resize_2, trans_totensor])
trans_resize_2 = trans_compose(img)
writer.add_image("Resize2", trans_resize_2)

writer.close()
