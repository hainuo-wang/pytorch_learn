import os
import shutil

from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

img_path = "hymenoptera_data/train/ants/0013035.jpg"
img = Image.open(img_path)

logs_path = "logs"
if os.path.exists(logs_path):
    shutil.rmtree(logs_path)
writer = SummaryWriter(logs_path)

tensor_trans = transforms.ToTensor()  # 实例化
tensor_img = tensor_trans(img)

writer.add_image("Tensor_img", tensor_img)
writer.close()
