import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt


class CustomedDataSet(Dataset):
    def __init__(self, train=True, train_x=None, train_y=None, test_x=None, test_y=None, val=False, transform=None):
        self.train = train
        self.val = val
        self.transform = transform
        if self.train:
            self.dataset = train_x
            self.labels = train_y
        elif val:
            self.dataset = test_x
            self.labels = test_y
        else:
            self.dataset = test_x

    def __getitem__(self, index):
        if self.train:
            return torch.Tensor(self.dataset[index].astype(float)).to(device), self.labels[index].to(device)
        elif self.val:
            return torch.Tensor(self.dataset[index].astype(float)).to(device), self.labels[index].to(device)
        else:
            return torch.Tensor(self.dataset[index].astype(float)).to(device)

    def __len__(self):
        return self.dataset.shape[0]
