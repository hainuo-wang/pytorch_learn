import os
import shutil

import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda")


# prepare dataset
class DiabetesDataset(Dataset):
    def __init__(self, filepath):
        xy = np.loadtxt(filepath, delimiter=',', dtype=np.float32)
        self.len = xy.shape[0]  # shape(多少行，多少列)
        self.x_data = torch.from_numpy(xy[:, :-1])
        self.y_data = torch.from_numpy(xy[:, [-1]])

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


dataset = DiabetesDataset('diabetes.csv', )
length = len(dataset)
train_dataloader = DataLoader(dataset=dataset, batch_size=32, shuffle=True, num_workers=1)  # num_workers 多线程
valid_ds_dataloader = DataLoader(dataset=dataset, batch_size=32, shuffle=True, num_workers=1)


# design model using class
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(8, 6)
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x


model = Model()
model = model.to(device)
# construct loss and optimizer
loss_func = torch.nn.BCELoss(reduction='mean')
loss_func = loss_func.to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# training cycle forward, backward, update
if __name__ == '__main__':
    epoch = 50
    total_train_step = 0
    total_valid_step = 0
    logs_path = "logs_diabetes"
    if os.path.exists(logs_path):
        shutil.rmtree(logs_path)
    writer = SummaryWriter(logs_path)

    for i in range(epoch):
        print("------第{}轮训练开始------".format(i + 1))
        model.train()
        for data in train_dataloader:
            x, y = data
            x = x.to(device)
            y = y.to(device)
            opt = model(x)
            loss = loss_func(opt, y.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_step += 1

            if total_train_step % 5 == 0:
                print("训练次数:{},loss:{}".format(total_train_step, loss))
                writer.add_scalar("train_loss", loss, total_train_step)
        model.eval()
        total_valid_loss = 0
        total_valid_step = 0
        total_accuracy = 0
        sum_loss = 0
        total_batch_size = 0
        current = 0
        with torch.no_grad():
            for data in valid_ds_dataloader:
                x, y = data
                current_batch_size = y.shape[0]
                x = x.to(device)
                y = y.to(device)

                opt = model(x)
                loss = loss_func(opt, y.float())
                total_valid_loss += loss
                sum_loss += current_batch_size * (loss.item())
                total_batch_size += current_batch_size
                pred = torch.max(opt, 1)[1]
                current += (pred == y).float().sum()
            print("valid loss: %.3f and accuracy: %.3f" % (sum_loss / total_batch_size, current / total_batch_size))
            writer.add_scalar("valid loss", sum_loss / total_batch_size, total_valid_step)
            writer.add_scalar("accuracy", current / length, total_valid_step)
    writer.close()
