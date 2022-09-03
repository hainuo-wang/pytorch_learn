import os
import shutil
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda")

train = pd.read_csv('train.csv')
# print("Shape:", train.shape)

test = pd.read_csv('test.csv')
# print("Shape:", test.shape)

# Counter(train['OutcomeType'])
# Counter(train['Name']).most_common(5)
train_X = train.drop(columns=['OutcomeType', 'OutcomeSubtype', 'AnimalID'])
Y = train['OutcomeType']
test_X = test
stacked_df = train_X.append(test_X.drop(columns=['ID']))
stacked_df = stacked_df.drop(columns=['DateTime'])

for col in stacked_df.columns:
    if stacked_df[col].isnull().sum() > 10000:
        # print("dropping", col, stacked_df[col].isnull().sum())
        stacked_df = stacked_df.drop(columns=[col])
for col in stacked_df.columns:
    if stacked_df.dtypes[col] == "object":
        stacked_df[col] = stacked_df[col].fillna("NA")
    else:
        stacked_df[col] = stacked_df[col].fillna(0)
    stacked_df[col] = LabelEncoder().fit_transform(stacked_df[col])
# making all variables categorical
for col in stacked_df.columns:
    stacked_df[col] = stacked_df[col].astype('category')
X = stacked_df[0:26729]
test_processed = stacked_df[26729:]
# check if shape[0] matches original
# print("train shape: ", X.shape, "orignal: ", train.shape)
# print("test shape: ", test_processed.shape, "original: ", test.shape)
Y = LabelEncoder().fit_transform(Y)
# sanity check to see numbers match and matching with previous counter to create target dictionary
# print(Counter(train['OutcomeType']))
# print(Counter(Y))
target_dict = {
    'Return_to_owner': 3,
    'Euthanasia': 2,
    'Adoption': 0,
    'Transfer': 4,
    'Died': 1
}
X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.10, random_state=0)

# categorical embedding for columns having more than two values
embedded_cols = {n: len(col.cat.categories) for n, col in X.items() if len(col.cat.categories) > 2}
embedded_col_names = embedded_cols.keys()
len(X.columns) - len(embedded_cols)  # number of numerical columns
embedding_sizes = [(n_categories, min(50, (n_categories + 1) // 2)) for _, n_categories in embedded_cols.items()]

print("X:", type(X), X.shape)
print("Y:", type(Y), Y.shape)


class ShelterOutcomeDataset(Dataset):
    def __init__(self, X, Y, embedded_col_names):
        X = X.copy()
        self.X1 = X.loc[:, embedded_col_names].copy().values.astype(np.int64)  # categorical columns
        self.X2 = X.drop(columns=embedded_col_names).copy().values.astype(np.float32)  # numerical columns
        self.y = Y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):  # 返回单条数据
        return self.X1[idx], self.X2[idx], self.y[idx]


# creating train and valid datasets
train_ds = ShelterOutcomeDataset(X_train, y_train, embedded_col_names)
valid_ds = ShelterOutcomeDataset(X_val, y_val, embedded_col_names)

train_data_size = len(train_ds)
valid_data_size = len(valid_ds)

train_dataloader = DataLoader(train_ds, batch_size=512, shuffle=True)
valid_ds_dataloader = DataLoader(valid_ds, batch_size=512, shuffle=True)


class ShelterOutcomeModel(nn.Module):
    def __init__(self, embedding_sizes, n_cont):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(categories, size) for categories, size in embedding_sizes])
        n_emb = sum(e.embedding_dim for e in self.embeddings)  # length of all embeddings combined
        self.n_emb, self.n_cont = n_emb, n_cont
        self.lin1 = nn.Linear(self.n_emb + self.n_cont, 200)
        self.lin2 = nn.Linear(200, 70)
        self.lin3 = nn.Linear(70, 5)
        self.bn1 = nn.BatchNorm1d(self.n_cont)
        self.bn2 = nn.BatchNorm1d(200)
        self.bn3 = nn.BatchNorm1d(70)
        self.emb_drop = nn.Dropout(0.6)
        self.drops = nn.Dropout(0.3)

    def forward(self, x_cat, x_cont):
        x = [e(x_cat[:, i]) for i, e in enumerate(self.embeddings)]
        x = torch.cat(x, 1)
        x = self.emb_drop(x)
        x2 = self.bn1(x_cont)
        x = torch.cat([x, x2], 1)
        x = F.relu(self.lin1(x))
        x = self.drops(x)
        x = self.bn2(x)
        x = F.relu(self.lin2(x))
        x = self.drops(x)
        x = self.bn3(x)
        x = self.lin3(x)
        return x


model = ShelterOutcomeModel(embedding_sizes, 1)
model = model.to(device)

loss_func = nn.CrossEntropyLoss()
loss_func = loss_func.to(device)

learning_rate = 1e-3
optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=0.0)


def train(epoch):
    running_loss = 0.0
    # 返回了数据下标和数据
    for batch_idx, data in enumerate(train_dataloader, 0):
        # 送入两个张量，一个张量是64个图像的特征，一个张量图片对应的数字
        x1, x2, y = data
        # 把输入输出迁入GPU
        x1, x2, y = x1.to(device), x2.to(device), y.to(device)
        # 梯度归零
        optimizer.zero_grad()
        # forward+backward+update
        outputs = model(x1, x2)
        # 计算损失，用的交叉熵损失函数
        loss = loss_func(outputs, y.long())
        # 反馈
        loss.backward()
        # 随机梯度下降更新
        optimizer.step()

        # 每300次输出一次
        running_loss += loss.item()
        if batch_idx % 5 == 0:
            print('[%d,%5d] loss:%.3f' % (epoch + 1, batch_idx + 1, running_loss / 300))
            running_loss = 0.0


def test():
    correct = 0
    total = 0
    # 不会计算梯度
    with torch.no_grad():
        for data in valid_ds_dataloader:  # 拿数据
            x1, x2, y = data
            # 把输入输出迁入GPU
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)
            outputs = model(x1, x2)  # 预测
            # outputs.data是一个矩阵，每一行10个量，最大值的下标就是预测值
            _, predicted = torch.max(outputs.data, dim=1)  # 沿着第一维度，找最大值的下标，返回最大值和下标
            total += y.size(0)  # labels.size(0)=64 每个都是64个元素，就可以计算总的元素
            # (predicted == labels).sum()这个是张量，而加了item()变为一个数字，即相等的数量
            correct += (predicted == y).sum().item()
    print('Accuracy on test set:%d %%' % (100 * correct / total))  # 正确的数量除以总数
    return 100 * correct / total


if __name__ == '__main__':
    total_accuracy = []
    for epoch in range(15):
        train(epoch)
        single_accuracy = test()
        total_accuracy.append(single_accuracy)
    figure = plt.figure(figsize=(8, 8))
    plt.title("ShelterAnimals")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.grid(visible=True)
    plt.plot(range(15), total_accuracy)
    plt.show()
