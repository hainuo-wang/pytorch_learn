"""
View more, visit my tutorial page: https://mofanpy.com/tutorials/
My Youtube Channel: https://www.youtube.com/user/MorvanZhou
Dependencies:
torch: 0.4
matplotlib
numpy
"""
import numpy
import pandas as pd
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

# torch.manual_seed(1)    # reproducible

# Hyper Parameters
TIME_STEP = 76319  # rnn time step
INPUT_SIZE = 1  # rnn input size
LR = 0.02  # learning rate
device = torch.device("cuda")


def trans(use_cols):
    x = pd.read_excel(io='第4题-处理后的大数据集-遗传算法用.xlsx', usecols=[use_cols])
    x = np.array(x.stack())
    # x = x.tolist()
    return x


def getSrcData():
    """获取原始数据；"""
    # 初始化的输入数据；
    receive_distance = trans("厚度")
    # 初始化输出数据；
    ratio = trans("过滤效率")
    # 返回初始数据；
    return receive_distance, ratio


# show data
# steps = np.linspace(0, np.pi * 2, 100, dtype=np.float32)  # float32 for converting torch FloatTensor
# x_np = np.sin(steps)
# y_np = np.cos(steps)
# plt.plot(steps, y_np, 'r-', label='target (cos)')
# plt.plot(steps, x_np, 'b-', label='input (sin)')
# plt.legend(loc='best')
# plt.show()


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        self.rnn = nn.RNN(
            input_size=INPUT_SIZE,
            hidden_size=32,  # rnn hidden unit
            num_layers=1,  # number of rnn layer
            batch_first=True,  # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )
        self.out = nn.Linear(32, 1)

    def forward(self, x, h_state):
        # x (batch, time_step, input_size)
        # h_state (n_layers, batch, hidden_size)
        # r_out (batch, time_step, hidden_size)
        r_out, h_state = self.rnn(x, h_state)

        outs = []  # save all predictions
        for time_step in range(r_out.size(1)):  # calculate output for each time step
            outs.append(self.out(r_out[:, time_step, :]))
        return torch.stack(outs, dim=1), h_state

        # instead, for simplicity, you can replace above codes by follows
        # r_out = r_out.view(-1, 32)
        # outs = self.out(r_out)
        # outs = outs.view(-1, TIME_STEP, 1)
        # return outs, h_state

        # or even simpler, since nn.Linear can accept inputs of any dimension
        # and returns outputs with same dimension except for the last
        # outs = self.out(r_out)
        # return outs


rnn = RNN()
rnn = rnn.to(device)
print(rnn)

optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)  # optimize all cnn parameters
loss_func = nn.MSELoss()
loss_func = loss_func.to(device)

h_state = None  # for initial hidden state

plt.figure(1, figsize=(12, 5))
plt.ion()  # continuously plot

for step in range(100):
    start, end = step * np.pi, (step + 1) * np.pi  # time range
    # use sin predicts cos
    steps = np.linspace(start, end, TIME_STEP, dtype=np.float32,
                        endpoint=False)  # float32 for converting torch FloatTensor
    Receive_distance, Ratio = getSrcData()
    x_np = Receive_distance
    y_np = Ratio

    x = torch.from_numpy(x_np[np.newaxis, :, np.newaxis])  # shape (batch, time_step, input_size)
    y = torch.from_numpy(y_np[np.newaxis, :, np.newaxis])

    x = x.float()
    y = y.float()

    x = x.to(device)
    y = y.to(device)

    prediction, h_state = rnn(x, h_state)  # rnn output
    # !! next step is important !!
    h_state = h_state.data  # repack the hidden state, break the connection from last iteration

    loss = loss_func(prediction, y)  # calculate loss
    optimizer.zero_grad()  # clear gradients for this training step
    loss.backward()  # backpropagation, compute gradients
    optimizer.step()  # apply gradients

    # plotting
    plt.plot(steps, y_np.flatten(), 'r-')
    plt.plot(steps, prediction.data.cpu().numpy().flatten(), 'b-')
    plt.draw()
    plt.pause(0.05)

plt.ioff()
plt.show()
