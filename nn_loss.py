import torch
from torch import nn

ipts = torch.tensor([1, 2, 3], dtype=torch.float32)
targets = torch.tensor([1, 2, 5], dtype=torch.float32)

ipts = torch.reshape(ipts, (1, 1, 1, 3))
targets = torch.reshape(targets, (1, 1, 1, 3))

loss = nn.L1Loss(reduction="mean")
result = loss(ipts, targets)

loss_mse = nn.MSELoss()
result_mse = loss_mse(ipts, targets)

print(result)
print(result_mse)

x = torch.tensor([0.1, 0.2, 0.3])
y = torch.tensor([1])

x = torch.reshape(x, (1, 3))
result_cross = nn.CrossEntropyLoss(reduction='mean')

print(result_cross)
