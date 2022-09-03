import torch
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

data = [[1, 2], [3, 4]]
x_tensor = torch.tensor(data)

np_arr = np.array(data)
x_from_np = torch.from_numpy(np_arr) # numpy --> tensor
x_from_tensor = x_tensor.numpy() # tensor --> numpy

print(f"Shape of tensor: {x_tensor.shape}")
print(f"Datatype of tensor: {x_tensor.dtype}")
print(f"Device tensor is stored on: {x_tensor.device}")

# We move our tensor to the GPU if available
if torch.cuda.is_available():
  x_tensor = x_tensor.to('cuda')
print(f"Device tensor is stored on: {x_tensor.device}")

x = torch.tensor(data).to(device)
print(f"Device tensor is stored on: {x_tensor.device}")
