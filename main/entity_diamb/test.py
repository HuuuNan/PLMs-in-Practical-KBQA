import torch
from torch import nn

m = nn.Linear(20, 1)
input = torch.randn(128, 20)
output = m(input)
print(output.size())