import torch
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt
from scipy.io import loadmat

x = torch.tensor([3,1,4],dtype=torch.float,requires_grad=True)
print(x)
y = 3*x*x + 2*x + 5
print(y)
v = torch.tensor([0,.2,0],dtype=torch.float)
# vector-Jacobian product
y.backward(v) # set v as a param, assume l = g(y) returns a scalar, and dl/dy = v
print(x.grad) # get dl/dx
