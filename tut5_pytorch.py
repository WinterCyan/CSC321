import torch
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt
from scipy.io import loadmat


def f(x):
    return x**3 + 2*x


x = Variable(torch.from_numpy(np.array([4.0])), requires_grad = True)
y = f(x)
y.backward()
g = x.grad.data
print(g)