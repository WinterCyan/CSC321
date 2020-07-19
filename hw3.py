import autograd.numpy as np
import numpy
from autograd import grad
import autograd.numpy.random as npr
import matplotlib.pyplot as plt
from autograd.misc.optimizers import sgd

# generate training data
N=10000
D=4
HD=3
X=npr.randn(N,D)
t=numpy.zeros([N,1])
for i in range(numpy.shape(X)[0]):
    temp_abs_sum = numpy.abs(X[i][0])+numpy.abs(X[i][1])+numpy.abs(X[i][2])+numpy.abs(X[i][3])
    for j in range(numpy.shape(X)[1]):
        X[i][j]=X[i][j]/temp_abs_sum
    if X[i][0] < X[i][1] < X[i][2] < X[i][3]:
        t[i]=1

b1=numpy.zeros([HD,1])
b2=0

# def forward(params,X):
    