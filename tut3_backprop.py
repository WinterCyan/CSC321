import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.special import expit as sigmoid
import math

np.random.seed(0)

def make_dataset(num_points):
    radius = 5
    data = []
    labels = []
    for i in range(num_points//2):
        r = np.random.uniform(0,radius*.5)
        angle = np.random.uniform(0, 2*math.pi)
        x=r*math.sin(angle)
        y=r*math.cos(angle)
        data.append([x,y])
        labels.append(1)
    for i in range(num_points//2):
        r = np.random.uniform(radius*.7, radius)
        angle = np.random.uniform(0, 2 * math.pi)
        x = r * math.sin(angle)
        y = r * math.cos(angle)
        data.append([x, y])
        labels.append(0)
    data = np.asarray(data)
    labels = np.asarray(labels)
    return data,labels


params = {'U': np.random.randn(3, 2), 'b': np.zeros(3), 'W': np.random.randn(3), 'c': 0}
def forward(X,params):
    N=X.shape[0]
    G=np.dot(X,params['U'].T)+params['b']
    H=np.tanh(G)
    z=np.dot(H,params['W'].T)+params['c']
    y=sigmoid(z)
    return y


def backprop(X,t,params):
    N=X.shape[0]
    G=np.dot(X,params['U'].T)+params['b']
    H=np.tanh(G)
    z=np.dot(H,params['W'].T)+params['c']
    y=sigmoid(z)
    loss=(1./N)*np.sum(-t*np.log(y)-(1-t)*np.log(1-y))
    z_bar = (1.0/N)*(y-t)
    W_bar = np.dot(H.T,z_bar)
    c_bar = np.dot(z_bar, np.ones(N))
    H_bar = np.outer(z_bar, params['W'].T)
    G_bar = H_bar*(1-np.tanh(G)**2)
    U_bar = np.dot(G_bar.T,X)
    b_bar = np.dot(G_bar.T,np.ones(N))
    grads = {'U': U_bar, 'b': b_bar, 'W': W_bar, 'c': c_bar}
    return grads,loss


num_datapoints = 1000
data,labels = make_dataset(num_datapoints)
plt.scatter(data[:num_datapoints//2, 0], data[:num_datapoints//2, 1], color='red')
plt.scatter(data[num_datapoints//2:, 0], data[num_datapoints//2:, 1], color='green')

num_steps = 1000
alpha = 1
for step in range(num_steps):
    grads,loss = backprop(data, labels, params)
    for k in params:
        params[k] -= alpha*grads[k]
    if step%50 == 0:
        print('step {:3d} | loss {:3.2f}'.format(step, loss))
num_points = 200
x1s = np.linspace(-6.0, 6.0, num_points)
x2s = np.linspace(-6.0, 6.0, num_points)

points = np.transpose([np.tile(x1s, len(x2s)), np.repeat(x2s, len(x1s))])
Y = forward(points, params).reshape(num_points, num_points)
X1, X2 = np.meshgrid(x1s, x2s)

plt.figure()
plt.pcolormesh(X1, X2, Y, cmap=plt.cm.get_cmap('YlGn'))
plt.colorbar()
plt.scatter(data[:num_datapoints//2, 0], data[:num_datapoints//2, 1], color='red')
plt.scatter(data[num_datapoints//2:, 0], data[num_datapoints//2:, 1], color='blue')
plt.show()