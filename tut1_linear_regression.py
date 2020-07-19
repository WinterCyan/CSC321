import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston

def cost(w1,w2,b,X,t):
    N = len(t)
    w = np.array([w1,w2])
    y = np.dot(X,w)+b*np.ones(N)
    return np.sum((y-t)**2)/(2.0*N)


def gradfn(w,X,t):
    N,D = np.shape(X)
    err = np.matmul(X,w) - t
    return np.matmul(np.transpose(X),err)/float(N)


def solve_with_gradfn(X,t,print_freq=5000,iter=100000,alpha=0.005):
    N,D = np.shape(X) # D=3
    w = np.zeros(D)
    for i in range(iter):
        dw = gradfn(w,X,t)
        w = w-alpha*dw
        if i%print_freq == 0:
            print('w after %d iteration: %s' % (i,str(w[0:D-1])))
            print('b after %d iteration: %s' % (i, str(w[D-1])))
            print('-------')
    return w


boston_data = load_boston()
print(boston_data['DESCR'])

data = boston_data['data']
x_input = data[:, [2,5]] # INDUS and RM
y_target = boston_data['target']

plt.figure(1)
plt.title('INDUS v.s. PRICE')
plt.scatter(x_input[:, 0], y_target)
plt.xlabel('INDUS')
plt.ylabel('PRICE')

plt.figure(2)
plt.title('RM v.s. PRICE')
plt.scatter(x_input[:, 1], y_target)
plt.xlabel('RM')
plt.ylabel('PRICE')

w1s = np.arange(-1.0,0.0,0.01)
w2s = np.arange(6.0,10.0,0.1)
z_cost = []
for w2 in w2s:
    z_cost.append([cost(w1, w2, -22.898, x_input, y_target) for w1 in w1s])
z_cost = np.array(z_cost)
np.shape(z_cost)
W1,W2 = np.meshgrid(w1s,w2s) # W1: x-coordinates of points in mesh, W2: y-coordinates of points in mesh
plt.figure(3)
CS = plt.contour(W1, W2, z_cost, 25)
plt.clabel(CS, inline=1, fontsize=10)
plt.title('Costs - w')
plt.xlabel('w1')
plt.xlabel('w2')
plt.plot([-0.3347],[7.8220],'o')
plt.show()

x_in = np.concatenate([x_input, np.ones([np.shape(x_input)[0], 1])], axis=1)
w_final = solve_with_gradfn(X=x_in, t=y_target)
print(w_final)
