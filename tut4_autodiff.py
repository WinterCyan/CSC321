import autograd.numpy as np
from autograd import grad
import autograd.numpy.random as npr
from autograd.misc import flatten
from autograd.extend import primitive, defvjp
import matplotlib.pyplot as plt
from autograd.misc.optimizers import sgd

np.set_printoptions(suppress=True)

# def tanh(x):
#     y = np.exp(-x)
#     return (1.0-y)/(1.0+y)
#
#
# grad_tanh = grad(tanh)
# # print(grad_tanh(1.0))
# # print((tanh(1.0001)-tanh(0.9999))/0.0002)
#
# def grad_sigmoid_manual(x):
#     # forward
#     a=-x
#     b=np.exp(a)
#     c=1+b
#     s=1.0/c
#     # backward
#     dsdc=(-1.0/c**2)
#     dsdb=dsdc*1
#     dsda=dsdb*np.exp(a)
#     dsdx=dsda*(-1)
#
#     return dsdx
#
#
# def fun_sigmoid(x):
#     y=1.0/(1.0+np.exp(-x))
#     return y
#
#
# grad_sigmoid_auto = grad(fun_sigmoid)
# # print('manual: ',grad_sigmoid_manual(2.0))
# # print('auto: ',grad_sigmoid_auto(2.0))
#
# # param is a list with shape (3,)
# param=[[1.0,2.0,3.0],[4.0,5.0],[6.0,7.0,8.0,9.0]]
# print('param: ',type(param),np.shape(param),param)
# flat_param, unflatten_func=flatten(param)
# # convert to ndarray, with shape (9,)
# print('flattened param: {}'.format(flat_param),type(flat_param),np.shape(flat_param))
# # treat every number (1.0, 2.0, ...) as array: (array(1.), array(2.), ...), is a list with shape (3,)
# unflat_param = unflatten_func(flat_param)
# print('unflattened param: {}'.format(unflat_param),type(unflat_param),np.shape(unflat_param))
#
# print()
# # every matrix is a array, array([[...],[...]])
# param2=[npr.randn(3,3),npr.randn(4,4),npr.randn(3,3)]
# print('param2: ',param2,type(param2),np.shape(param2))
# flat_param2, unflatten_func2=flatten(param2)
# print('flattened param2: {}'.format(flat_param2),type(flat_param2),np.shape(flat_param2))
# unflat_param2 = unflatten_func2(flat_param2)
# print('unflattened param2: {}'.format(unflat_param2),type(unflat_param2),np.shape(unflat_param2))


# pa={'x':1.,'y':2.} # the params must be non-INT!
# def xyz(p):
#     return p['x']**2+3*p['y'] # x^2 + 3y
#
# grad_xyz=grad(xyz)
# print(grad_xyz(pa))

# def linear_regression_cost(params):
#     N = np.shape(params['X'])[0]
#     loss = (1.0/(2*N))*np.sum((np.dot(params['X'],params['w'])+params['b']*np.ones(N)-params['t'])**2)
#     return loss
#
#
# grad_linear_regression_cost = grad(linear_regression_cost)
# N = 100
# x = np.linspace(0,10,N)
# t = 4*x+10+npr.normal(0,2,x.shape[0])
#
# w=npr.normal(0,1)
# b=npr.normal(0,1)
# params={'X': x, 'w': w, 'b': b, 't': t}
# num_epoch=1000
# alpha=0.01
# for i in range(num_epoch):
#     cost=grad_linear_regression_cost(params)
#     params['w']-=alpha*cost['w']
#     params['b']-=alpha*cost['b']
# print(params['w'],params['b'])
#
# plt.plot(x,t,'r.')
# plt.plot([0,10],[params['b'],params['w']*10+params['b']],'b-') # draw a line which connects (0,b) and (10,10w+b)
# plt.show()

# n=100
# x=np.linspace(-3,3,n)
# t=x**4-10*x**2+10*x+npr.normal(0,3,x.shape[0])
# m=4
# # create 100 * 5 data matrix, data_num * maxorder+1, here (1, x, x^2, x^3, x^4)
# xfm=np.array([ [item**i for i in range(m+1)] for item in x] )
# w=npr.randn(xfm.shape[-1])
#
# def linear_with_fm_cost(params):
#     n=np.shape(params['x'])[0]
#     return (1.0/(2*n))*np.sum(( np.dot(params['x'],params['w'])-params['t'] )**2)
#
#
# grad_linear_with_fm_cost = grad(linear_with_fm_cost)
# params={'x':xfm, 'w': w, 't': t}
# num_epoch=10000
# alpha=0.001
# for i in range(num_epoch):
#     cost=grad_linear_with_fm_cost(params)
#     w-=alpha*cost['w']
# print(w)
#
# plt.plot(x,t,'r.')
# plt.plot(x,np.dot(xfm,w), 'b-')
# plt.show()

x=np.linspace(-5,5,1000)
t=x**3-20*x+10+npr.normal(0,4,x.shape[0])

inputs=x.reshape(x.shape[-1],1)
W1=npr.randn(1,4)
b1=npr.randn(4)
W2=npr.randn(4,4)
b2=npr.randn(4)
W3=npr.randn(4,1)
b3=npr.randn(1)
params={'W1': W1, 'W2': W2, 'W3': W3, 'b1': b1, 'b2': b2, 'b3': b3}

def relu(x):
    return np.maximum(0,x)


nonlinearity = np.tanh
def forward(params, inputs):
    h1=nonlinearity(np.dot(inputs,params['W1'])+params['b1'])
    h2=nonlinearity(np.dot(h1,params['W2'])+params['b2'])
    output=np.dot(h2,params['W3'])+params['b3']
    return output


def loss(params, i=0):
    output = forward(params,inputs)
    # convert col-shape output to row-shape output, align with t
    return (1.0/(2*inputs.shape[0]))*np.sum( (output.reshape(output.shape[0])-t)**2 )

print(loss(params))
optimized_params=sgd(grad(loss),params, step_size=0.01, num_iters=5000)
print(optimized_params)
print(loss(optimized_params))
final_y=forward(optimized_params, inputs)
plt.plot(x,t,'r.')
plt.plot(x, final_y, 'b-')
plt.show()
