import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import scatter_matrix
from sklearn.datasets import load_iris

def plot_sep(w1, w2, color='green'):
    plt.scatter(sepal_len, sepal_wid, c=labels, cmap=plt.cm.Paired)
    plt.title('Separation in input space')
    plt.xlim([-1.5,2])
    plt.ylim([-1.5,1.5])
    plt.xlabel('sepal_len')
    plt.ylabel('sepal_wid')
    if w2!=0:
        k = -w1/w2
        t=1 if w2>0 else -1
        plt.plot([-1.5,2.0],[-1.5*k,2.0*k],'y',color=color)
        plt.fill_between([-1.5,2.0],[-1.5*k,2.0*k],[1.5*t,1.5*t],alpha=0.2,color=color)
    if w2==0:
        t=1 if w1>0 else -1
        plt.plot([0.0,0.0],[-1.5,2.0],'-y',color=color)
        plt.fill_between([0,2.0*t],[-1.5,-2.0],[1.5,2],alpha=0.2,color=color)


def plot_weight_space(len, wid, lab=1, color='steelblue', lim=2.0):
    plt.title('Inputs in weight space')
    plt.xlim([-lim,lim])
    plt.ylim([-lim,lim])
    plt.xlabel('w1')
    plt.ylabel('w2')
    if wid!=0:
        k=-len/wid
        t=1*lab if wid>0 else -1*lab
        plt.plot([-lim,lim],[-lim*k,lim*k],'-y',color=color)
        plt.fill_between([-lim,lim],[-lim*k,lim*k],[t*lim,t*lim],alpha=.2,color=color)
    if wid==0:
        t=1*lab if len>0 else -1*lab
        plt.plot([0,0],[-lim,lim],'-y',color=color)
        plt.fill_between([0,2.0*t],[-lim,-lim],[lim,lim],alpha=.2,color=color)


iris = load_iris()
print(iris['DESCR'])
iris_data = pd.DataFrame(data=iris['data'],columns=iris['feature_names'])
iris_data['target'] = iris['target']
#color_wheel = {1:"#0392cf", 2:"#7bc043", 3:"#ee4035"}
#colors = iris_data['target'].map(lambda x: color_wheel.get(x+1))
#ax = scatter_matrix(iris_data, color=colors, alpha=0.6, figsize=(15,15), diagonal='hist')
sepal_len = iris['data'][:100, 0]
sepal_wid = iris['data'][:100, 1]
labels = iris['target'][:100]
sepal_len -= np.mean(sepal_len)
sepal_wid -= np.mean(sepal_wid)
#plt.figure()
#plt.scatter(sepal_len, sepal_wid, c=labels, cmap=plt.cm.Paired)
#plt.xlabel('SEPAL_LEN')
#plt.ylabel('SEPAL_WID')

a1 = sepal_len[41]
a2 = sepal_wid[41]
b1 = sepal_len[84]
b2 = sepal_wid[84]
w1=-.5
w2=1.0

plt.figure()
plot_weight_space(a1,a2,color='blue')
plot_weight_space(b1,b2,color='red',lab=-1)
plt.plot(w1,w2,'ob')
plt.plot(-1,1,'ok')

plt.figure()
plot_sep(-1, 1)

plt.show()
