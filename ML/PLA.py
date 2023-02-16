import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import Perceptron
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
n_samples = 500
data, labels = make_blobs(n_samples=n_samples, centers=([1.1, 3], [4.5, 6.9]), cluster_std=1.3,random_state=0)
colours = ('green', 'orange')
fig, ax = plt.subplots()
for n_class in range(2):
    ax.scatter(data[labels==n_class][:, 0], 
               data[labels==n_class][:, 1], 
               c=colours[n_class], 
               s=50, label=str(n_class))
plt.show()
X_train, X_test, Y_train, Y_test = train_test_split(data,labels,test_size=0.2)
pct = Perceptron(early_stopping=True)
pct.fit(data,labels)
w = pct.coef_
b = pct.intercept_
X_temp = np.linspace(-2,9,100)
y = -w[0][0]/w[0][1]*X_temp-b/w[0][1]
fig, ax = plt.subplots()
for n_class in range(2):
    ax.scatter(data[labels==n_class][:, 0], 
               data[labels==n_class][:, 1], 
               c=colours[n_class], 
               s=50, label=str(n_class))
plt.plot(X_temp,y)
plt.xlabel('x')
plt.ylabel('y')
plt.show()