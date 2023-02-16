import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
n_samples = 500
data, labels = make_blobs(n_samples=n_samples, centers=([1.1, 3], [4.5, 6.9]), cluster_std=1.3)
colours = ('green', 'orange')
fig, ax = plt.subplots()
for n_class in range(2):
    ax.scatter(data[labels==n_class][:, 0], 
               data[labels==n_class][:, 1], 
               c=colours[n_class], 
               s=50, label=str(n_class))
plt.show()
X_train, X_test, Y_train, Y_test = train_test_split(data,labels,test_size=0.2)
new_model = LogisticRegression()
new_model.fit(X_train,Y_train)
y_predict = new_model.predict(X_test)
print(accuracy_score(y_true=Y_test,y_pred=y_predict))
new_percep = Perceptron()
new_percep.fit(X_train,Y_train)
y_predict1 = new_percep.predict(X_test)
print(accuracy_score(y_pred=y_predict1,y_true=Y_test))
#so sánh giữa logistic regression và PLA
#chạy nhiều lần thì ta thấy logistic regression đem lại hiệu suất cao hơn