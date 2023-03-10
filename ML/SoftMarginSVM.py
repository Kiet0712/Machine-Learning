from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
plt.xkcd()
n_samples = 2000
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
clf = make_pipeline(StandardScaler(),LinearSVC(random_state=0, tol=1e-5,loss='hinge',fit_intercept=True,class_weight='balanced'))
clf.fit(X_train, Y_train)
y_predict = clf.predict(X_test)
print('SVM:',accuracy_score(y_true=Y_test,y_pred=y_predict)*100)
w = clf.named_steps['linearsvc'].coef_
b = clf.named_steps['linearsvc'].intercept_
print(w)
print(b)
xtemp = np.linspace(-2,8,100)
ytemp = -w[0][0]/w[0][1]*xtemp-b[0]/w[0][1]
model = Perceptron()
model.fit(X_train, Y_train)
y_predict1 = model.predict(X_test)
print('PLA:',accuracy_score(y_true=Y_test,y_pred=y_predict1)*100)
w1 = model.coef_
b1 = model.intercept_
xtemp1 = np.linspace(-2,8,100)
ytemp1 = -w1[0][0]/w1[0][1]*xtemp1-b1[0]/w1[0][1]
fig, ax = plt.subplots()
plt.title('SVM and PLA')
##plt.xkcd()
plt.xlabel('x')
plt.ylabel('y')
for n_class in range(2):
    ax.scatter(data[labels==n_class][:, 0], 
               data[labels==n_class][:, 1], 
               c=colours[n_class], 
               s=50, label=str(n_class))
plt.plot(xtemp,ytemp, label = "SVM")
plt.plot(xtemp,ytemp+1/w[0][1], label = "SVM1")
plt.plot(xtemp,ytemp-1/w[0][1], label = "SVM2")
plt.plot(xtemp1,ytemp1, label = "PLA")
plt.legend()
plt.show()
#c?? th??? th???y r???ng ???????ng th???ng t???o ra b???i SVM ??em l???i cho ch??ng ta hi???u su???t cao h??n
