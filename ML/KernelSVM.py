from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
plt.xkcd()
n_samples = 1000
data, labels = make_blobs(n_samples=n_samples, centers=([4.5, 4], [4.5, 6.9]), cluster_std=1.3)
colours = ('green', 'orange')
fig, ax = plt.subplots()
# for n_class in range(2):
#     ax.scatter(data[labels==n_class][:, 0], 
#                data[labels==n_class][:, 1], 
#                c=colours[n_class], 
#                s=50, label=str(n_class))
#plt.show()
X_train, X_test, Y_train, Y_test = train_test_split(data,labels,test_size=0.2)
model = SVC(kernel='linear')
model.fit(X_train,Y_train)
y_predict = model.predict(X_test)
print(accuracy_score(Y_test,y_predict)*100)
x_temp = np.array([1,2,3,4,5])
Y_sigmoid = np.array([47.5,44,47,49,50.5])
Y_poly = np.array([86.5, 87, 88.5, 87.5, 78.5])
Y_linear = np.array([88, 86.5, 89.5, 86, 83.5])
Y_rbf = np.array([89.5, 85, 87, 86.5, 89.5])
plt.plot(x_temp,Y_sigmoid,label ='sigmoid')
plt.plot(x_temp,Y_poly,label ='poly')
plt.plot(x_temp,Y_linear,label ='linear')
plt.plot(x_temp,Y_rbf,label ='rbf')
plt.xlabel('Time try')
plt.ylabel('Accurate(%)')
plt.title('KernelSVM')
plt.legend()
plt.show()
#linear 88 86.5 89.5 86 83.5
#poly 86.5 87 88.5 87.5 78.5(very very slow)
#rbf gamma=5 89.5 85 87 86.5 89.5
#sigmoid just 47.5 44 47 49 50.5
