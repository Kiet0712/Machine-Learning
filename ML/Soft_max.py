import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
#sử dụng tiếp dataset make_blobs đã được phân cụm thành 3 cụm
plt.xkcd()
data,labels = make_blobs(n_samples=500,n_features=2,centers=3,random_state=0)
colours = ('green', 'orange','blue')
fig, ax = plt.subplots()
for n_class in range(3):
    ax.scatter(data[labels==n_class][:, 0], 
               data[labels==n_class][:, 1], 
               c=colours[n_class], 
               s=50, label=str(n_class))
plt.show()
X_train, X_test, Y_train, Y_test = train_test_split(data,labels, test_size=0.2)
new_model = LogisticRegression(solver='lbfgs',multi_class='multinomial',C=1e5)
new_model.fit(X_train,Y_train)
y_predict = new_model.predict(X_test)
print(accuracy_score(Y_test,y_predict))
# vì độ phân tách của 3 cluster tạo ra là lớn nên độ chính xác chúng ta nhận được ở đây là rất cao, tuy nhiên với những dữ liệu phức tạp hơn chúng ta phải xử dụng thêm một số phương thức khác như multi layer peceptron
w = new_model.coef_
b = new_model.intercept_
X1 = np.linspace(-7,7,100)
X2 = X1.copy()
X3 = X1.copy()
y1 = -w[0][0]/w[0][1]*X1-b[0]/w[0][1]
y2 = -w[1][0]/w[1][1]*X1-b[1]/w[1][1]
y3 = -w[2][0]/w[2][1]*X1-b[2]/w[2][1]
fig, ax = plt.subplots()
plt.xkcd()
for n_class in range(3):
    ax.scatter(data[labels==n_class][:, 0], 
               data[labels==n_class][:, 1], 
               c=colours[n_class], 
               s=50, label=str(n_class))
plt.axis([-6, 6, -2, 10])
plt.plot(X1,y1)
plt.plot(X2,y2)
plt.plot(X3,y3)
plt.xlabel('x')
plt.ylabel('y')
plt.show()