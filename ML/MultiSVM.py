import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
#sử dụng lại data đã làm ở MLP
plt.xkcd()
data,labels = make_blobs(n_samples=500,n_features=2,centers=15,random_state=0)
colours = ('green', 'orange','blue','red','black','yellow','purple','pink','gray','olive','coral','seagreen','lightskyblue','gold','forestgreen')
X_train, X_test, Y_train, Y_test = train_test_split(data,labels,test_size=0.2,random_state=0)
model = SVC(kernel='linear')
model.fit(X_train,Y_train)
Y_predict = model.predict(X_test)
print(accuracy_score(Y_test,Y_predict)*100)
new_model = MLPClassifier(learning_rate_init=1,solver='lbfgs',max_iter=3000,hidden_layer_sizes=(15,15,15,15,15), random_state=0,early_stopping=True)
new_model.fit(X_train,Y_train)
y_predict1 = new_model.predict(X_test)
print(accuracy_score(Y_test,y_predict1)*100)
w = np.array(model.coef_)
xtemp = np.array(np.linspace(-13,13,100))
k = w.shape[0]
b = np.array(model.intercept_)
fig, ax = plt.subplots()
for n_class in range(15):
    ax.scatter(data[labels==n_class][:, 0], 
               data[labels==n_class][:, 1], 
               c=colours[n_class], 
               s=50, label=str(n_class))
plt.title('Multi SVM')
plt.xlabel('x')
plt.ylabel('y')
for i in range(k):
     ytemp = -w[i][0]/w[i][1]*xtemp-b[i]/w[i][1]
     plt.plot(xtemp,ytemp)
plt.ylim(-12,12)
plt.show()
#tốc độ tính toán của SVC là nhanh hơn và đem lại hiệu xuất cao hơn
#đối với MLP ta phải chọn số lượng hiddenlayer và hiddennode sao cho phù hợp với bộ dataset
