#khi mà các nhóm dữ liệu trở nên phức tạp dần, việc sử dụng trực tiếp soft_max function là không đủ, chính vì vậy chúng ta cần sử dụng neural network hay mlp 
from sklearn.neural_network import MLPClassifier
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import plot_confusion_matrix

#sử dụng tiếp dataset make_blobs
plt.xkcd()
data,labels = make_blobs(n_samples=500,n_features=2,centers=15,random_state=0)
colours = ('green', 'orange','blue','red','black','yellow','purple','pink','gray','olive','coral','seagreen','lightskyblue','gold','forestgreen')

fig, ax = plt.subplots()
for n_class in range(15):
    ax.scatter(data[labels==n_class][:, 0], 
               data[labels==n_class][:, 1], 
               c=colours[n_class], 
               s=50, label=str(n_class))
plt.title('Multi Layer Peceptron')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
X_train, X_test, Y_train, Y_test = train_test_split(data,labels,test_size=0.2,random_state=0)
new_model = MLPClassifier(learning_rate_init=1,solver='lbfgs',max_iter=3000,hidden_layer_sizes=(10,10,10,10,10,10,10), random_state=0,early_stopping=True)
new_model.fit(X_train,Y_train)
y_predict = new_model.predict(X_test)
print(accuracy_score(Y_test,y_predict)*100)