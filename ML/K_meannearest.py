import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import neighbors, datasets
from sklearn.model_selection import train_test_split
data = pd.read_csv('weather.csv')
from sklearn.metrics import accuracy_score
dataset = data[['Humidity3pm','Pressure3pm','RainTomorrow']]

dataset_clean = dataset.dropna()
x = dataset_clean[['Humidity3pm','Pressure3pm']]
y = dataset_clean[['RainTomorrow']]
X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=100)

print("Training size: %d"%len(y_train))
print("Test size: %d"%len(y_test))
clf = neighbors.KNeighborsClassifier(n_neighbors = 10, p = 2, weights='distance')
clf.fit(X_train,y_train)
y_predict = clf.predict(X_test)
print('%.2f%%' %((accuracy_score(y_pred=y_predict,y_true=y_test)*100 )))

X_map = np.random.rand(10000,2)
X_map = X_map*(100,50)+(0,990)
y_map = clf.predict(X_map)
fig, ax = plt.subplots()
ax.scatter(x = X_map[:,0], y = X_map[:,1])
plt.show()