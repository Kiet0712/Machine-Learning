import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
data = pd.read_csv('weather.csv')
X = data[['WindSpeed3pm','Pressure3pm']]
y = data[['RainTomorrow']]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=100, random_state=None)
clf = KNeighborsClassifier(n_neighbors=10,weights='distance')
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)

print(accuracy_score(y_pred=y_pred,y_true=y_test))