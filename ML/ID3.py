import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
data = pd.read_csv('ID3.csv')
label_outlook = LabelEncoder()
label_temperature = LabelEncoder()
label_humidity = LabelEncoder()
label_wind = LabelEncoder()
label_play = LabelEncoder()
data['outlook'] = label_outlook.fit_transform(data['outlook'])
data['temperature'] = label_outlook.fit_transform(data['temperature'])
data['humidity'] = label_outlook.fit_transform(data['humidity'])
data['wind'] = label_outlook.fit_transform(data['wind'])
data['play'] = label_outlook.fit_transform(data['play'])
from sklearn import tree
model = tree.DecisionTreeClassifier()
X = data[['outlook','temperature','humidity','wind']]
y = data[['play']]
model.fit(X,y)
print(model.score(X,y))

