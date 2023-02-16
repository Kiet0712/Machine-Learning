import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd

plt.xkcd()
data = pd.read_csv('KmeanClustering.csv')
X = data[['Age','Income']]
plt.xlabel('Age')
plt.ylabel('Income')
X.plot.scatter(x = 'Age', y = 'Income', c = 'b')
plt.show()
km = KMeans(3,init='random')
y_km = np.array(km.fit_predict(X))
X = np.array(X)
for i in range(X.shape[0]):
    if y_km[i]==0:
        plt.scatter(X[i][0],X[i][1],c='b',marker='*')
    elif y_km[i]==1:
        plt.scatter(X[i][0],X[i][1],c='r',marker='^')
    else:
        plt.scatter(X[i][0],X[i][1],c='g',marker='o')
plt.scatter(
    km.cluster_centers_[:, 0], km.cluster_centers_[:, 1],
    marker='*',
    c='red', edgecolor='black',
    label='centroids'
)
plt.xlabel('Age')
plt.ylabel('Income')
plt.title('Income of some people')
plt.show()