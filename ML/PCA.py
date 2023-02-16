import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
data = load_breast_cancer()
cancer = pd.DataFrame(data=data['data'],columns=data['feature_names'])
scale = StandardScaler()
scale.fit(cancer)
scale_cancer_data = scale.transform(cancer)
PCA_model = PCA(n_components=2)
PCA_model.fit(scale_cancer_data)
PCA_cancer_data = PCA_model.transform(scale_cancer_data)
clf = SVC(kernel='sigmoid', gamma=4, coef0=0)
label = pd.DataFrame(data=data['target'])
X_train, X_test, Y_train, Y_test = train_test_split(PCA_cancer_data,label, test_size=0.2, random_state=0)

clf.fit(X_train, Y_train)
Y_predict = clf.predict(X_test)
print(accuracy_score(Y_test,Y_predict)*100)
plt.xlabel('Feature 1 new')
plt.ylabel('Feature 2 new')
plt.title('PCA and binary classifier')
plt.xkcd()
plt.scatter(PCA_cancer_data[:,0],PCA_cancer_data[:,1],c= data['target'])
plt.show()
