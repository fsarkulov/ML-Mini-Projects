# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#from dataprep.eda import plot
from sklearn.preprocessing import StandardScaler
#preparing data converting to dataframe
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns = data.feature_names)

#recoding diagnosis into numeric
df['diagnosis'] = data.target
df['diagnosis'] = df['diagnosis'].map({1: 'Malignant', 0: 'Benign'})


#splitting data into testing and training sets

xtrain, xtest, ytrain, ytest = train_test_split(data.data, data.target, test_size = 0.25, random_state = 100)

#scaling inputs to normal
scaler = StandardScaler()
scaler.fit(xtrain)
xtrain = scaler.transform(xtrain)
xtest = scaler.transform(xtest)

#implementing KNN with k=5
k = 5
classifier = KNeighborsClassifier(n_neighbors=k)
classifier.fit(xtrain,ytrain)
ypred = classifier.predict(xtest)

#calculating error for different values of K

error = []
for i in range(1,40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(xtrain,ytrain)
    pred_i = knn.predict(xtest)
    error.append(np.mean(pred_i != ytest))

plt.figure(figsize=(12, 6))
plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')