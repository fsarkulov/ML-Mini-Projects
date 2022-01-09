# -*- coding: utf-8 -*-
"""
Created on Sun Jan  9 15:37:39 2022

@author: farus
"""


from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#from dataprep.eda import plot
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import sklearn as sk
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

#Applying Logistic Regression to see which group of parameters accurately predicts diagnosis

model = LogisticRegression()

model.fit(xtrain, ytrain)
prediction = model.predict(xtest)

print(classification_report(ytest, prediction))

print(confusion_matrix(ytest,prediction))

#Logistic Regression tells us that 51 are True Negative (Benign) and 87 are True Positive (Malignant). The model gave us 5 incorrect classifications for True Positive. 

#Now we implement a Naive Bayes Classifier and see how it performs

from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()
classifier.fit(xtrain,ytrain)
naive_pred = classifier.predict(xtest)

conf_mat = confusion_matrix(ytest, naive_pred)
accu = accuracy_score(ytest,naive_pred)

#here through the naive bayes model we have an accuracy of 94%, which is good in general but terrible for cancer detection. 6% false positive or false negatives is a high mis-classification rate for cancer. 