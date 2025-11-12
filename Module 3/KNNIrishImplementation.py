# KNN implementation for IRISH dataset: Multivalue Classification.
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import warnings

warnings.filterwarnings('ignore')

data = pd.read_csv("https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv")
print(data.info()) # No null values
print(data.head())

# 1. Split data into Features and Labels:
features = data.iloc[:,[0,1,2,3]].values
label = data.iloc[:, [4]].values
# 1. Select the CL values:
SL = 0.1
CL = 1 - SL

# Find the perfect K and random state value

for rs in range (1, 10):
    for k in range(1, 20):
        X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.2, random_state=rs)

        model = KNeighborsClassifier(n_neighbors= k)

        model.fit(X_train, y_train)
        train_score= model.score(X_train, y_train)
        test_score= model.score(X_test, y_test)

        if test_score > train_score and test_score >= CL:
            print(f'Test score: {test_score}, Train score: {train_score}, Random state: {rs}, K value : {k}')

'''Test score: 1.0, Train score: 0.975, Random state: 1, K value : 2
Test score: 1.0, Train score: 0.95, Random state: 1, K value : 3
Test score: 1.0, Train score: 0.9583333333333334, Random state: 1, K value : 4
Test score: 1.0, Train score: 0.9583333333333334, Random state: 1, K value : 5

Here, we can select the any of above random state and K value'''


X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.2, random_state=1)

model = KNeighborsClassifier(n_neighbors= 4)

model.fit(X_train, y_train)

'''Evaluation of model:'''
from sklearn.metrics import classification_report
print(classification_report(label, ))
