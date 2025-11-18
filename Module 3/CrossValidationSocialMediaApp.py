#Apply Cross validation technique of Social Network Ads

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv('Social_Network_Ads.csv')

print(data.info())

features= data.iloc[:,[0,1]].values
labels= data.iloc[:,[2]].values

model = KNeighborsClassifier()
score = cross_val_score(model, features, labels, cv = 5)
print(score)
# [0.8125 0.8625 0.725  0.7625 0.675 ]
print(f'Mean value of the scores: {score.mean()}')
print(f'Suggested value for SL: {1 - score.mean()}')
print(f"Suggested SL : {1 - score.min()}")
'''
Suggested value for CL: 0.23250000000000015
Suggested SL : 0.32499999999999996'''

CL = 1- 0.23

# Initialize the Kfold
kfold = KFold(n_splits= 5, shuffle=True, random_state= 1)

iteration = 5

for trainIndex, testIndex in kfold.split(features):
    iteration +=1

    X_train, X_test = features[trainIndex], features[testIndex]
    y_train, y_test = labels[trainIndex], labels[testIndex]

    model.fit(X_train,y_train)

    if model.score(X_test, y_test) >= CL:
        print(f"TestScore {model.score(X_test,y_test)} and Train Score {model.score(X_train,y_train)} for iteration {iteration}")

'''
TestScore 0.8 and Train Score 0.853125 for iteration 7
TestScore 0.8625 and Train Score 0.853125 for iteration 8
TestScore 0.7875 and Train Score 0.86875 for iteration 9
TestScore 0.8 and Train Score 0.86875 for iteration 10'''
