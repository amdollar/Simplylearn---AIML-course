'''
# Default alpha values | SL -----> 0.05, 0.01, 0.1
# To generate/get the practical SL value we use corss validation.

# Cross validation is applied on entire dataset (Pre-modelling phase | EDA)
# Ideal goal of this phase:
# 1. Get the score benchmark
# 2. Get the approximate Optimal Score
# 3. Extract the best training sample that may provide the optimal score.
'''
import warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
warnings.filterwarnings('ignore')

data= pd.read_csv('https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/639388c2cbc2120a14dcf466e85730eb8be498bb/iris.csv')


features = data.iloc[:,[0,1,2,3]].values
label = data.iloc[:,[4]].values

# Encode string labels to numeric values
encoder = LabelEncoder()
label = encoder.fit_transform(label.ravel())

model = LinearRegression()
scores = cross_val_score(model, features, label, cv = 5)

print(scores)

# [0.         0.85215955 0.         0.76225759 0.        ]

# Calculating the benchmark score:
print(f'Minimum threshold score: {scores.min()}')
print(f'Mean threshold score: {scores.mean()}')
print(f'Suggested SL values: { 1- scores.mean()}')

'''Minimum threshold score: 0.0
Mean threshold score: 0.32288342759973726
Suggested SL values: 0.6771165724002628'''


# To extract the best training model:
CL = 0.93 # According to class outputs

# Step 1: Init the algo
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()

# Step 2: Init the K-fold cross validation

from sklearn.model_selection import KFold

kfold = KFold(n_splits=5,        #This is the cv value u declared in cross_val_score
              shuffle=True,
              random_state=1) #This random state is to help me reproduce the same output

iterationo = 0

for train_index, test_index in kfold.split(features):
    iterationo +=1
    
    X_train, X_test = features[train_index], features[test_index]
    y_train, y_test = label[train_index], label[test_index]

    model.fit(X_train, y_train)

    if model.score(X_test, y_test) > CL:
        print(f"TestScore {model.score(X_test,y_test)} and Train Score {model.score(X_train,y_train)} for iteration {iterationo}")

#  TestScore 0.9666666666666667 and Train Score 0.9833333333333333 for iteration 1
# TestScore 0.9666666666666667 and Train Score 0.9666666666666667 for iteration 2
# TestScore 0.9333333333333333 and Train Score 0.9833333333333333 for iteration 3
# TestScore 0.9333333333333333 and Train Score 0.975 for iteration 4
# TestScore 1.0 and Train Score 0.9666666666666667 for iteration 5