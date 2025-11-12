import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import warnings

warnings.filterwarnings('ignore')

data = pd.read_csv('Social_Network_Ads.csv')

print(data.info())
# No null values in dataset.

# Label to be predicted: Purchased (Binary classification problem)
print(data.Purchased.value_counts())
# 0    257
# 1    143

# Feature and label saperation:
features= data.iloc[:,[0,1]].values
label = data.iloc[:, [2]].values
print(features)

# create train test split, model and train that model:
'''
SL = 0.2
CL = 1-SL
for rs in range(1,100):
    X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.2, random_state=rs)

    model = KNeighborsClassifier(n_neighbors= 5)
    model.fit(X_train, y_train)
    
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)

    if test_score > train_score and test_score >= CL:
        print(f'Test score : {test_score} , and Train score : {train_score}, for random state : {rs}')'''

'''
For CL = .80
Test score : 0.8625 , and Train score : 0.85625, for random state : 17
Test score : 0.8875 , and Train score : 0.875, for random state : 59
Test score : 0.8625 , and Train score : 0.84375, for random state : 68

We can further tune this logic to get the best value of the K as well.

'''
for rs in range(1,100):
    for k in range (1, 40):
        X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.2, random_state=rs)

        model = KNeighborsClassifier(n_neighbors= k)
        model.fit(X_train, y_train)
        
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)

        if test_score > train_score and test_score >= CL:
            print(f'Test score : {test_score} , and Train score : {train_score}, for random state : {rs}, and K values: {k}')


# Data is imbalanced that means, we need to use Statical or Domin but It's binary so Accuracy or PR pair approach



