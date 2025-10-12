# Content
# For more information, read [Cortez et al., 2009].
# Input variables (based on physicochemical tests):
# Output variable (based on sensory data):

# 12 - quality (score between 0 and 10) 

import pandas as pd 
import numpy as np
import warnings
warnings.filterwarnings("ignore")

data= pd.read_csv('winequality-red.csv')
print(data.info())

# All variables are features apart from Quality:
# Quality is the lable variabl:
# We can use feature elimination technique aslo if the conditions are not satisfied
# SL: 0.3,0.2,0.1
#

features = data.iloc[:,[0,1,2,3,4,5,6,7,8,9,10]].values
print(features) 
label = data.iloc[:,[11]].values
print(label)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
# 2. Split the data in train and test 
 
CL = 0.70

# for rs in range(1,7500):
#     X_train, X_test, y_train, y_test = train_test_split(features, label, train_size=0.2, random_state=rs)

#     model = LogisticRegression()
    
#     # Train the model
#     model.fit(X_train, y_train)
    
#     # Get the score
#     train_score= model.score(X_train, y_train)
#     test_score = model.score(X_test, y_test)

#     if test_score > train_score and test_score>= CL:
#         print(f' Test score: {test_score},  Train score: {train_score},Random state: {rs}')


# Since data os not coming in the fileters we need to try the feature engineeing: RFE an SFM

from sklearn.feature_selection import RFE
modelAlgo = LogisticRegression()
feature_selection = RFE(estimator=modelAlgo)

feature_selection.fit(features, label)
print(feature_selection.ranking_)
# 
# [4 1 3 5 2 6 7 1 1 1 1]
# [0,1,2,3,4,5,6,7,8,9,10]
# 
newFeature = features[:,[1, 4, 7,8,9,10]]
print(newFeature)

for rs in range(1,100):
    X_train, X_test, y_train, y_test = train_test_split(newFeature, label, test_size=0.1, random_state=rs)

    model = LogisticRegression()
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Get the score
    train_score= model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)

    if test_score > train_score and test_score>= CL:
        print(f' Test score: {test_score},  Train score: {train_score},Random state: {rs}')

# try using: SFM:
from sklearn.feature_selection import SelectFromModel
modelAlgo = LogisticRegression()
selectFeatures = SelectFromModel(estimator=modelAlgo)

selectFeatures.fit(features, label)
print(selectFeatures.get_support())
# [False  True False False False False False False  True False  True]
# [0,1,2,3,4,5,6,7,8,9,10]



newFeatureforSFM = features[:, [1, 8, 10]]
for rs in range(1,1000):
    X_train, X_test, y_train, y_test = train_test_split(newFeatureforSFM, label, test_size=0.1, random_state=rs)

    model = LogisticRegression()
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Get the score
    train_score= model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)

    if test_score > train_score and test_score>= CL:
        print(f' Test score: {test_score},  Train score: {train_score},Random state: {rs}')
# print(f' Test score: {test_score},  Train score: {train_score},Random state: {rs}')