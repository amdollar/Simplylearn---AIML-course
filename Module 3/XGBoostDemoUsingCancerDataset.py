# This program is to show the process of Boosting: Here we will be using XGBoost and XGRFBoost algorithm using Cancer dataset. 
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
from xgboost import XGBClassifier


data = pd.read_csv('cancer.csv')

print(data.info())
print(data.head())

# diagnosis is the label column, other all are the feature columns.
# XGBoost algorithm expects the label column to be in Numerical (Discreate Numerical)

features = data.iloc[:, 2:32].values
label = data.iloc[:,[1]].values

encoder = LabelEncoder()
encoded_label = encoder.fit_transform(label)

CL = 0.8

for rs in range(1,101):
    X_train, X_test, y_train, y_test = train_test_split(features , encoded_label, test_size=0.2, random_state= rs)
    
    model = XGBClassifier()

    model.fit(X_train, y_train)

    test_score  = model.score(X_test, y_test)
    train_score = model.score(X_train, y_train)

    # if test_score > train_score and test_score >= CL:
        # print('Generalized Model!!')
    print(f'Test score: {test_score}, Train score: {train_score}, for random state: {rs}')
    # Test score: 1.0, Train score: 1.0, for random state: 44


    