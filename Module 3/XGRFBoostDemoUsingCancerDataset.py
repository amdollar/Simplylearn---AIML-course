import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBRFClassifier



data = pd.read_csv('cancer.csv')

features = data.iloc[:, 2:32].values
label = data.iloc[:,[1]].values

# encode the label col
encoder = LabelEncoder()
encoded_label = encoder.fit_transform(label)

CL = 0.8
for rs in range(1, 101):
    X_train, X_test, y_train, y_test = train_test_split(features, encoded_label, test_size= 0.2, random_state= rs)

    model = XGBRFClassifier()
    model.fit(X_train, y_train)

    test_score = model.score(X_test, y_test)
    train_score = model.score(X_train, y_train)

    if test_score > train_score and test_score >= CL:
        print('Generalized model.')





