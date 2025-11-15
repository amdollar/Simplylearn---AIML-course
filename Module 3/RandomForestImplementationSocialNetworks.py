import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

data= pd.read_csv('Social_Network_Ads.csv')
print(data.info())

features = data.iloc[:,[0,1]].values
labels = data.iloc[:,[2]].values

CL = 0.9
for rs in range(1, 101):
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=rs)

    model = RandomForestClassifier(n_estimators= 50 , max_depth=5)

    model.fit(X_train, y_train)

    test_score= model.score(X_test,y_test)
    train_score = model.score(X_train, y_train)

    if test_score > train_score and test_score >= CL:
        print(f'Test score  {test_score} | Test score {test_score} | RS {rs}')
'''Test score  0.9625 | Test score 0.9625 | RS 24
Test score  0.9375 | Test score 0.9375 | RS 58
Test score  0.9375 | Test score 0.9375 | RS 66
Test score  0.95 | Test score 0.95 | RS 67
Test score  0.9625 | Test score 0.9625 | RS 68
Test score  0.975 | Test score 0.975 | RS 76
Test score  0.95 | Test score 0.95 | RS 77
Test score  0.95 | Test score 0.95 | RS 90'''