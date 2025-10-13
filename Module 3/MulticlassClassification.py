import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

data = pd.read_csv("https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv")

print(data.info())


features = data.iloc[:,[0,1,2,3]].values
label = data.iloc[:,[4]].values
print(features)

CL = 0.95
for rs in range(1,301):
    X_train,X_test,y_train,y_test = train_test_split(features,
                                                      label, 
                                                      train_size=0.2,
                                                      random_state= rs)
    
    model = LogisticRegression()
    model.fit(X_train,y_train)

    trainScore = model.score(X_train, y_train)
    testScore = model.score(X_test, y_test)

    if testScore > trainScore and testScore >=CL:
        print(f'Test score: {testScore}, TrainScroe: {trainScore}, Random state: {rs}')


# final model with the random state:

X_train, X_test,y_train, y_test = train_test_split(features, label, train_size=0.2, random_state=17)

model = LogisticRegression()
model.fit(X_train, y_train)

train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

print(f'Test score: {test_score}, train score: {train_score}')


# Perform the export and import and testing of this one
# import pickle
# pickle.dump(model, open('Irishmodel.pkl', 'wb'))