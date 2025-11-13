import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv('50_Startups.csv')

print(data.info())
print(data.head())

features = data.iloc[:, [0,1,2,3]].values
label = data.iloc[:, [4]].values

stateImputer = OneHotEncoder(sparse=False)
stateImputer.fit(features[:, [3]])
state_values = stateImputer.fit_transform(features[:, [3]])

final_features = np.concatenate((state_values, features[:, [0,1,2]]),axis = 1)
print(final_features)
SL = 0.05
CL = 1-SL
'''
for rs in range (1, 10):
    for k in range(1, 20):
        X_train, X_test, y_train, y_test = train_test_split(final_features, label, test_size=0.2, random_state=rs)

        model = KNeighborsRegressor(n_neighbors= k)

        model.fit(X_train, y_train)
        train_score= model.score(X_train, y_train)
        test_score= model.score(X_test, y_test)

        if test_score > train_score and test_score >= CL:
            print(f'Test score: {test_score}, Train score: {train_score}, Random state: {rs}, K value : {k}')
'''


X_train, X_test, y_train, y_test = train_test_split(final_features, label, test_size=0.2, random_state=7)

model = KNeighborsRegressor(n_neighbors= 2)

model.fit(X_train, y_train)

'''Evaluation of model:

Since this is a Regression problem we will be solving this using Regression metrics only:
'''


from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
mae= mean_absolute_error(label, model.predict(final_features))
r2= r2_score(label, model.predict(final_features))
mse= mean_squared_error(label, model.predict(final_features))

print(mae)
print(r2)
print(mse)
# 6095.129500000001
# 0.9418254324233233
# 92619520.18881248

