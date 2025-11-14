#Apply Bagging on 50 Startups dataset (All 3 --- Shuffling , Sampling with replacement, Sampling without Replacement)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import BaggingRegressor
import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv('50_Startups.csv')

print(data.info())

features = data.iloc[:,[0,1,2,3]].values
encoder = OneHotEncoder(sparse=False)

encoder.fit_transform(features[:,[3]])

encoded = encoder.transform(features[:,[3]])
final_features = np.concatenate((encoded, features[:,[0,1,2]]), axis=1)
print(final_features)

label = data.iloc[:,[4]].values



X_train, X_test, y_train, y_test = train_test_split(final_features, label, test_size=0.2, random_state=3)

model = KNeighborsRegressor()

# 1. Bagging regressor: Shuffling
bagging_regressor_shuffling = BaggingRegressor(n_estimators=5,
                                              base_estimator= model,
                                              random_state=3)
bagging_regressor_shuffling.fit(X_train, y_train)

print('Bagging regressor Shuffling:')
print(bagging_regressor_shuffling.score(X_train, y_train))
print(bagging_regressor_shuffling.score(X_test, y_test))

# 2.1 Sampling with replacement
algo = KNeighborsRegressor()
sample = int(round(np.sqrt(len(X_train))))
bagging_regressor_sampling = BaggingRegressor(n_estimators=5, 
                                              base_estimator= algo,
                                              bootstrap=True,
                                              max_samples=sample,
                                              random_state=3)

bagging_regressor_sampling.fit(X_train, y_train)
print('Bagging regressor Sampling with replacement:')
print(bagging_regressor_sampling.score(X_train, y_train))
print(bagging_regressor_sampling.score(X_test, y_test))

# 2.2 Sampling without replacement
algo = KNeighborsRegressor()
sample = int(round(np.sqrt(len(X_train))))
bagging_regressor_sampling_wr = BaggingRegressor(n_estimators=5, 
                                              base_estimator= algo,
                                              bootstrap=False,
                                              max_samples=sample,
                                              random_state=3)

bagging_regressor_sampling_wr.fit(X_train, y_train)
print('Bagging regressor Sampling without replacement:')
print(bagging_regressor_sampling_wr.score(X_train, y_train))
print(bagging_regressor_sampling_wr.score(X_test, y_test))

'''Bagging regressor Shuffling:
0.7765618030484531
0.5718490983816903
Bagging regressor Sampling with replacement:
0.21001013785971845
0.09286945159917781
Bagging regressor Sampling without replacement:
0.14439853301959504
0.24945173060344406'''