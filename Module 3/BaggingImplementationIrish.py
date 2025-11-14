import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv("https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv")
print(data.info()) # No null values
print(data.head())

features = data.iloc[:,[0,1,2,3]].values
label = data.iloc[:, [4]].values
SL = 0.1
CL = 1 - SL

algo = KNeighborsClassifier()
X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.2, random_state=4)

#1 shuffling:
algo.fit(X_train, y_train)

shuffel_ensembler = BaggingClassifier(n_estimators=5, 
                                      base_estimator=algo,
                                      random_state=4)
shuffel_ensembler.fit(X_train, y_train)
print('Bagging with shuffling: ')
print(shuffel_ensembler.score(X_train, y_train))
print(shuffel_ensembler.score(X_test, y_test))

# Bagging with shuffling:
# 1.0
# 0.9666666666666667

# 2.1 sampling: with replacement
sample = int(round(np.sqrt(len(X_train))))
algo = KNeighborsClassifier()
sampling_ensembler = BaggingClassifier(n_estimators= 5, 
                                       base_estimator=algo, 
                                       bootstrap=True,
                                       max_samples=sample, 
                                       random_state= 3)

sampling_ensembler.fit(X_train, y_train)
print('Bagging with sampling -> replacement: ')
print(sampling_ensembler.score(X_train, y_train))
print(sampling_ensembler.score(X_test, y_test))

# 2.2 Sampling: without replacement.
algo = KNeighborsClassifier()
sampling_ensembler_r = BaggingClassifier(n_estimators= 5, 
                                       base_estimator=algo, 
                                       bootstrap=False,
                                       max_samples=sample, 
                                       random_state= 3)

sampling_ensembler_r.fit(X_train, y_train)
print('Bagging with out sampling -> replacement: ')
print(sampling_ensembler_r.score(X_train, y_train))
print(sampling_ensembler_r.score(X_test, y_test))


'''
Bagging with shuffling: 
0.9916666666666667
0.9333333333333333
Bagging with sampling -> replacement: 
0.8666666666666667
0.8
Bagging with out sampling -> replacement:
0.625
0.3333333333333333
'''