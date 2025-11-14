# Targets:
# Shuffling, Sampling with Replacement, Sampling without replacement
import numpy as np
import warnings
warnings.filterwarnings('ignore')
dataset = [1,2,3,4,5,6]
# Sampling with replacement:
replacement_dataset = np.random.choice(dataset, size=3, replace=True)
print(replacement_dataset)

# Sampling without replacement (All values will be unique)
unique_values_dataset = np.random.choice(dataset, size = 3, replace=False)
print(unique_values_dataset)
# [1 2 2]
# [3 6 4]

# Bagging Implementation:
'''
1. Read the dataset
2. Split into features and labels
3. Create the variables with test and training data
4. Choose the training algorithm
5. Write for Bagging with Shuffling
6. Write for Bagging with Sampling:
    i. with replacement
    ii. without replacement
7. Check the score of the ensemble model.

'''

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier

data = pd.read_csv('Social_Network_Ads.csv')
print(data.info())

features = data.iloc[:, [0,1]].values
label = data.iloc[:, [2]].values

X_train,X_test,y_train,y_test = train_test_split(features, label, test_size=0.2, random_state=1)
model = KNeighborsClassifier()

# This is the part where we will be doing Bagging:
# i. Decide the learners size:
# learners = int(round(np.sqrt(len(X_train))))


# Default: Shuffling

ensemble_model = BaggingClassifier(n_estimators=5, 
                                   base_estimator=model,
                                   random_state=2)

ensemble_model.fit(X_train, y_train)
#Train score:
print(ensemble_model.score(X_train, y_train)) # 0.909375
#Test score:
print(ensemble_model.score(X_test, y_test)) # 0.7625