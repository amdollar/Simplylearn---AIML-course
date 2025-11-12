import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import warnings
from sklearn.metrics import classification_report, confusion_matrix
warnings.filterwarnings('ignore')
'''
Create a model that can predict the customer is good or bad based on age and estimated salary.
'''

data = pd.read_csv('Social_Network_Ads.csv')

print(data.info())

# split the data in feature and lables:

features = data.iloc[:, [0,1]].values
label = data.iloc[:,[2]].values

# Decide the SL and CL values
CL = 0.8

# Create a train and test split
# Find the best random state 

# Random state is selected: 797 So now building and training the model for prediction:

X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.2, random_state=797)

model = LogisticRegression()

model.fit(X_train, y_train)

age_i = int(input('Enter the age of the customer: '))
salary_i = float(input('Enter the salary of the customer: '))

inputM = np.array([[age_i, salary_i]])
ans = model.predict(inputM)
print(ans)


''' 
Evauation metrics for this model:
Since this is a Classification problem: We can use Accuracy, Precision, Recall, F1 score to judege the model.
'''

print(confusion_matrix(label, model.predict(features)))
'''[[236  21]  S 257
 [ 40 103]] S 143 
Imbalanced dataset So we are going with the PR approach.

# 1. 0Bad customer ---- 1Good customer
# 2. 1Good customer ---- 0Bad customer
'''

print(classification_report(label, model.predict(features)))
# Domain Expert Selected 1Good customer ---- 0Bad customer

# P0 = 0.85
# R1 = 0.72
# Micro average = P0+R1 / 2 = 0.785 that is not greater than CL ---- Discard this model.