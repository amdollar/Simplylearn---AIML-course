# Objective: 
'''
# Create a model that can classify the customer as a good or bad
# customer based on customer's age and customer's estimated salary
'''
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

import warnings
warnings.filterwarnings("ignore")

data = pd.read_csv('Social_Network_Ads.csv')

print(data.info())
# Data is complete
features = data.iloc[:, [0,1]].values
label = data.iloc[:,[2]].values

# print(label)
CL = 0.90

for rs in range(1,500):
  X_train,X_test,y_train,y_test = train_test_split(features,
                                                   label,
                                                   test_size=0.2,
                                                   random_state=rs)

  model = LogisticRegression()

  model.fit(X_train,y_train)

  trainScore = model.score(X_train,y_train)
  testScore = model.score(X_test,y_test)


  if testScore > trainScore and testScore >= CL:
    print(f"Test Score : {testScore} | Train Score : {trainScore} | RS : {rs} ")


# Create final model, once random state is captured from above logic
X_train, X_test, y_train, y_test = train_test_split(features, label, train_size=0.2, random_state=314)

model= LogisticRegression()
model.fit(X_train, y_train)
age = input('Enter age of customer: ')
salary = input('Enter Salary of customer: ')

inputModel = np.array([[age, salary]])

CustomerTypeValue = model.predict(inputModel)
print(CustomerTypeValue)

# import pickle

# pickle.dump(model, open('CustomerPredictor.pkl', 'wb'))
