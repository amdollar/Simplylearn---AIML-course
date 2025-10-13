import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# 1. create a model that can predict the price of hourse based on Car parking availability:
# Feature: Car parking
# Label: House prise
data = pd.read_csv('melb_data.csv')

# 1. Data pre processing: if data is NAN 
print('Car:')
print(data['Car'].isna().count())
print('BuildingArea:')
print(data['BuildingArea'].isna().count())
print('Landsize:')
print(data['Landsize'].isna().count())
print('YearBuilt:')
print(data['YearBuilt'].isna().count())
# No nan values

data['Car'].fillna(data['Car'].median(), inplace = True)
data['BuildingArea'].fillna(data['BuildingArea'].mean(), inplace = True)
data['Landsize'].fillna(data['Landsize'].mean(), inplace = True)
data['YearBuilt'].fillna(data['YearBuilt'].median(), inplace = True)


# 2. Split into train and test data:
features = data.iloc[:, [12,13,14,15]].values
print(features)

label = data.iloc[:, [4]].values
print(label)
print(data.info())

# 3. split the train and test data:
CL = 0.50
for rs in range(1,101):
  X_train,X_test,y_train,y_test = train_test_split(features,
                                                   label,
                                                   test_size=0.2,
                                                   random_state=rs
                                                   )

  model = LinearRegression()

  model.fit(X_train,y_train)

  trainScore = model.score(X_train,y_train)
  testScore = model.score(X_test,y_test)

#   if testScore > trainScore and testScore >= CL :
print(f"Test Score : {testScore} TrainScore : {trainScore} for RandomState {rs}")

