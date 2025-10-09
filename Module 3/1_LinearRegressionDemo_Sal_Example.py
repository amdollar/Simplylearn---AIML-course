import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

#Goal: Create a model that can predict the salary of the employee based on his/her years of experience 
data = pd.read_csv('Salary_Data.csv')
# print(data)


# Meta data: 
# feature= Age, label= salary, label data_type= Continuous_ND, Algo= Regression

# Rules for regression:
# 1. Data must be complete - Fill the data with statical algos if NA
# 2. Data must be strictly numerical - Change categorical data to numerical using Encodings
# 3. Ensure data (features/labels) are represented in Numpy Array

# 1. Split data in the features and labels:
data.dropna()

features = data.iloc[:, [0]].values
# print(features)

label = data.iloc[:, [1]].values
# print(label)

# 2. Split the data in train and test
# x_train ------> feature training set
# x_test ------> feature testing set
# y_train -----> label training set
# y_test ------> label testing set
# we consider X axis as feature and Y axis as Labels
# use train_test_split class from Sklearn.modle_selection

x_train,x_test,y_train,y_test = train_test_split(features, label, test_size=0.2, random_state=9)
print(x_test)
# Here as mentioned 20% of the feature and label data will be used to perform testing and other remaining 80% for training the model
