import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

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
data.dropna(inplace=True)

features = data.iloc[:, [0]].values
print(features)

label = data.iloc[:, [1]].values
print(label)

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

# 3. Initialization of Algo: LinearRegression

model = LinearRegression()

# start the training:
model.fit(x_train, y_train)

# Check the quality of the model:
# a. Calculate the evaluation score of the train and test score
#       train_score ->    score generated for training data
#       test_score -> score generated for testing data
# b. Below condition: 
#       if test_score > train_score and test_score >= CL (Confidence Level)
#                   approve model
#       else: 
#                   reject model

train_score = model.score(x_train, y_train)
test_score = model.score(x_test, y_test)
print(f"Test Score is {train_score} and Train Score is {test_score}")

# We need to identify the train and test split that gives best result, so we can perfrom this testing for all the dataset 
# Example:
# WE will choose the nerest values: 
# CL = 0.95
# for rs in range(1,101):
#   X_train,X_test,y_train,y_test = train_test_split(features,
#                                                    label,
#                                                    test_size=0.2,
#                                                    random_state=rs
#                                                    )

#   model = LinearRegression()

#   model.fit(X_train,y_train)

#   trainScore = model.score(X_train,y_train)
#   testScore = model.score(X_test,y_test)

#   if testScore > trainScore and testScore >= CL :
#     print(f"Test Score : {testScore} TrainScore : {trainScore} for RandomState {rs}")

# Deploy the model:

y_experience = float(input("Enter the years of experience: "))
predict_Salary = model.predict(np.array([[y_experience]]))

print(f" Salary for {y_experience} years of experience is $ {np.round(predict_Salary, 2)}")

# Export the model for the deployment using pickle

pickle.dump(model, open('SalaryPredictor.pkl', 'wb'))