import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

# Question: Create a model that can predict the profit of the company baed on company's 
#   Spending pattern and company's Location
# SL: 0.1
# Deploy the model

data = pd.read_csv('50_Startups.csv')
print(data)
# print(data.info())

# R&D Spend  Administration  Marketing Spend       State     Profit
# No nan data in the cols, so we need not to handle the filling the data

# 1. Saperate the features and label cols:
# Features: R&D Spend  Administration  Marketing Spend       
# Label: State 

features = data.iloc[:, [0,1,2,3]].values
print(features)
# There is one column named State: we need to convert that to the numerical
# Because for sklearn there are following conditions:
# 1. Data should be complete  -> done
# 2. Data should be numerical -> need to do for State col (OHE)
# 3. Data should be in 2d Numpy array

# 2. Convert state data to numerical using OHE:

state_imputer = OneHotEncoder(sparse_output=False)
state_values = state_imputer.fit_transform(features[:,[3]])
print(state_values)

final_features = np.concatenate((state_values, features[:, [0,1,2]]), axis=1)
print(final_features)

label = data.iloc[:, [4]].values
# print(label)


#2. Split the data in test and train:
x_train, x_test, y_train, y_test = train_test_split(final_features, label, 
                                                    test_size=0.2, 
                                                    random_state=5)

#3. Select the model: Linear Regression:
model = LinearRegression()
model.fit(x_train, y_train)

#4. test the score of the model:
train_score = model.score(x_train, y_train)
test_score = model.score(x_test, y_test)

print(f"Test Score : {test_score} TrainScore : {train_score}")

# While taking input the state input will be provided in String, 
# We need to convert this input in OHE to execute the model;
state_category = state_imputer.categories_
print(state_imputer.transform(np.array([['Florida']])))

print(state_category)
# 5. Predict, Deploy the model

state_name = input("Enter the state name: ")
r_d_spend = float(input('Enter the R&D Spend values: '))
administration = float(input('Enter the Administration Spend values: '))
marketing_spend = float(input('Enter the Marketing Spend Spend values: '))

print(f'State name: {state_name}, R&D spend value: {r_d_spend}, Adminstration value: {administration}, Marketing spend value: {marketing_spend}')

if state_name in state_imputer.categories_[0]:
    state = state_imputer.transform(np.array([[state_name]]))
    feature_value = np.concatenate((state, np.array([[r_d_spend, administration, marketing_spend]])), axis = 1)
    predicted_value = model.predict(feature_value)
    print(f'Predicted profit based on provided values is: {predicted_value}')
else:
    print(f'State is not recognized by ai model!')



# Task 2:  Export the model and use it in different file:

pickle.dump(model, open('ProfitPredictor.pkl', 'wb'))
pickle.dump(state_imputer, open('StateConvertor.obj', 'wb'))