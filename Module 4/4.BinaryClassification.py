# Creating a model that can Predict if a customer is a good or Bad, based on if he is going to make a Purchase or not. 
# This prediction will be made on the basis of Age and Salary of an individual

import pandas as pd
import numpy as np
from scipy.stats import shapiro
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

data = pd.read_csv('Social_Network_Ads.csv')
print(data.head(5))

# 1. Check the data quality and perform Preprocessing:
print(data.isna().sum()) # No null values
# Age                0
# EstimatedSalary    0
# Purchased          0

# 2. Splitting data in features and labels:
features = data.iloc[:,[0,1]].iloc
labels = data.iloc[:,[2]].iloc

# 3. Check if data is Normally distributed or not:
cols = ['Age', 'EstimatedSalary', 'Purchased']

def get_scaler(cols):
    threshold = 0.05
    for col in cols:
        print(col)
        curr, p_val = shapiro(data[col])

        if p_val >= threshold:
            print(f'{col} Data is normally distributed')
            return StandardScaler()
        else:
            print(f'{col} Data is not normally distributed')
            return RobustScaler()

rs = RobustScaler()
s_age = rs.fit_transform(features[:, [0,1]])

mn = MinMaxScaler()
s_purchased = mn.fit_transform(labels[:, [2]])

# 
