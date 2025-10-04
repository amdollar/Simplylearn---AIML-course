import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

data = pd.read_csv('preprocessExample.csv')

print(data.info())

# print(data.isna().sum())

# Country      1 : mode[0]
# Age          1 : median
# Salary       1 : mean
# Purchased    0

#1.  Handeling the data Null/NAN columns using Pandas.
# data['Country'].fillna(data['Country'].mode()[0], inplace = True)
# data['Age'].fillna(data['Age'].median(), inplace=True)
# data['Salary'].fillna(data['Salary'].mean(), inplace=True)
# print(data)

''' Predict based on data (Age, Salary and Country) that he will make an order (Purchase) on the website or not:
1. In this we will have to divide the data in two parts:
    i. Feature (Data used to predict) : Country, Age, Salary
    ii. Label (Data that will be target): Purchased
ML Algo category: Superwised learning | Classification: Binary (It is yes or no type of data to be predicted)

2. Model Training: Inferential statics: 
    For this data has to be:
    i. Complete.
    ii. Numerical format
3. Here we will use new package scikitlearn: For that:
    i. Data has to be in NUMPY Array for faster processing.
     

'''
# Step 1: Saperate the data in the form of Features and Label:

features = data.iloc[: , [0,1,2]].values
# print(features)

label = data.iloc[:, [3]].values
# print(label)

# Step 2: Handle the NAN data using sklearn.

# i. import the package

print(features[:,0]) 
# ['France' 'Spain' 'Germany' 'Spain' 'Germany' 'France' 'Spain' 'France' nan 'France'] Handle this nan value
print(features)
 
# ii. Instintiate the object of SimpleImputer

simpleimputer_country = SimpleImputer(missing_values=np.nan, strategy='most_frequent')

# iii. Fit the missing value in the Object: in this case calculate the mode of the col
simpleimputer_country.fit(features[:,[0]])

# iv. Transform the data in dataset with this values
features[:, [0]] = simpleimputer_country.transform(features[:, [0]])
print(features)

# Repeate same for Age and Salary:

si_age = SimpleImputer(fill_value=np.nan, strategy='median')
si_age.fit(features[:, [1]])

features[:,[1]] = si_age.transform(features[:, [1]])

si_salary = SimpleImputer(strategy='mean', fill_value = np.nan)
si_salary.fit(features[:, [2]])

features[:,[2]] = si_salary.transform(features[:, [2]])
print(features)

# Round off the col float values. ie: 63777.77777777778
np.round(features[:, [2]].astype(float), 2)
print(features)

# Convert data to Numerical values : Using OHE here. 
# This can be done using sklean's OneHotEncoder 