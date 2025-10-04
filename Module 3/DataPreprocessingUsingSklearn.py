import pandas as pd
import numpy as np

data = pd.read_csv('preprocessExample.csv')

print(data.info())

print(data.isna().sum())

# Country      1 : mode[0]
# Age          1 : median
# Salary       1 : mean
# Purchased    0

#1.  Handeling the data Null/NAN columns using Pandas.
data['Country'].fillna(data['Country'].mode()[0], inplace = True)
data['Age'].fillna(data['Age'].median(), inplace=True)
data['Salary'].fillna(data['Salary'].mean(), inplace=True)
print(data)

''' Predict based on data (Age, Salary and Country) that he will make an order (Purchase) on the website or not:
1. In this we will have to divide the data in two parts:
    i. Feature (Data used to predict) : Country, Age, Salary
    ii. Label (Data that will be target): Purchased

'''
#2. Handeling the data Null/Nan columns using sklearn