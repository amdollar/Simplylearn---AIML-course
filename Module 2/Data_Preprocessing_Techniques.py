import pandas as pd
import numpy as np


data = pd.read_csv('preprocessExample.csv')
print(data.info())

#  #   Column     Non-Null Count  Dtype
# ---  ------     --------------  -----
#  0   Country    9 non-null      object
#  1   Age        9 non-null      float64
#  2   Salary     9 non-null      float64
#  3   Purchased  10 non-null     object

# Identify if missing value exists in columns
print(data.isna().sum())
# We can see there is some NAN values in the 0,1 and 2nd col, we need to deal with this data. 

print(data.describe())
# Identify the col and data types before analysis
print(data.info())


#  0   Country    9 non-null      object  ---> Categorical : Fill with Mode's first value Mode[0]
#  1   Age        9 non-null      float64 ---> Numerical -> + allowed, - not allowed, decimal allowed (discrete) :  fill with median
#  2   Salary     9 non-null      float64 ---> Numerical -> + allowed, - not allowed, decimal allowed (continuous) :  fill with mean
#  3   Purchased  10 non-null     object  ---> Categorical

# Handling country col:
data['Country'].fillna(data['Country'].mode()[0], inplace=True)
print(data)
# Handeling age col:
age_median = data['Age'].median()
data['Age'].fillna(data['Age'].median(), inplace=True)
print(data)

# Handeling Salary col:
sal_mean = data['Salary'].mean()
print(sal_mean)
data['Salary'].fillna(data['Salary'].mean(), inplace = True)
print(data)

# Now this is how we have handeled the missing data before processing the data. 