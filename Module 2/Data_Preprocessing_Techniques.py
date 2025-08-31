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


# Now this is how we have handeled the missing data before processing the data. 

# act

# 100 MBPS * 3 Month -> Free installation / Free router. (943 PM )  
# 50MBPS -> 236 (650 inc) 6 month -> 3900 actuoal : 3692 ->  
# Shifting: 400 / first time free. 

# ---------------------------- Handeling the categorical data (i.e.: Country): Using One Hot Encoding technique.
# Its nothing but, creating the dummy variables to represent the data only for active rows. 
# If doing manually:
#  1. Fetch unique values
# 2. sort in asc order
# 3. Create table of dummy valiabes , such a way that only active data is displayed in rows.
#    France  Germany  Spain
# 0       1        0      0
# 1       0        0      1
# 2       0        1      0
# 3       0        0      1
# 4       0        1      0
# 5       1        0      0
# 6       0        0      1
# 7       1        0      0
# 8       1        0      0
# 9       1        0      0

print(pd.get_dummies(data['Country'], dtype=int))
# print(final_data)

# Next we can concat this data with real table 


final_data= pd.concat([pd.get_dummies(data['Country'], dtype=int), data.iloc[:,[1,2,3]] ], axis = 1) # axis = 1: columnwise concatination.
print(final_data)
#    France  Germany  Spain   Age        Salary Purchased
# 0       1        0      0  44.0  72000.000000        No
# 1       0        0      1  27.0  48000.000000       Yes
# 2       0        1      0  30.0  54000.000000        No
# 3       0        0      1  38.0  61000.000000        No
# 4       0        1      0  40.0  63777.777778       Yes
# 5       1        0      0  35.0  58000.000000       Yes
# 6       0        0      1  38.0  52000.000000        No
# 7       1        0      0  48.0  79000.000000       Yes
# 8       1        0      0  50.0  83000.000000        No
# 9       1        0      0  37.0  67000.000000       Yes

print(final_data.info())