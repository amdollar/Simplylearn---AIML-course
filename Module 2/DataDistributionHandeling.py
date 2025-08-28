import pandas as pd
import numpy as np
import seaborn as sns

salaries = pd.DataFrame([[1000],[2000],[3000],[4000],[5000],[6000],[7000],[8000],[9000],[2000000],[100000000]] , columns=['salary'])
# print(f'Data frame: \n {salaries}')

# EDA --- Exploratory Data Analysis
# Data Exploration ------------------ Descriptive Stats
#
#                       EDA
#                        |
#   ------------------------------------------------
#   |                                              |
# Numerical Based                           Visual Based
# EDA                                       EDA

#Get the statistical summary of the numerical column (salary)

#describe()---- used to calc the statistical summary

pd.set_option('display.float_format','{:.2f}'.format)

print(salaries['salary'].describe())
# count  1.100000e+01
# mean   9.276818e+06
# std    3.009543e+07
# min    1.000000e+03
# 25%    3.500000e+03
# 50%    6.000000e+03
# 75%    8.500000e+03
# max    1.000000e+08


# print(sns.displot(data=salaries['salary'], kind='kde'))

print(salaries['salary'].mean())


#To Extract Quartile Values, in numpy we have a function called percentile
q1 =  np.percentile(salaries['salary'], 25)
print(f'Quartile 1 values: {q1}')

q1, q2 = np.percentile(salaries['salary'], (25,50))
print(f'Quartile1 and 2 values: {q2}')

# Using pandas:
q1p,q2p,q3p,q4p = salaries['salary'].quantile([0.25,0.50,0.75, 1])
print(f'Quartile values using Pandas: \n {q1p,q2p,q3p,q4p}')

# Calculate the Outliers:
# Using Tukey's Method. (IQR Method)

# 1. Arrange data in asc order
# 2. Calculate the Q1 and Q3 quartiles
# 3. Calculate IQR
# 4. Calculate the Lower and upper bound (range).
# 5. Get the outliners

print(salaries)
# 1. Arrange data in asc order
np.sort(salaries)

# 2. Calculate the Q1 and Q3 quartiles
dq1,dq3 = salaries['salary'].quantile([0.25,0.75])
print('Calculated Quartiles 1 and Quartiles 3: ')
print(dq1,dq3)

# 3. Calculate IQR
IQR = dq3-dq1
print(f'Calculated IQR: {IQR}')

# 4. Calculate the Lower and upper bound (range).
upper_bound = dq3 + (1.5 * IQR) 
lower_bound = dq1 - (1.5 * IQR)
print('Calculated upper and lower bound')
print(upper_bound, lower_bound)

# 5. Get the outliners
processed_data = salaries[(salaries['salary'] < upper_bound) & (salaries['salary'] > lower_bound)]

print(f'Processed Data: \n {processed_data}')
print(processed_data.describe())
print(processed_data['salary'].mean())