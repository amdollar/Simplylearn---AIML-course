import pandas as pd
import numpy as np

data = pd.read_csv('retail_dataset.csv')
print(data.head())

# Handling outliers: Calculate the NAN values 
# if available deal with mean or median
# Calculate the quartiles 
# calculate the IRS values to get Lower and Upper range
# get the data:

np.sort(data)
print(data)
q1s, q3s = data['Monthly_Sales'].quantile([0.25, 0.75])
print(q1s,  q3s)


q1f, q3f =data['Customer_Feedback_Score'].quantile([0.25,0.75])
# q3f = data['Customer_Feedback_Score'].quantile()

print(q1f,  q3f)

iqr1 = q3s - q1s
print(iqr1)

iqr2 = q3f - q1f
print(iqr2)

upper_limit1 = q3s+ (1.5 * iqr1)
lower_limit1 = q1s - (1.5 * iqr1)
print('Upper, lower limits 1: ')
print(upper_limit1, lower_limit1)
upper_limit2 = q3f + (1.5 * iqr2)
lower_limit2 = q1f - (1.5 * iqr2)
print('Upper, lower limits 2: ')
print(upper_limit2, lower_limit2)

final_data_s = data[(data['Monthly_Sales'] > lower_limit1) & (data['Monthly_Sales'] < upper_limit1)]
print(f'Processed Data: \n {final_data_s}')
print(final_data_s.describe())

# We no need to see this data, considering the Customer feedbase data is basically Discrete data.
# To validate this, we can check if the Max and Min values are within the range of Outliers or not ? 

# final_data_f = data[(data['Customer_Feedback_Score'] > lower_limit2) & (data['Customer_Feedback_Score'] < upper_limit2)]
# print(final_data_f)
