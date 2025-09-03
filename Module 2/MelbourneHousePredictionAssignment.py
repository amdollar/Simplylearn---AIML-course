import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')
data = pd.read_csv('melb_data.csv')
# print(data)
# print(data.info())



# as per the requirement, we only need to use the data based on: Car Parking availability, Landsize, BuildingArea, YearBuilt
# These values should be handeled using inferial stats

# BuildingArea   7130 non-null   float64   -> + allowed, - not allowed, deciml allowed : Continuous.
# Landsize       13580 non-null  float64   -> + allowed, - not allowed, deciml allowed: Continuous.
# YearBuilt      8205 non-null   float64   -> + allowed, -not allowed, decimal not allowed : Discrete.
# Car            13518 non-null  float64   -> + allowed, - not allowed, deciml not allowed: Discrete.

# Create a small dataset based on these values:
small_data = data[['BuildingArea', 'Landsize', 'YearBuilt','Car']]
# To handle the continuous data: replace missing values with the Mean values
# BuildingArea, LandSize

print(small_data['BuildingArea'].isna().sum())
print(small_data['Landsize'].isna().sum())
print(small_data['YearBuilt'].isna().sum())
print(small_data['Car'].isna().sum())

small_data['BuildingArea'].fillna(small_data['BuildingArea'].mean(), inplace=True)
small_data['Landsize'].fillna(small_data['Landsize'].mean(), inplace=True)
print(small_data['BuildingArea'].isna().sum())
print(small_data['Landsize'].isna().sum())

# To handle the discrete data: replace missing values with Median values.
small_data['YearBuilt'].fillna(small_data['YearBuilt'].median(), inplace=True)
small_data['YearBuilt'] = small_data['YearBuilt'].astype(int)

# We are already doing Domain and statical handeling below
small_data['Car'].fillna(small_data['Car'].median(), inplace=True)
small_data['Car'] = small_data['Car'].astype(int)
print(small_data['Car'].unique())

#As per the domain data, we need to handle the following conditions:
# The usual  car parking alloted to each house as per govt policy is 1 : replace NAN/0 with 1
# The Minimum build area for the building as per govt norms is 80: Replace NAN and 0 with 80
# Maximums buildings as per the census survey was built in year 2000.: Replace data greater than 2000 
data2 = pd.read_csv('melb_data.csv')
domain_data = data2[['Car', 'BuildingArea', 'YearBuilt']]

domain_data['Car'].fillna(domain_data['Car'].median(), inplace=True)
domain_data['Car'] = domain_data['Car'].astype(int)

# Replace 0 car parkings with default value 1. 
domain_data['Car'] = domain_data['Car'].replace(0, 1)
print(domain_data['Car'].unique())

domain_data['BuildingArea'].fillna(domain_data['BuildingArea'].mean(), inplace=True)
domain_data['BuildingArea'] = domain_data['BuildingArea'].astype(int)

domain_data['BuildingArea'] = domain_data['BuildingArea'].mask(domain_data['BuildingArea']< 80 , 80)

print(domain_data.head())

# Handle year build, if data is 0 then replace with 2000
domain_data['YearBuilt'].fillna(2000, inplace=True)
domain_data['YearBuilt'] = domain_data['YearBuilt'].astype(int)
print(domain_data['YearBuilt'].unique())
print(domain_data.head())



# Handle the null/NAN values in the columns.

# 0   Suburb         13580 non-null  object     -> String
#  1   Address        13580 non-null  object    -> String
#  2   Rooms          13580 non-null  int64     -> + allowed, - not allowed, deciml not allowed
#  3   Type           13580 non-null  object    -> Categorical
#  4   Price          13580 non-null  float64   -> + allowed, - not allowed, deciml allowed
#  5   Method         13580 non-null  object    -> Categorical
#  6   SellerG        13580 non-null  object    -> Categorical
#  7   Date           13580 non-null  object    -> Date type
#  8   Distance       13580 non-null  float64   -> + allowed, - not allowed, deciml allowed
#  9   Postcode       13580 non-null  float64   -> + allowed, - not allowed, deciml not allowed
#  10  Bedroom2       13580 non-null  float64   -> + allowed, - not allowed, deciml not allowed
#  11  Bathroom       13580 non-null  float64   -> + allowed, - not allowed, deciml not allowed
#  12  Car            13518 non-null  float64   -> + allowed, - not allowed, deciml not allowed
#  13  Landsize       13580 non-null  float64   -> + allowed, - not allowed, deciml allowed
#  14  BuildingArea   7130 non-null   float64   -> + allowed, - not allowed, deciml allowed
#  15  YearBuilt      8205 non-null   float64   -> + allowed, -not allowed, decimal not allowed
#  16  CouncilArea    12211 non-null  object    -> Categorical
#  17  Lattitude      13580 non-null  float64   -> + allowed, -not allowed, decimal allowed
#  18  Longtitude     13580 non-null  float64   -> + allowed, -not allowed, decimal allowed
#  19  Regionname     13580 non-null  object    -> Categorical
#  20  Propertycount  13580 non-null  float64   -> int -> + allowed, - not allowed, deciml not allowed

# def continuous_vals(colname):

#     # check for missing values:
#     mean = data[colname].mean()
#     data[colname].fillna(mean, inplace= True)

# # To be handeled with mean:
# values_to_handle = ['YearBuilt', 'BuildingArea', ' ',]

# for i in values_to_handle:
#     continuous_vals(i)

# print(data.info())
# print(data.describe())

# # Handle categorical data: CouncilArea
# # Fill with mode:

# def category_data(colname):
#     mode = data[colname].mode()[0]
#     data[colname].fillna(mode, inplace=True)


# values_to_handle_con = ['CouncilArea']
# for i in values_to_handle_con:
#     category_data(i)


# # Handle range data: 
# print(data['BuildingArea'].describe())

# def range_data(colname):
#     np.sort(data[colname])
#     q1,q3 = data[colname].quantile([0.25,0.75])
#     # print(q1, q3)

#     IQR = q3 - q1
#     ur = q3 + (1.5 * IQR)
#     lr = q1 - (1.5 * IQR)

#     return ur, lr   

# # print(data.info())
# ur, lr = range_data('BuildingArea')
# print(ur, lr)
# # print(data['BuildingArea'].describe())