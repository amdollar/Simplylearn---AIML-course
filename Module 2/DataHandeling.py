import pandas as pd
import numpy as np

data = pd.read_csv('datasetExample.csv')

print(data.info())

#  #   Column           Non-Null Count  Dtype
# ---  ------           --------------  -----
#  0   CustomerID       11 non-null     int64
#  1   Age_Group        11 non-null     object
#  2   Rating(1-5)      11 non-null     int64
#  3   Hotel            11 non-null     object
#  4   FoodPreference   11 non-null     object
#  5   Bill             11 non-null     int64
#  6   NoOfPax          11 non-null     int64
#  7   EstimatedSalary  11 non-null     int64
#  8   Age_Group.1      11 non-null     object

# here .1 means it's 1st duplicate copy of a COL.

print(data)
'''
-------------------------------------------------------------------------------1_Handling Inappropriate Data---------------------------------------
1 . Identify datatype of each column:
# CustomerId:       Numerical (Continuous)
# Age_Group:        Categorical
# Rating(1-5):      Numerical (Discrete) 
# Hotel:            Categorical 
# FoodPreference:   Categorical
# Bill:             Numerical   (Continuous)
# NoOfPax:          Numerical   (Continuous)
# EstimatedSalary:  Numerical   (Continuous)
# Age_Group.1:      Categorical (duplicate col)


2. Remove the Duplicate Records: 

 # we have age_group as a duplicate col here, we can remove that
 # In Pandas we can detect and remove Duplicate records using drop_duplicates()
 # This operation will not make any change in real data, we need to save this by 
 # indicating, inplace = true
'''
data.drop_duplicates(inplace=True, ignore_index=True)
 
print(f'After droping duplicate columns \n : {data}')

'''
3. Check and Remove the duplicate columns. 
   
# In pandas duplicate column can be identified only if the dataset has header and
# column names are same

# Use coln name to identify duplicate column, --- if you find anything with suffix .(number)
# .1 e.g., chances are its a duplicate column
print(data.info())
# Validate if these columns are really same:


'''
print(data[ data['Age_Group'] != data['Age_Group.1'] ])
#If the above is a null set, it means the given columns are same. Therefore eliminate one of them

data.drop(columns='Age_Group.1', inplace=True)
print(f'After droping the duplicate column \n : {data.info()}')

'''
4. Handle the numeric values: a. Continuous, b. Discrete
    
    a. first for CONTINUOUS data, Follow the above listed rules for numerical data:

    1. customerId : numerical ---> + allowed - not allowed decimal not allowed.
    2. Bill       : numerical ---> + allowed - not allowed decimal allowed
    3. NoOfPax    : numerical ---> + allowed - not allowed decimal not allowed.
    4. Estimated sal: numerical--> + allowed - not allowed decimal allowed

'''
print(data[data['CustomerID'] < 0])
# Output:
# Columns: [CustomerID, Age_Group, Rating(1-5), Hotel, FoodPreference, Bill, NoOfPax, EstimatedSalary]
# Index: []
# means there is no negative values in this column, No actions. 
print(data[data['Bill'] < 0])
# Output:
#    CustomerID Age_Group  Rating(1-5)   Hotel FoodPreference  Bill  NoOfPax  EstimatedSalary
#   9          10     30-35            5  RedFox        non-Veg -6755        4            87777
# Means we need to handle this data, by deleting this column entry. Setting it as NAN

# Steps: 
#Deleting column entry ---> Technically it means replace value with NaN
#This NaN is available as a value in np package
data['Bill'].loc[data['Bill'] < 0] = np.nan
print(f'After handeling the negative Bills \n {data}')

# handle the NoOfPax 
data['NoOfPax'].loc[data['NoOfPax'] < 0]= np.nan
print(f'After handeling the negative No of pax:  \n  {data}')


# Handle salary, change the datatype to the float.
# Handle the negative salary
data['EstimatedSalary'].loc[data['EstimatedSalary']< 0] = np.nan
data['EstimatedSalary'] = data['EstimatedSalary'].astype(float)
print(data)
print(f'After handeling the negative and empty Estimated salary:  \n {data}')


# Deal with discrete numerical data: Rating 
# Rating(1-5)      11 non-null     int64   -----> Numerical (Discrete) - +ve allowed, -ve not allowed, decimal not allowed, data in range of 1 to 5 only
# Handeling range and -ve values

data['Rating(1-5)'].loc[(data['Rating(1-5)'] > 5.0) | (data['Rating(1-5)'] < 1.0)]  = np.nan
print(f'After handeling negaving Ratings : \n {data}')


# 5. If the column are Categorical Columns, perform the following:
#       - Get the unique values of the column
#       - Handle the data that has SPELLING ERRORS, CASE ERRORS or any FORMATTING ERRORS. Ensure Case is UNIFIED. (NorMAlized --- e.g. lowercase only)
#       - Check whether the categories/groups found in unique values match the domain spec.
#
#         If any unusual category found, DELETE that SPECIFIC RECORD.

#       Age-Group         Categorical
#       Hotel:            Categorical (assume)
#       FoodPreference:   Categorical

print(data['Age_Group'].unique())
# ['20-25' '30-35' '25-30' '35+']
print(data['Hotel'].unique()) 
# ['Ibis' 'LemonTree' 'RedFox' 'Ibys']
# Ibis and Ibys are two nearly duplicated values, we can make hande this by replacing one with another

# data['Hotel'].loc[data['Hotel'] == 'Ibys'] = 'Ibis'
# print(f'After handeling Hotels as unique cols : \n {data}')

# We can do this another way also:
data['Hotel'].replace('Ibys', 'Ibis', inplace= True)
print(f'After handeling Hotels as unique cols : \n {data}')

# Food preferences:
print(data['FoodPreference'].unique())
# ['veg' 'Non-Veg' 'Veg' 'Vegetarian' 'non-Veg'

# There should be two options only, Veg and Non-veg, but we see here multiple, we need to change them as well. 

data['FoodPreference'].replace(['veg','Vegetarian'], 'Veg', inplace=True)
data['FoodPreference'].replace('non-Veg', 'Non-Veg', inplace=True)
print(f'After handeling Food preferences as unique cols : \n {data}')
# Validating again:
print(data['FoodPreference'].unique())
# ['Veg' 'Non-Veg']