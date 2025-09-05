import pandas as pd
import warnings
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

warnings.filterwarnings('ignore')

'''--------------------------------------------------Part 1: Exploratory Data Analysis (EDA)---------------------------'''
# 1. Load the dataset and display the first 10 rows.

data = pd.read_csv('customer_data_final.csv')
print(data.head(10))


# 2. Show dataset shape (#rows, #columns).
print(data.shape) # (200, 9)

# 3. Generate summary statistics for numerical features.
print(data.describe())
#        CustomerID         Age  AnnualIncome  SpendingScore  PurchaseCount  CustomerSatisfactionScore  LastPurchaseAmount
# count  200.000000  190.000000  1.890000e+02     190.000000     190.000000                 200.000000          200.000000
# mean   100.500000   46.773684  1.615097e+05      52.494737      23.010526                  55.208606           26.935221
# std     57.879185   22.118161  7.216219e+05      44.293125      14.467621                  33.088752           59.704055
# min      1.000000  -10.000000 -5.000000e+03     -50.000000       0.000000                  22.875095            0.051747
# 25%     50.750000   28.250000  5.841300e+04      25.250000      10.000000                  44.542404            5.902471
# 50%    100.500000   47.000000  1.084610e+05      52.000000      24.000000                  50.523127           15.223660
# 75%    150.250000   65.750000  1.605070e+05      77.750000      35.000000                  57.460044           28.062734
# max    200.000000  150.000000  9.999999e+06     500.000000      49.000000                 301.523671          562.082350

# 4. Count missing values per column and calculate their percentage.
def missing_data_percentage(col_name):
    
    size = data[col_name].isna().sum()
    percentage = size/200 * 100 
    return percentage

data_cols =data.columns

for i in data_cols:
    missing_per = missing_data_percentage(i)
    print(f'{i} col has total {missing_per} % of missing data')

# 5. Identify categorical features and list unique values.
print(data.describe(include='all'))
# Gender, City
print(data['Gender'].unique())  # ['Female' 'Male' nan 'Other']
print(data['City'].unique()) # ['Bangalore' 'Chennai' 'Mumbai' 'Pune' 'Delhi' nan]

# 6. Check for duplicate records.
data.drop_duplicates()
print(data.describe())

'''--------------------------------------------------Part 2: Handling Missing & Inappropriate Data----------------------------'''

# 1. Identify and impute missing values:
# ◦ Numerical → median/mean.
# ◦ Categorical → mode or “Unknown”.

#  Numerical values: CustomerID (Discrete)   Age (Discrete) AnnualIncome (Continuous)  SpendingScore (Continuous) 
#  PurchaseCount (Continuous) CustomerSatisfactionScore (Continuous) LastPurchaseAmount (Continuous)

data['Age'].fillna(data['Age'].median(), inplace=True)
data['AnnualIncome'].fillna(data['AnnualIncome'].mean(), inplace=True)
data['SpendingScore'].fillna(data['SpendingScore'].mean(), inplace=True)
data['PurchaseCount'].fillna(data['PurchaseCount'].mean(), inplace=True)
data['CustomerSatisfactionScore'].fillna(data['CustomerSatisfactionScore'].mean(), inplace=True)
data['LastPurchaseAmount'].fillna(data['LastPurchaseAmount'].mean(), inplace=True)

# print(data.describe())

# Categorical → mode or “Unknown”.
# Gender, City
data['Gender'].fillna(data['Gender'].mode()[0], inplace=True)
data['City'].fillna(data['City'].mode()[0], inplace=True)
print(data.describe(include='all'))
# count   200.000000  200.000000     200  2.000000e+02     200.000000     200.000000    200                 200.000000          200.000000

# 2. Find invalid ages (<10 or >100) and treat them as missing.

data['Age'].iloc[(data['Age'] < 10) | (data['Age'] > 100)] = np.nan

print(data.describe())

# 3. Correct invalid incomes (negative or >1,000,000)
data['AnnualIncome'].iloc[(data['AnnualIncome'] < 0) | (data['AnnualIncome'] > 1000000)] = np.nan
print(data.describe())

# 4. Ensure purchase counts are non-negative integers.
data['PurchaseCount'].iloc[(data['PurchaseCount'] < 0) ] = np.nan
print(data.describe())

# 5. Validate gender and city columns for unexpected categories.
# data['Gender'].replace(['Other'], ['Gay'], inplace=True) # ['Female' 'Male' 'Gay
print(data['Gender'].unique())
print(data['City'].unique())
# ['Female' 'Male' 'Other']
# ['Bangalore' 'Chennai' 'Mumbai' 'Pune' 'Delhi']

'''--------------------------------------------------Part 3: Handling Outliers--------------------------------'''
# 1. Detect outliers in AnnualIncome and SpendingScore using boxplots.
print(data['AnnualIncome'].describe())             
# sns.boxplot(data['AnnualIncome'])
# plt.show()

sns.boxplot(data['SpendingScore'])
# plt.show()
# sns.boxplot(data['SpendingScore'])
# plt.show()
# In visuals we can see, one -ve data, also one 500 aprox data.

# 2. For CustomerSatisfactionScore, decide whether it is closer to a normal distribution and
# handle outliers (e.g., Z-score method).

print(data['CustomerSatisfactionScore'].describe())
# The difference b/w mean and 50% is just 5 points or 10%, we can say data is normally distributed. Anyhow, we can handle outliers

data = data[np.abs(stats.zscore(data['CustomerSatisfactionScore']) <= 3)].copy()
print(data)

# 3. Analyze the correlation between AnnualIncome and SpendingScore (before and after
# handling outliers).
print(data['AnnualIncome'].describe(), data['SpendingScore'].describe())

# 4. Compare average purchase count by gender.
# female: 20%
# male: 80%

total_purchase = data['PurchaseCount'].sum()
male_data = data[data['Gender'] == 'Male']
purchase_count_m = male_data['PurchaseCount'].sum()
avg_male_per = (purchase_count_m / total_purchase) * 100
print(purchase_count_m)
print(avg_male_per)

female_data = data[data['Gender'] == 'Female']
purchase_count_f = female_data['PurchaseCount'].sum()
avg_fmale_per = (purchase_count_f/total_purchase) * 100
print(purchase_count_f)
print(avg_fmale_per)

# 5. Which age group (Young <30, Middle 30–55, Senior >55) has the highest spending score?
# creation of datasets for these age groups
# fetch the sum of each dataset's Spending score and compare
# Can we automate this complete things :
young_ds = data[data['Age'] < 30]
print(young_ds.head())
print(young_ds['SpendingScore'].sum())

middle_ds = data[(data['Age'] > 30) & (data['Age'] < 55)]
print(middle_ds.head())
print(middle_ds['SpendingScore'].sum())

senior_ds = data[(data['Age'] > 55)]
print(senior_ds.head())
print(senior_ds['SpendingScore'].sum())

highest_score = max(young_ds['SpendingScore'].sum(),middle_ds['SpendingScore'].sum(),senior_ds['SpendingScore'].sum())
print(f'Highest score is: {highest_score} , by senior group.')

# 6. Find top 5 customers with the highest LastPurchaseAmount.
print(data.nlargest(5, 'LastPurchaseAmount'))


# 7. Compare CustomerSatisfactionScore across cities – which city has the most satisfied customers?
cities = data['City'].unique()
print(cities)
print(data['CustomerSatisfactionScore'])

# ['Bangalore' 'Chennai' 'Mumbai' 'Pune' 'Delhi']
# Calculate the data for each city individually 

bangalore_s = data['CustomerSatisfactionScore'].loc[(data['City'] == 'Bangalore')].sum() 
print(f'Customer satisfaction score sum for Bangalore : {bangalore_s}')

Chennai_s = data['CustomerSatisfactionScore'].loc[(data['City'] == 'Chennai')].sum()
print(f'Customer satisfaction score sum for Chennai : {Chennai_s}')

Mumbai_s = data['CustomerSatisfactionScore'].loc[(data['City'] == 'Mumbai')].sum()
print(f'Customer satisfaction score sum for Mumbai : {Mumbai_s}')

Pune_s = data['CustomerSatisfactionScore'].loc[(data['City'] == 'Pune')].sum()
print(f'Customer satisfaction score sum for Pune : {Pune_s}')

Delhi_s = data['CustomerSatisfactionScore'].loc[(data['City'] == 'Delhi')].sum()
print(f'Customer satisfaction score sum for Delhi : {Delhi_s}')

max = max(bangalore_s, Mumbai_s, Pune_s, Delhi_s, Chennai_s)
print(f'Maximium satisfied customers are from : Delhi')

# 8. Do customers with high satisfaction scores also spend more on average?

# 9. Find the relationship between PurchaseCount and CustomerSatisfactionScore.
# Not sure about the ask
print(data['PurchaseCount'].head(), data['CustomerSatisfactionScore'].head())

# 10. Identify which gender shows the highest repeat purchases (PurchaseCount).
genders = data['Gender'].unique()
print(genders)

male_re = data['PurchaseCount'].loc[data['Gender'] == 'Male'].sum()
print(male_re)

female_re = data['PurchaseCount'].loc[data['Gender'] == 'Female'].sum()
print(female_re)

other_re = data['PurchaseCount'].loc[data['Gender'] == 'Other'].sum()
print(other_re)

res = ''
if male_re >= female_re:
    res = male_re
else: 
    res = female_re
if res <= other_re:
    res = other_re
print(res)