import pandas as pd
import numpy as np
import warnings
from scipy.stats import pearsonr
from scipy.stats import shapiro

warnings.simplefilter('ignore')


data = pd.read_csv('assignment_dataset.csv')
# print(data.head())

# ---  ------             --------------  -----
#  0   CustomerID         120 non-null    object        Numerical (Discrete)  
#  1   Age                120 non-null    int64         Numerical (Discrete)
#  2   Gender             115 non-null    object        Categorical
#  3   AnnualIncome       117 non-null    float64       Numerical (Continuous)
#  4   SpendingScore      119 non-null    float64       Numerical (Continuous)
#  5   PreferredCategory  119 non-null    object        Categorical

'''
1. Read the requirement and validate the data according to the given nature (Domain):
CustomerID (Unique identifier for customers) : ------------------- String values 
Age (Customer's age) ( 15 to 80 ) -------------------------------- Numerical (Discrete)
Gender (Male/Female/Other) --------------------------------------- Categorical
AnnualIncome (in $1000s) ----------------------------------------- Numerical (Continuous)
SpendingScore (1–100, indicating spending habits) ---------------- Numerical (Discrete)
PreferredCategory (Customer's preferred product category) (Valid categories are Electronics, Fashion, Home Decor, Sports, Groceries) ----Categorical


2. Make the data aligned with the given conditions. 
3. Handle NAN values using Statical methods:
    i. for numerical: mean, median.
    ii. for categorical: mode
4. 

'''
print(data.info())
# ========================================================== Handeling Numerical data:
# -------------------- Handle Age range: 15-80
data['Age'].loc[(data['Age'] < 15) | (data['Age'] > 80)] = np.nan
# Replace NANs with the median of this col value.
data['Age'].fillna(data['Age'].median(), inplace=True)
print(data['Age'])

# ------------------- Handle Annual Income values; (in $1000s)  so Handle -ve values, 
print((data['AnnualIncome'] < 0).sum())
# No negative values, since handle only missing values: 3
data['AnnualIncome'].fillna(data['AnnualIncome'].mean(), inplace=True)
print(data['AnnualIncome'].isna().sum()) 


# ------------------- Handle SpendingScore discrete values: Range: (1-100)
# print(data['SpendingScore'].loc[(data['SpendingScore'] < 1) | (data['SpendingScore'] > 100)])
data['SpendingScore'].loc[(data['SpendingScore'] < 1) | (data['SpendingScore'] > 100)] = np.nan
# print(data['SpendingScore'].loc[(data['SpendingScore'] < 1) | (data['SpendingScore'] > 100)])
# handle the nan values: 3
# print(data['SpendingScore'].isna().sum())
data['SpendingScore'].fillna(data['SpendingScore'].mean(), inplace=True)
print(data['SpendingScore'].isna().sum())

# =============================================== Handle Categorical data:

# Handle Gender col for specific given value: 
print(data['Gender'].unique()) # ['Female' 'Other' 'Male' nan 'male' 'female']
data['Gender'].replace(['male', 'female'], ['Male', 'Female'], inplace=True)
data['Gender'].fillna(data['Gender'].mode()[0], inplace=True)
print(data['Gender'].isnull().sum())


# Handle Preferred Category with the provided condition:
print(data['PreferredCategory'].unique()) # ['Sports' 'Groceries' 'Electronics' 'Home Decor' 'Fashion' nan 'Health', 'sports']a
data['PreferredCategory'].replace(['sports'], ['Sports'], inplace=True)
data = data[data['PreferredCategory'] != 'Health']
print(data['PreferredCategory'].unique()) # ['Sports' 'Groceries' 'Electronics' 'Home Decor' 'Fashion' nan]

data['PreferredCategory'].fillna(data['PreferredCategory'].mode()[0], inplace=True)
print(data['PreferredCategory'].isnull().sum())


# 2. Categorical Data Handling
# Use pd.get_dummies to encode the categorical column(s) into numerical format                   .

# Categorical cols: Gender, PreferredCategory

gender_h = pd.get_dummies(data['Gender'], dtype=int)
print(gender_h)


preffered_cat_h = pd.get_dummies(data['PreferredCategory'], dtype = int)
print(preffered_cat_h)

gen_con_data = pd.concat([gender_h, data.iloc[:]], axis=1)
# print(gen_con_data)

final_con_data = pd.concat([gen_con_data, data.iloc[:]], axis=1)
print(final_con_data)


# 3. Hypothesis Testing for Normality
# Perform a Shapiro-Wilk test or Kolmogorov-Smirnov test to check if numerical columns follow a normal distribution.
# Report the p-values and interpret the results.







'''
# 1. Missing Value Analysis
# Check for missing values in the dataset.
# Handle the missing values using appropriate techniques (e.g., mean/mode imputation or removal).

print(data.describe())
# AnnualIncome  and SpendingScore have 3 and 1 missing values respectively

print(data['AnnualIncome'].isnull().sum()) # 3
print(data['SpendingScore'].isnull().sum()) # 1

# Fill this missing data will mean()
data['AnnualIncome'].fillna(data['AnnualIncome'].mean(), inplace=True)
print(data['AnnualIncome'].isnull().sum())

data['SpendingScore'].fillna(data['SpendingScore'].mean(), inplace=True)
print(data['SpendingScore'].isnull().sum())


# 2. Categorical Data Handling
# Use pd.get_dummies to encode the categorical column(s) into numerical format                   .

# Categorical cols: Gender, PreferredCategory

data['Gender'].replace(['male','female'], ['Male', 'Female'], inplace=True)
print(data['Gender'].unique())  #['Female' 'Other' 'Male' nan 'male' 'female']

gender_h = pd.get_dummies(data['Gender'], dtype=int)
print(gender_h)

data['PreferredCategory'].replace(['sports'], ['Sports'], inplace=True)
print(data['PreferredCategory'].unique())

preffered_cat_h = pd.get_dummies(data['PreferredCategory'], dtype = int)
print(preffered_cat_h)

gen_con_data = pd.concat([gender_h, data.iloc[:]], axis=1)
# print(gen_con_data)

final_con_data = pd.concat([gen_con_data, data.iloc[:]], axis=1)
print(final_con_data)


# 3. Hypothesis Testing for Normality
# Perform a Shapiro-Wilk test or Kolmogorov-Smirnov test to check if numerical columns follow a normal distribution.
# Report the p-values and interpret the results.

# 1. TO do this we need to use one of the statical testing: Sharpio. 
# from scipy.stats import shapiro

SL = 0.05
corr, pval = shapiro(data['Age'])

if pval >= SL:
    print('Normally distributed')
else:
    print('Not normally distributed')



# 4. Hypothesis Testing for Correlation
# Calculate the correlation matrix for numerical columns.
# Perform a hypothesis test (e.g., Pearson's correlation test) for Age v/s Spending Score

# 1. Have a goal, and define a question:
# Is there any corelation/ linear relationship bw/ Age and Spending Score ? 

# 2 . Change the question into Hypothesis:
# H0: There is no relation b/w AGE and Spending Score
# H1: There is a liner relationship b/w AGE and Spending Score


# 3. Choose a Statical Fourmula to calculate the variable's correlationship
# from scipy.stats import pearsonr

# 4. Define the value of SL: 
SL = 0.05

# 5. Calculate the p-value with the help of statical method

curr, pvalue = pearsonr(data['Age'], data['SpendingScore'])
print(pvalue)

if pvalue <= SL:
    print('H(1): There is a linear relatioship b/w Age and Spending Score')
else:
    print('H(0): There is no linear relatioship b/w Age and Spending Score')

    # H(0): There is no linear relatioship b/w Age and Spending Score

# 5. Feature Elimination: Parametric and Non-Parametric Tests
# Age v/s SpendinG Score
# Annual Income v/s Spending Score


# 6. Chi-Square Test for Categorical Columns
# Apply a Chi-Square test to evaluate the relationship between categorical columns and the target variable for feature elimination.
# Gender v/s Spending Score

'''