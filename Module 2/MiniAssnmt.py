import pandas as pd
import numpy as np
from sklearn.preprocessing  import LabelEncoder # 1.6.1
from category_encoders import TargetEncoder # 2.6.4
from scipy import stats
import warnings

warnings.filterwarnings('ignore')
data = pd.read_csv('categorical_data_assignment.csv')

# print(data.head())

print(data.describe(include='all'))


#        Customer_ID        Age  Gender     City  Annual_Income Education_Level  Purchase_Frequency Subscription_Status  Credit_Score Product_Category
# count           50  45.000000      50       50      50.000000              50           50.000000                  50     45.000000               50
# unique          50        NaN       3        5            NaN               4                 NaN                   2           NaN                5
# top         CUST_1        NaN  Female  Chicago            NaN             PhD                 NaN              Active           NaN           Sports
# freq             1        NaN      18       12            NaN              20                 NaN                  25           NaN               14
# mean           NaN  39.777778     NaN      NaN   52896.709285             NaN           17.960000                 NaN    654.899579              NaN
# std            NaN  13.059112     NaN      NaN   24322.517256             NaN           19.187538                 NaN     54.945323              NaN
# min            NaN  19.000000     NaN      NaN   21218.431771             NaN            1.000000                 NaN    548.742871              NaN
# 25%            NaN  29.000000     NaN      NaN   39978.286208             NaN            6.000000                 NaN    617.333538              NaN
# 50%            NaN  39.000000     NaN      NaN   49540.807661             NaN           16.000000                 NaN    654.093707              NaN
# 75%            NaN  50.000000     NaN      NaN   56843.105558             NaN           22.000000                 NaN    682.569563              NaN
# max            NaN  64.000000     NaN      NaN  150000.000000             NaN          100.000000                 NaN    842.636575              NaN

# 1. Load the dataset and show first 5 records:
print(data.head())

# 2. Categorical colums:
# Gender, City(we can as of now), Education_level, Subscription_Status, Product_Category

# 3. Convert Gender column into numerical using One-Hot-Encoding:
numerical_dummies = pd.get_dummies(data['Gender'], dtype=int)
print(numerical_dummies)

# 4. Apply Lable encoding in City column:
label_encoder = LabelEncoder()
encoded_city = label_encoder.fit_transform(data['City'], )
print(encoded_city)
# [0 4 0 3 2 2 4 1 4 1 2 4 3 0 1 0 0 4 0 3 0 4 3 2 0 4 2 2 1 3 0 4 1 3 1 3 3
#  0 3 1 3 0 0 1 2 2 3 4 3 1]

# 5. Convert the Product_Category column using Frequency Encoding:
frequency_en = data['Product_Category'].value_counts(normalize=True)
print(frequency_en)

# 6. Perform Target Encoding on Subscription_Status column using Purchase_Frequency as the target  
encoder = TargetEncoder(cols=['Subscription_Status']) # Specify the categorical column(s) to encode
# Fit the encoder on the training data (categorical feature and target)
encoder.fit(data['Subscription_Status'], data['Purchase_Frequency'])
data['Subscription_Encoded'] = encoder.transform(data['Subscription_Status'])
print(data)


# 7. Compare Label Encoding and One-Hot-Encoding results:
#     Female  Male  Other
# 0        0     0      1
# 1        0     0      1
# 2        0     1      0
# 3        0     0      1

# [0 4 0 3 2 2 4 1 4 1 2 4 3 0 1 0 0 4 0 3 0 4 3 2 0 4 2 2 1 3 0 4 1 3 1 3 3
#  0 3 1 3 0 0 1 2 2 3 4 3 1]


'''------------------------------------------------ Handle Missing Values-----------------------------------'''
# 8. Identify missing values in the Dataset:
print(data.describe(include='all'))

# Customer_ID     Age  Gender     City  Annual_Income Education_Level  Purchase_Frequency Subscription_Status  Credit_Score  Product_Category  Subscription_Encoded
# 50            45.000000   50     50      50.000000              50               50.000000                  50         45.000000               50             50.000000

# 9. Fill the missing values in 'Age' column using median
print(data['Age'].isna().sum())
# 5
data['Age'].fillna(data['Age'].median(), inplace = True)
print(data['Age'].isna().sum())
# 0

# 10. Replace Missing values in Credit_Score column using the median
print(data['Credit_Score'].isna().sum()) #5
data['Credit_Score'].fillna(data['Credit_Score'].mean(), inplace=True)
print(data['Credit_Score'].isna().sum()) #0

# 11. Missing data will lead to disturbed and incorrect output or prediction of the Model.

'''-------------------------------------------Handle Ordinal Data---------------------------------------------'''

# 12. Define an appropriate mapping for the Education_Level column and transform it into Numerical values.
# i. Check the unique values :
print(data['Education_Level'].unique()) #["Master's" "Bachelor's" 'PhD' 'High School']  
mapping = {
    'High School': 1,
    'Bachelor\'s': 2,
    'Master\'s': 3,
    'PhD': 4
}

data['Education_Level_Mapped'] = data['Education_Level'].map(mapping)
print('Ordinal data handeling: ')
print(data['Education_Level'])

# 13. Verify the transformed values are retaining the correct order (HS < B < M < P)

# 14. Here we have multiple values (i.e: 4), and we want to represent the order also, with the help of weights in the number, 
# whereas O_H_E will represent values only in 0 or 1 that represents only absence or presence of value. 

'''------------------------------------------------------------4. Outlier Detection and Removal --------------------------------'''

# 15. Detect Outlier in the Annual_Income column using IQR Method.

def iqr(col_name):
    q1,q3 = data[col_name].quantile([0.25,0.75])
    IQR = q3 - q1    
    lr = q1 - (1.5 * IQR)
    ur = q3 + (1.5 * IQR)
    return lr, ur

lr, ur = iqr('Annual_Income')

print(data['Annual_Income'].describe())
print(lr)
print(data['Annual_Income'].describe())
print(ur)

# 16. Remove Outliers from the 'Annual_Income' col  based on IQR Column. 
data = data[(data['Annual_Income'] >= lr) & (data['Annual_Income'] <= ur)]
print(data['Annual_Income'].isna().sum())

# 17. Detect outliers in Purchase_Frequency col using z-score method.
# Rule: If score is more than 3 that means there are outliers
print(data['Purchase_Frequency'])

# 18. Remove the outlies from Purchase_Frequency col based on z-score method
data = data[np.abs(stats.zscore(data['Purchase_Frequency'])) <= 3].copy()
print(data)