import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

data = pd.read_csv('HR_comma_sep.csv')
# print(data.info())


#  0   satisfaction_level     14999 non-null  float64
#  1   last_evaluation        14999 non-null  float64
#  2   number_project         14999 non-null  int64  
#  3   average_montly_hours   14999 non-null  int64  
#  4   time_spend_company     14999 non-null  int64  
#  5   Work_accident          14999 non-null  int64
#  6   left                   14999 non-null  int64
#  7   promotion_last_5years  14999 non-null  int64
#  8   sales                  14999 non-null  object
#  9   salary                 14999 non-null  object
# number_project, average_montly_hours, time_spend_company, promotion_last_5years, salary

def data_quality_check(data):
    # Get all the NAN values:
    print(data.info())
    # I can see all the data cols have 14999 values. SO there is no nan or null values


# 1.	Perform data quality checks by checking for missing values, if any.
data_quality_check(data)


# 2.	Understand what factors contributed most to employee turnover at EDA.
features = data.iloc[:,[2,3,4,7,9]].values
label = data.iloc[:,[0]]
# print(features)
# print(label)
# salary is a categorical values, we need to check the all available value do EDA and change in numbers

# print(data['salary'].unique()) 

# ['low' 'medium' 'high'] since we have unique values here only, we can create a feature col by performing OTE

salary_encoder = OneHotEncoder(sparse_output=False)
salaries = salary_encoder.fit_transform(features[:, [4]])

final_featureset = np.concatenate(([salaries, features[:,[0,1,2,3]]]), axis=1)
print(final_featureset)
print(data['turnover'])