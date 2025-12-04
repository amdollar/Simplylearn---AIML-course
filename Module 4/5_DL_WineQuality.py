import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler

data = pd.read_csv('winequality-red.csv')

# print(data.info())
# print(data.head(2))

# Quality is the label here: 
print(data.quality.unique())
# [5 6 7 4 8 3]

# Preprocssing:
# Null values check:
print(data.isna().sum()) # No null values

# Drop null cols:
data.dropna(inplace=True)

# saperate label and feature cols:
features = data.iloc[:,0:10].values
labels = data.iloc[:, 11].values

print(features)
print(labels)

' Rules for deep learning models:'
'1. Data must be strictly numerical'
'2. Data must be represented in form of 2D array'
'3. Feature colms are better if Scaled'
'4. Label col must be Scaled: '
'           Binary classification ---> Discrete numeric in form of 0 and 1 '
'           Multiclass classification ---> Dummy variables or Discrete numeric'


def check_normality(col):
    