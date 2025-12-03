# Create a model that can classify the iris flower species based on flowers' properties

import pandas as pd
import numpy as np

data = pd.read_csv("https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv")

print(data.info())

#  #   Column        Non-Null Count  Dtype
# ---  ------        --------------  -----
#  0   sepal.length  150 non-null    float64
#  1   sepal.width   150 non-null    float64
#  2   petal.length  150 non-null    float64
#  3   petal.width   150 non-null    float64
#  4   variety       150 non-null    object

# 1. Check for null values
print(data.isna().sum())
# No null found

# 2. Seprate the feature and label cols

features = data.iloc[:,[0,1,2,3]].values
label = data.iloc[:,[4]].values

# Normalize the feature and label cols:
threshold = 0.5

from scipy.stats import shapiro

def get_scaler(col):
    p_val, curr = shapiro(data[col])
    if p_val > threshold:
        print(f'{col} is Normally distributed..')
    else:
        print(f'{col} is not Normally distributed..')

cols = ['sepal.length', 'sepal.width','petal.length', 'petal.width' ,'variety']

for col in cols:
    get_scaler(col)
