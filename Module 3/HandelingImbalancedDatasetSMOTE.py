import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE


data = pd.read_csv('Social_Network_Ads.csv')
print(data.info())

# Check the label col, to check if dataset is balanced or imbalanced:
print(data.Purchased.value_counts())

# Purchased
# 0    257
# 1    143
# Data Is imbalanced, so we need to balance this data. 

features = data.iloc[:, [0,1]]
label = data.iloc[:, [2]]

smoteobj = SMOTE(random_state=1)
sampled_features, sampled_labeles = smoteobj.fit_resample(features, label)

resampledDF = pd.concat([sampled_features,sampled_labeles] , axis =1 )


print(resampledDF.Purchased.value_counts())
# Purchased
# 0    257
# 1    257
