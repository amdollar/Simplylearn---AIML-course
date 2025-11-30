# NOTE: Setup the Python workspace before running:
# 1) From this folder run: create_venv.bat
# 2) Activate the created virtualenv (.venv\Scripts\activate) and then run this script.
# 3) If using PowerShell run: .\create_venv.ps1

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf

data = pd.read_csv("Social_Network_Ads.csv")
print(data.head(5))
print(data.info())

# saperate the features and label columns
print(data.Purchased.value_counts())


''' Rules for classification in ANN:
1. Data must be complete.
2. Data must be strictly numeric
3. Features and lables data must be in the form of np 2D array
4. Data should be Scaled (Features: Optional, Labels: Mandatory):
    a. Binary classification: Label must be represented in the form of 0 and 1.
    b. Multiclass classification: Label must be represented in form of Discreate Numerical or Dummy Variables

'''

features = data.iloc[:,[0,1]].values
labels = data.iloc[:,[2]].values

# Scale Features:
sscaler = StandardScaler()
transformed_features = sscaler.fit_transform(features)

# Lables are already 0 and 1

# Train test split:
X_train, X_test, y_train, y_test = train_test_split(transformed_features, labels, test_size=0.2, random_state=1)

'''Deep learning Model creation Steps:

1. Architecting the Model
2. Compile the Model
3. Fit the Model (Training)
4. Check quality of the Model
5. Deploy the Model

'''

