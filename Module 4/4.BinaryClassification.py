# Creating a model that can Predict if a customer is a good or Bad, based on if he is going to make a Purchase or not. 
# This prediction will be made on the basis of Age and Salary of an individual

import pandas as pd
import numpy as np
from scipy.stats import shapiro
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf

data = pd.read_csv('Social_Network_Ads.csv')
print(data.head(5))

# Class to control the flow/limit of execution.
class MyCLRuleMonitor(tf.keras.callbacks.Callback):
  def __init__(self,CL):
    super(MyCLRuleMonitor,self).__init__()
    self.CL = CL


  def on_epoch_end(self,epoch,logs=None):
    testScore = logs['val_r2_score']
    trainScore = logs['r2_score']

    if testScore > trainScore and testScore >= self.CL:
      self.model.stop_training = True

# 1. Check the data quality and perform Preprocessing:
print(data.isna().sum()) # No null values
# Age                0
# EstimatedSalary    0
# Purchased          0

# 2. Splitting data in features and labels:
features = data.iloc[:,[0,1]].iloc
labels = data.iloc[:,[2]].iloc

# 3. Check if data is Normally distributed or not:
cols = ['Age', 'EstimatedSalary', 'Purchased']

def get_scaler(cols):
    threshold = 0.05
    for col in cols:
        print(col)
        curr, p_val = shapiro(data[col])

        if p_val >= threshold:
            print(f'{col} Data is normally distributed')
            return StandardScaler()
        else:
            print(f'{col} Data is not normally distributed')
            return RobustScaler()

rs = RobustScaler()
s_features = rs.fit_transform(features[:, [0,1]])

mn = MinMaxScaler()
s_labels = mn.fit_transform(labels[:, [2]])

# Train test split:

X_train, X_test, y_train, y_test = train_test_split(s_features, s_labels, test_size= 0.2, random_state=122)

# Building model:
# 1. Architecting model
# 2. Compile Model
# 3. Fit the Model/ Train
# 4. Evaluate the Model
# 5. Deploy the Model

# 1. Architecting the Model:
model = tf.keras.Sequeltial()

#    Creating the Input, Hidden and Output layers:
model.add(tf.keras.layers.Dense(units = 100, activation='sgd', input_shape = (2,)))

model.add(tf.keras.layers.Dense(units=100, activation='sgd'))
model.add(tf.keras.layers.Dense(units=100, activation='sgd'))
model.add(tf.keras.layers.Dense(units=100, activation='sgd'))
model.add(tf.keras.layers.Dense(units=1, activation='linear'))

# 2. Model compilation:

model.compile(optimizer = 'adam', )