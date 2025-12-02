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
    testScore = logs['val_accuracy']
    trainScore = logs['accuracy']

    if testScore > trainScore and testScore >= self.CL:
      self.model.stop_training = True

# 1. Check the data quality and perform Preprocessing:
print(data.isna().sum()) # No null values
# Age                0
# EstimatedSalary    0
# Purchased          0

# 2. Splitting data in features and labels:
features = data.iloc[:,[0,1]].values
labels = data.iloc[:,[2]].values

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
s_features = rs.fit_transform(features)

mn = MinMaxScaler()
s_labels = mn.fit_transform(labels)

# Train test split:

X_train, X_test, y_train, y_test = train_test_split(s_features, s_labels, test_size= 0.2, random_state=122)

# Building model:
# 1. Architecting model
# 2. Compile Model
# 3. Fit the Model/ Train
# 4. Evaluate the Model
# 5. Deploy the Model

# 1. Architecting the Model:
model = tf.keras.Sequential()

#    Creating the Input, Hidden and Output layers:
model.add(tf.keras.layers.Dense(units = 60, activation='relu', input_shape = (2,)))

model.add(tf.keras.layers.Dense(units=60, activation='relu'))
model.add(tf.keras.layers.Dense(units=60, activation='relu'))
model.add(tf.keras.layers.Dense(units=60, activation='relu'))
model.add(tf.keras.layers.Dense(units=1, activation='linear'))

# 2. Model compilation:
#For Classification
#
# 1. Binary Classification ---- binary_crossentropy
# 2. MultiClass Classification --- categorical_crossentropy | sparse_categorical_crossentropy

# i. forward propogation
model.compile(optimizer = 'sgd', loss = 'binary_crossentropy', metrics = ['accuracy'])

# i. backward propogation
model.fit(X_train,y_train,validation_data=(X_test,y_test), epochs = 10000000000, callbacks=[MyCLRuleMonitor(0.8)])

# Epoch 131/10000000000
# 10/10 ━━━━━━━━━━━━━━━━━━━━ 0s 11ms/step - accuracy: 0.8406 - loss: 0.3160 - val_accuracy: 0.8500 - val_loss: 0.4860

# Input:
i_age = input('Enter age of person: ')
i_sal = input('Enter Estimated Salary of person: ')

# 1. data must be Numeric
# 2. Data must be in 2d array
# 3. Data must be scaled

input_arr = np.array([[i_age, i_sal]])

scaled_input_features = rs.fit_transform(input_arr)

predicted_value = model.predict(scaled_input_features)
print(predicted_value) # [[0.16993988]]

predicted_class = (predicted_value > 0.5).astype(int)
print(predicted_class) #[[0]]

if predicted_class[0][0] > 0.5:
   print('Good customer')
else:
   print('Bad customer')