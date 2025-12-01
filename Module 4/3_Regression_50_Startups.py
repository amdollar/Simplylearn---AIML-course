# Create a model that can predict the profit of the company based on company's
# spending pattern and company's location
#
# SL = 0.15
#
# 50_Startups.csv (dataset)
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf

class MyCLRuleMonitor(tf.keras.callbacks.Callback):
  def __init__(self,CL):
    super(MyCLRuleMonitor,self).__init__()
    self.CL = CL


  def on_epoch_end(self,epoch,logs=None):
    testScore = logs['val_r2_score']
    trainScore = logs['r2_score']

    if testScore > trainScore and testScore >= self.CL:
      self.model.stop_training = True


from scipy.stats import shapiro

def check_normality(name):
  SL = 0.05

  curr, pvalue = shapiro(data[name])

  if pvalue >= SL:
      print('H1: the column is Normally distributed')
  else:
      print('H0: The column is not normally distributed.')

data = pd.read_csv('50_Startups.csv')
# print(data.head())
print(data.info())
data.dropna(inplace=True)
print(data.info())

# 1. Features and Labels split
features = data.iloc[:, [0,1,2,3]].values
labels = data.iloc[:, [4]].values

# 2. Scale the features and lables
# a. Features need to be scaled using standard scaler, 
# Transform the State values using One Hot Encoder concat it with the featues and then process

ohe = OneHotEncoder(sparse_output=False)
ohe.fit(features[:,[3]])
states = ohe.fit_transform(features[:,[3]])
print(states)
final_featues = np.concatenate((states, features[:,[0,1,2]]),axis=1)
print(final_featues)

# sc = StandardScaler()
# s_features = sc.fit_transform(final_featues)
# print(s_features)

# data_cols = ['R&D Spend','Administration','Marketing Spend']
# for col in range(data_cols.shape):
#    check_normality(col)
   

rs = RobustScaler()
s_features = rs.fit_transform(final_featues)

# b. Labels need to be scaled using Min Max Scaler
mn = MinMaxScaler()
m_labels = mn.fit_transform(labels)

# 3. Train test split
X_train, X_test, y_train, y_test = train_test_split(s_features, m_labels, test_size=0.2, random_state = 10)

# Deep learning model steps:
# 1. Model architeching (Creating Input/Hidden and Output layers)
# 2. Model compile
# 3. Model fit (Training)
# 4. Model evaluation
# 5. Model deployment

# https://www.tensorflow.org/api_docs/python/tf/keras/Model
# Model architecting:


model = tf.keras.Sequential()

model.add(tf.keras.layers.Dense(units = 100, activation='relu', input_shape=(6,)))
# Handle the no of feature cols and activation function as per rules.
model.add(tf.keras.layers.Dense(units= 100, activation='relu'))
model.add(tf.keras.layers.Dense(units= 100, activation='relu'))
model.add(tf.keras.layers.Dense(units= 1, activation='linear'))

# 2. Model compile:
model.compile(optimizer = 'adam',loss = 'mean_squared_error' ,metrics = [tf.keras.metrics.R2Score()])


# 3. Fit the model/ Train
model.fit(X_train, y_train, validation_data = (X_test, y_test), epochs= 100000 ,callbacks=[MyCLRuleMonitor(.85)])



# Testing app:
rdspends = float(input('Enter the r&d spends: '))
adminspends = float(input('Enter the admin spends: '))
marketingspends = float(input('Enter the marketing spends: '))
state = input('Enter the location: ')

# Encode the state value:
state_values = ohe.categories_[0]
while state not in state_values:
   state = input('Enter the location: ')

# Transform this value:
input_state = ohe.transform(np.array([[state]]))
input_final_features = np.concatenate((input_state, np.array([[rdspends, adminspends, marketingspends]])), axis = 1)

# Scale the features:
scaled_input = rs.fit_transform(input_final_features)
predicted_values = model.predict(scaled_input)

# reverse transform the predicted value:
profit  = mn.inverse_transform(predicted_values)

print(f'For values: {rdspends},{adminspends}, {marketingspends}, and {state}: Expected profit is: {profit}')
# 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 102ms/step
# For values: 1233.0,1232.0, 4433.0, and Florida: Expected profit is: [[31891.148]]
   

# The problem was with the concat operation when I did not concat the value of 3rd col in the feature dataset

# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 76ms/step - loss: 0.1263 - r2_score: 0.8703 - val_loss: 0.0402 - val_r2_score: 0.8732


# Learning: 
# 1. With the sigmoid and 100 unit layers the model is giving random r2 and val_r2_scores 
# 2. The data  is not giving Generalized model, and it seems the data is not Normalized, so we will use Robust scaler instead of Standard, and see the results
 

  

# With 3 layers of 15 units and SGD:

  
# Epoch 2231/100000
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 92ms/step - loss: 0.0013 - r2_score: 0.9736 - val_loss: 0.0038 - val_r2_score: 0.9333
# Epoch 2232/100000
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 181ms/step - loss: 0.0013 - r2_score: 0.9736 - val_loss: 0.0039 - val_r2_score: 0.9322
# Epoch 2233/100000
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 100ms/step - loss: 0.0013 - r2_score: 0.9738 - val_loss: 0.0039 - val_r2_score: 0.9320
# Epoch 2234/100000
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 1

# No results:2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 112ms/step - loss: 6.7277e-04 - r2_score: 0.9862 - val_loss: 0.0045 - val_r2_score: 0.9218
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 95ms/step - loss: 6.7203e-04 - r2_score: 0.9862 - val_loss: 0.0045 - val_r2_score: 0.9219
# Epoch 5028/100000
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 107ms/step - loss: 6.7189e-04 - r2_score: 0.9862 - val_loss: 0.0045 - val_r2_score: 0.9215


# # 3 layers of 15 units and Adam and Robust scaler: Increasing the layer and units now
# /2 ━━━━━━━━━━━━━━━━━━━━ 0s 131ms/step - loss: 1.9212e-08 - r2_score: 1.0000 - val_loss: 0.0085 - val_r2_score: 0.8516
# Epoch 3239/100000
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 107ms/step - loss: 2.2403e-08 - r2_score: 1.0000 - val_loss: 0.0085 - val_r2_score: 0.8517
# Epoch 3240/100000
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 104ms/step - loss: 1.9474e-08 - r2_score: 1.0000 - val_loss: 0.0085 - val_r2_score: 0.8517
# Epoch 3241/100000
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 110ms/step - loss: 1.3799e-08 - r2_score: 1.0000 - val_loss: 0.0085 - val_r2_score: 0.8517
# Epoch 3242/100000
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 107ms/step - loss: 2.1551e-08 - r2_score: 1.0000 - val_loss: 0.0085 - val_r2_score: 0.8516
# Epoch 3243/100000
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 106ms/step - loss: 1.8879e-08 - r2_score: 1.0000 - val_loss: 0.0085 - val_r2_score: 0.8515
# Epoch 3244/100000

'''Success case: 1 Adam with 100 unit'''
# with adam and robust scaler with increased hidden layers with the Units: Trying again with sgd with same configs
# Epoch 10/100000
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 105ms/step - loss: 0.0189 - r2_score: 0.6124 - val_loss: 0.0177 - val_r2_score: 0.6922
# Epoch 11/100000
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 117ms/step - loss: 0.0124 - r2_score: 0.7448 - val_loss: 0.0098 - val_r2_score: 0.8297
# Epoch 12/100000
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 167ms/step - loss: 0.0100 - r2_score: 0.7945 - val_loss: 0.0073 - val_r2_score: 0.8737

'''Success case: 2 Adam with feature*3 = 18 unit'''
# Epoch 43/100000
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 113ms/step - loss: 0.0158 - r2_score: 0.6751 - val_loss: 0.0094 - val_r2_score: 0.8361
# Epoch 44/100000
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 112ms/step - loss: 0.0151 - r2_score: 0.6902 - val_loss: 0.0085 - val_r2_score: 0.8523

# Not giving any generalized model with SGD even with same configs:
# Epoch 4689/100000
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 108ms/step - loss: 9.5705e-05 - r2_score: 0.9980 - val_loss: 0.0037 - val_r2_score: 0.9349
# Epoch 4690/100000
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 112ms/step - loss: 9.5631e-05 - r2_score: 0.9980 - val_loss: 0.0037 - val_r2_score: 0.9349
# Epoch 4691/100000
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 107ms/step - loss: 9.6250e-05 - r2_score: 0.9980 - val_loss: 0.0038 - val_r2_score: 0.9346
# Epoch 4692/100000

'''Success results: 3 with Adamax Units = 100'''
# #Epoch 12/100000
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 104ms/step - loss: 0.0150 - r2_score: 0.6911 - val_loss: 0.0095 - val_r2_score: 0.8344
# Epoch 13/100000
# 2/2 ━━━━━━━━━━━━━━━━━━━━ 0s 104ms/step - loss: 0.0119 - r2_score: 0.7558 - val_loss: 0.0072 - val_r2_score: 0.8752

'''Success results: 3 with Adamax Units = 18'''

# Not getting generalized model even after 3k iterations