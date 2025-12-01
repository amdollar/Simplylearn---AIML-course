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

data_cols = ['R&D Spend','Administration','Marketing Spend']
for col in range(data_cols.shape):
   check_normality(col)
   

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

model.add(tf.keras.layers.Dense(units = 15, activation='relu', input_shape=(6,)))
# Handle the no of feature cols and activation function as per rules.
model.add(tf.keras.layers.Dense(units= 15, activation='relu'))
model.add(tf.keras.layers.Dense(units= 1, activation='linear'))

# 2. Model compile:
model.compile(optimizer = 'adam',loss = 'mean_squared_error' ,metrics = [tf.keras.metrics.R2Score()])


# 3. Fit the model/ Train
model.fit(X_train, y_train, validation_data = (X_test, y_test), epochs= 100000 ,callbacks=[MyCLRuleMonitor(.85)])
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

