# Create a model that can classify the iris flower species based on flowers' properties

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

data = pd.read_csv("https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv")
class MyCLRuleMonitor(tf.keras.callbacks.Callback):
  def __init__(self,CL):
    super(MyCLRuleMonitor,self).__init__()
    self.CL = CL


  def on_epoch_end(self,epoch,logs=None):
    testScore = logs['val_accuracy']
    trainScore = logs['accuracy']

    if testScore > trainScore and testScore >= self.CL:
      self.model.stop_training = True
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

cols = ['sepal.length', 'sepal.width','petal.length', 'petal.width' ]

for col in cols:
    get_scaler(col)
    # All the feature cols are Normally Distributed.

'''For data to be processed by Deep learning model:
1. Data must be complete.
2. Data needs to be strictly Numeric
3. Features and Labels must be in form of 2d array.
4. Data needs to be Scaled: Features (Optional,  but good to do) / Labels: Mandatory
    i. Binary-classification: labels must be represented in form of 0 or 1.
    ii. Multiclass-classification: lebels must be discrete numerical or Dymmy Variables '''

# Labels must be Numeric : Using LabelEncoder
le = LabelEncoder()
s_labels= le.fit_transform(label)
# print(s_labels)

sc = StandardScaler()
s_features = sc.fit_transform(features)
# print(s_features)

# Train test split 
X_train, X_test, y_train, y_test = train_test_split(s_features, s_labels, test_size=0.2, random_state = 232)

'''Deep learning Model Building steps:
1. Architect the model
2. Compile the model
3. Train the model
4. Evaluate the model
5. Deployment and User testing
'''

import tensorflow as tf

# 1. Model architecting:
model = tf.keras.Sequential()

model.add(tf.keras.layers.Dense(units = 100, activation= 'sigmoid', input_shape=(4,)))
model.add(tf.keras.layers.Dense(units = 100, activation= 'sigmoid'))
model.add(tf.keras.layers.Dense(units = 100, activation= 'sigmoid'))
model.add(tf.keras.layers.Dense(units = 100, activation= 'sigmoid'))
model.add(tf.keras.layers.Dense(units=1, activation='linear'))

# 2. Compile the model: Forward propogation:
model.compile(optimizer = 'sgd', loss= 'binart_', metics = ['accuracy'])

#3. train the model:
model.fit(epochs = 10000, callbacks= )