'''Trying the regularization technique : Drop Out'''

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from scipy.stats import shapiro
from sklearn.model_selection import train_test_split
import tensorflow as tf


class MyCLRuleMonitor(tf.keras.callbacks.Callback):
  def __init__(self,CL):
    super(MyCLRuleMonitor,self).__init__()
    self.CL = CL


  def on_epoch_end(self,epoch,logs=None):
    testScore = logs['val_accuracy']
    trainScore = logs['accuracy']

    if testScore > trainScore and testScore >= self.CL:
      self.model.stop_training = True


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
features = data.iloc[:,[0,1,2,3,4,5,6,7,8,9,10]].values
labels = data.iloc[:, [11]].values

# print(features)
# print(labels)

' Rules for deep learning models:'
'1. Data must be strictly numerical'
'2. Data must be represented in form of 2D array'
'3. Feature colms are better if Scaled'
'4. Label col must be Scaled: '
'           Binary classification ---> Discrete numeric in form of 0 and 1 '
'           Multiclass classification ---> Dummy variables or Discrete numeric'


sc = StandardScaler()
en_features = sc.fit_transform(features)


#Label's Rule for Classification
# Discrete labels must ordered whole number
# e.g. if your original label is [5,6,2,3], this should be transformed to ordered whole number which is achieved using LabelEncoder class.

# #Label Encoding ---> Converting label from categorical to discrete
le = LabelEncoder()
en_labels = le.fit_transform(labels)
# print(en_labels)

#Convert discrete label to dummy values
# ohe = tf.keras.utils.to_categorical(en_labels)
# print(ohe)

print(np.unique(labels))



X_train, X_test, y_train, y_test = train_test_split(en_features, en_labels, test_size=0.2, random_state=102)

'Deep learning model steps:'
'1. Model architecting'
'2. Model compile'
'3. Model training'
'4. Model evaluation'
'5. Model input'

model = tf.keras.Sequential()


model.add(tf.keras.layers.Dense(units= 100, activation='relu', input_shape= (11,)))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(units=100, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(units = 100, activation= 'relu'))
model.add(tf.keras.layers.Dropout(0.4))
model.add(tf.keras.layers.Dense(units= 1, activation='linear'))

model.compile(optimizer=tf.keras.optimizers.RMSprop(momentum=0.2), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, validation_data = (X_test, y_test), epochs= 10000, callbacks= [MyCLRuleMonitor(0.65)])


# features = data.iloc[:,[0,1,2,3,4,5,6,7,8,9,10]].values
# #Feature Scaling
# sc = StandardScaler()
# features = sc.fit_transform(features)

# label = data.iloc[:,[11]].values

# le = LabelEncoder()
# label = le.fit_transform(label)


# X_train,X_test,y_train,y_test = train_test_split(features,
#                                                  label,
#                                                  test_size=0.2,
#                                                  random_state=10)
# 'Deep learning model steps:'
# '1. Model architecting'
# '2. Model compile'
# '3. Model training'
# '4. Model evaluation'
# '5. Model input'


# model1 = tf.keras.Sequential()
# model1.add(tf.keras.layers.Dense(units=122, activation="relu", input_shape=(11,)))
# model1.add(tf.keras.layers.Dense(units=122, activation="relu"))
# model1.add(tf.keras.layers.Dense(units=122, activation="relu"))
# model1.add(tf.keras.layers.Dense(units=6, activation="softmax"))


# # model1.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
# model1.compile(optimizer= tf.keras.optimizers.Adam(learning_rate= 0.2), loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# # model1.fit(X_train, y_train, validation_data = (X_test, y_test), epochs = 100000, callbacks= [MyCLRuleMonitor(0.8)])

# model1.fit(X_train,y_train, validation_data=(X_test,y_test), epochs=100000, callbacks = [MyCLRuleMonitor(0.65)])