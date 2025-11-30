# Create a model that can predict the salary of the employee based on his/her years of experience


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf

data = pd.read_csv("Salary_Data.csv")
print(data.head(5))
print(data.info())

# saperate the features and label columns
# print(data.Purchased.value_counts())


''' Rules for classification in ANN:
1. Data must be complete.
2. Data must be strictly numeric
3. Features and lables data must be in the form of np 2D array
4. Data should be Scaled (Features: Optional, Labels: Mandatory):
    a. Binary classification: Label must be represented in the form of 0 and 1.
    b. Multiclass classification: Label must be represented in form of Discreate Numerical or Dummy Variables

'''

features = data.iloc[:,[0]].values
labels = data.iloc[:,[1]].values

# Scale Features:
sscaler = StandardScaler()
transformed_features = sscaler.fit_transform(features)

# Lables Scaling
minMaxLabel = MinMaxScaler()
label = minMaxLabel.fit_transform(labels)

# Train test split:
X_train, X_test, y_train, y_test = train_test_split(transformed_features, labels, test_size=0.2, random_state=1)

'''Deep learning Model creation Steps:

1. Architecting the Model
2. Compile the Model
3. Fit the Model (Training)
4. Check quality of the Model
5. Deploy the Model

'''

# Architecting the model:
model = tf.keras.Sequential()

model.add(tf.keras.layers.Dense(units=100, activation="sigmoid", input_shape=(1,)))
model.add(tf.keras.layers.Dense(units=100, activation='sigmoid'))
model.add(tf.keras.layers.Dense(units=100, activation='sigmoid'))
model.add(tf.keras.layers.Dense(units=1, activation='linear'))

# Compiling the Model:
model.compile(optimizer= 'sgd', loss= 'mean_squared_error', metrics= [tf.keras.metrics.R2Score()])

# Train the model:
print(model.fit(X_train, y_train, validation_data= (X_test, y_test), epochs=6000))

