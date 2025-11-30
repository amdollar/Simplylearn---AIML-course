import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf

data = pd.read_csv("Salary_Data.csv")
print(data.head(5))
data.dropna(inplace=True)
print(data.info())

'''
Rules for ANN:
1. Data must be complete
2. Data must be numerical
3. Data must be in 2D Numerical array
4. For regression use cases:
    a. Features must be NORMALIZED/STANDARDIZED using StandardScaler (if features are normally distributed) | RobustScaler (if all/any feature col are skew distributed)
    b. Label cols must be Scaled using MinMaxScaler with range 0 and 1 
'''

# Feature and Label split
features = data.iloc[:,[0]].values
labels = data.iloc[:,[1]].values

# 1. Normalization of features and labels
sc = StandardScaler()
normalized_featues = sc.fit_transform(features)

mm = MinMaxScaler()
normalized_labels= mm.fit_transform(labels)

# Train test split:
X_train, X_test, y_train, y_test = train_test_split(normalized_featues, normalized_labels, test_size=0.2, random_state=10)


# DL Model Creation Steps
'''
1. Model architecting
2. Model compile
3. Model training/fitting
4. Model evaluation
5. Model Deployment
'''
# 1. Architect the model

model = tf.keras.Sequential()

#Create InputLayer and First Hidden Layer
model.add(tf.keras.layers.Dense(units = 100, activation='relu', input_shape=(1,)))
#Create Second Hidden Layer
model.add(tf.keras.layers.Dense(units = 100, activation='relu'))
#Create Third Hidden Layer
model.add(tf.keras.layers.Dense(units = 100, activation='relu'))
#Create Output Layer
model.add(tf.keras.layers.Dense(units = 1, activation='linear'))

# 2. Compile
model.compile(optimizer="sgd", #declaring with back prop algo to use
              loss = "mean_squared_error", #declarying which error function to use
              metrics = [tf.keras.metrics.R2Score()]) #which metric to use to evaluate model

class MyCLRuleMonitor(tf.keras.callbacks.Callback):
  def __init__(self,CL):
    super(MyCLRuleMonitor,self).__init__()
    self.CL = CL


  def on_epoch_end(self,epoch,logs=None):
    testScore = logs['val_r2_score']
    trainScore = logs['r2_score']

    if testScore > trainScore and testScore >= self.CL:
      self.model.stop_training = True



# 3. Fit data to the model (Training | Model Convergence)
model.fit(X_train,y_train,
          validation_data=(X_test,y_test),
          epochs=5000000000000000000000,
          callbacks = [MyCLRuleMonitor(CL=0.8)])

# Epoch 419/5000000000000000000000
# 1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 96ms/step - loss: 0.0034 - r2_score: 0.9668 - val_loss: 0.0025 - val_r2_score: 0.9669

model.evaluate(X_train,y_train)

# 4. Save the model
model.save('SalaryDataPredictor.keras') # New Way
model.save('SalaryDataPredictor.h5') # Old way

# 4. Deployment:

#UserInput
yearsExperience = float(input("Enter years of experience: "))

#Converted it to np array 2d
yExpArray = np.array([[yearsExperience]])

#Scaled data
yExpScaled = sc.transform(yExpArray)

#Got prediction
salaryPredicted = model.predict(yExpScaled)

#Inverse transform label value

finalSalary = mm.inverse_transform(salaryPredicted)

print(f"Salary for {yearsExperience} is $ {finalSalary}")