import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression

data = pd.read_csv('50_Startups.csv')

print(data.info())

features = data.iloc[:,[0,1,2,3]].values
labels = data.iloc[:,[4]].values

encoder = OneHotEncoder(sparse_output=False)
feat = encoder.fit(features[:,[3]])
encoded_state = encoder.fit_transform(features[:,[3]])

final_features = np.concatenate((encoded_state, features[:,[0,1,2]]), axis=1)

print(final_features)
model = LinearRegression()
X_train, X_test, y_train, y_test = train_test_split(final_features, labels, test_size=0.2, random_state=2)

model.fit(X_train, y_train)

valid_state_values = encoder.categories_[0]

# Get input for the features
rdSpend = float(input("Enter R&D Spend: "))
administration = float(input("Enter Administration: "))
marketingSpend = float(input("Enter Marketing Spend: "))
state = (input("Enter state: "))

# validate the entered city:
while state not in valid_state_values:
    print('Invalid state value provided.')
    state = (input("Enter state: "))

print(state)    

state_i = encoder.transform(np.array([[state]]))
input_features = np.concatenate((state_i, np.array([[rdSpend, administration, marketingSpend]])), axis = 1)
print(input_features)
print(model.predict(input_features))
# [[52202.28961586]]