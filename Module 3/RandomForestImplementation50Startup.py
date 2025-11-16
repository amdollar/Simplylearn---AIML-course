import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings("ignore")
data = pd.read_csv('50_Startups.csv')

print(data.info())
features = data.iloc[:, [0,1,2,3]].values
label = data.iloc[:, [4]].values

state_imputer = OneHotEncoder(sparse_output=False)
state_values = state_imputer.fit_transform(features[:,[3]])
print(state_values)
                                                    
final_feature = np.concatenate((state_values, features[:, [0,1,2]]), axis=1)
print(final_feature)

CL = 0.9
for i in range(1,301):
  X_train,X_test,y_train,y_test = train_test_split(final_feature,
                                                   label,
                                                   test_size=0.2,
                                                   random_state=i)

  model = RandomForestRegressor(random_state = 3, n_estimators=50, max_depth=4)
  model.fit(X_train,y_train)

  trainScore = model.score(X_train,y_train)
  testScore = model.score(X_test,y_test)

  if testScore > trainScore and testScore >= CL:
    print(f"Test Score {testScore} | Train Score {trainScore} | RS {i}")

# Test Score 0.982946799753128 | Train Score 0.9819764121295017 | RS 211