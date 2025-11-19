import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


data = pd.read_csv("https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv")
print(data.info())
# Features and label saperation:

features = data.iloc[:,[0,1,2,3]].values
labels = data.iloc[:,[4]].values

# Perform Feature reduction using PCA (Principal Component Analysis)
 
# 1. Scale the features (Standardization is mandatory) (StandardScaler -if data is- Normally Dist) (RobustScaler -if data is- Skewed)

scaler = StandardScaler()
scaled_featues = scaler.fit_transform(features)

# 2. Create and Train PCA model
pca = PCA(n_components= 2)
pca.fit(scaled_featues)
final_features = pca.fit_transform(scaled_featues)

# 3. Model training
from sklearn.linear_model import LogisticRegression

X_train, X_test, y_train, y_test = train_test_split(final_features, labels, test_size=0.2, random_state=1)
model = LogisticRegression()

model.fit(X_train, y_train)

test_score = model.score(X_test, y_test)
train_score = model.score(X_train, y_train)

print(f'Train score: {train_score}, Test score: {test_score}')
# Train score: 0.9, Test score: 0.9666666666666667

# Deployment:
sepal_length = float(input('Enter sepal length: '))
sepal_width = float(input('Enter sepal width: '))
petal_length = float(input('Enter petal length: '))
petal_width = float(input('Enter petal width: '))

# create feature:
featureSet = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

# apply sc object:
stdfeatures = scaler.transform(featureSet)
# apply PCA:3.
transformedfeatues = pca.transform(stdfeatures)

print(model.predict(transformedfeatues))


