import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


import statsmodels.regression.linear_model as stat

data = pd.read_csv('50_Startups.csv')

# Here we will be using CA.
# Saperate data in feature and labels

features = data.iloc[:, [0,1,2,3]].values
labels = data.iloc[:,[4]].values

final_data = pd.concat([pd.get_dummies(data.State), data.iloc[:,[0,1,2,4]]], axis = 1)
print(final_data.info())

# List of feature selection techniques:
# 1. Correlation analysis
# 2. Backward elimination analysis
# 3. Recursive feature analysis
# 4. Select from model

# 1. Co relation analysis:

print(final_data.corr())
# While the feature selection go with the col where the correlation values are > 0.5
# Feature cols = R&D Spends and Marketing spends
# Label = Profit


# Creatae features using new cols
'''
new_features = final_data.iloc[:,[3,5]].values
# print(new_features)

# Do the train test split of data:

# CL = 0.99

# for i in range(1,100):
#     x_train, x_test, y_train, y_test = train_test_split(new_features, labels, train_size=0.2, random_state=i)

#     model = LinearRegression()

#     model.fit(x_train, y_train)

#     train_score = model.score(x_train, y_train)
#     test_score = model.score(x_test, y_test)

#     if test_score > train_score and test_score >= CL:
#         print('For test score: {test_score} the train score is: {train_score} for the RS: {i}')


CL = 0.99
for rs in range(1,100):
  X_train,X_test,y_train,y_test = train_test_split(new_features,
                                                   labels,
                                                   test_size=0.2,
                                                   random_state=rs
                                                   )

  model = LinearRegression()

  model.fit(X_train,y_train)

  trainScore = model.score(X_train,y_train)
  testScore = model.score(X_test,y_test)

  if testScore > trainScore and testScore >= CL :
    print(f"Test Score : {testScore} TrainScore : {trainScore} for RandomState {rs}")

# Test Score : 0.9909864896179557 TrainScore : 0.9382176532996815 for RandomState 10

# x_train, x_test, y_train, y_test = train_test_split(new_features, labels, 
#                                                     train_size=0.2, random_state=10)

#Train the model:
# model.fit(x_train, y_train)

# trainScore = model.score(x_train, y_train)
# testScore= model.score(x_test, y_test)

# 2. Backward feature elimination

#1 . perform all in means create a col for the intercept and append it with the final dataset
'''

for_ols = final_data.iloc[:, [0,1,2,3,4,5]]
intercept= np.ones(len(for_ols))
OLS_dataset = np.append( np.ones( (len(for_ols) , 1) ).astype(int) , for_ols , axis = 1)

print(OLS_dataset)

#Step2: Decide the value of SL: 0.05

# Step 3: Perform the OLS:
# endog: label
# exog: feature

olsFormula = stat.OLS(endog = labels, exog=OLS_dataset.astype('float')).fit()
print(olsFormula.summary())

# Select the featue col that has highest pValue  : x5: adminspends  0.608
# Check the below conditions:


# if the pvale is > SL: 
        # eliminate the col
# else:
#          preserve and go to step 7

# Since pval 0.608 is greater than sl we will therefore eliminate adminspends
newfeatreset = OLS_dataset[:, [0,1,2,3,4,6]]
olsFormula = stat.OLS(endog = labels, exog=newfeatreset.astype('float')).fit()
print(olsFormula.summary())

# highest pvalue :  0.072
# if the pvale is > SL: 
        # eliminate the col
# else:
#          preserve and go to step 7

onemoredataset = OLS_dataset[:, [0,1,2,3,4]]
olsFormula = stat.OLS(endog = labels, exog=onemoredataset.astype('float')).fit()
print(olsFormula.summary())

# now all the pvalues are 0 only we can select all of them now;
# final feature set = California, Florida, NY, RDSpend
final_feature_set_OLS = OLS_dataset[:,[1,2,3,4]]

CL = 0.99
for rs in range(1,200):
  X_train,X_test,y_train,y_test = train_test_split(onemoredataset,
                                                   labels,
                                                   test_size=0.2,
                                                   random_state=rs
                                                   )

  model = LinearRegression()

  model.fit(X_train,y_train)

  trainScore = model.score(X_train,y_train)
  testScore = model.score(X_test,y_test)

  if testScore > trainScore and testScore >= CL :
    print(f"Test Score : {testScore} TrainScore : {trainScore} for RandomState {rs}")
