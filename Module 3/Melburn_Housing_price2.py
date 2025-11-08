# Create a model that can predict the price of the house using all feature cols (SL for the project is 0.5)
import pandas as pd
import numpy as np
pd.options.mode.chained_assignment = None
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder

data = pd.read_csv('melb_data.csv')

# print(data.info())
# before using all the data we need to make sure:
# 1. Data is complete
# 2. Data is Strictly numeric

#  0   Suburb         13580 non-null  object ---> Categorical (Mode)
#  1   Address        13580 non-null  object  ---> Categorical (Mode)
#  2   Rooms          13580 non-null  int64 ----> Numeric:Discrete (Median)
#  3   Type           13580 non-null  object ----> Categorical (Mode)
#  4   Price          13580 non-null  float64 ----> Numeric: Continuous (Mean)
#  5   Method         13580 non-null  object ----->  Categorical (Mode)
#  6   SellerG        13580 non-null  object ----> Categorical (Mode)
#  7   Date           13580 non-null  object ----> Categorical (Mode)
#  8   Distance       13580 non-null  float64   ----> Numeric: Continuous (Mean)
#  9   Postcode       13580 non-null  float64  ----> Numeric:Discrete (Median)
#  3   Type           13580 non-null  object ----> Categorical (Mode)
#  10  Bedroom2       13580 non-null  float64 ----> Numeric:Discrete (Median)
#  11  Bathroom       13580 non-null  float64 ----> Numeric:Discrete (Median)
#  12  Car            13518 non-null  float64 ----> Numeric:Discrete (Median)
#  13  Landsize       13580 non-null  float64 ----> Numeric: Continuous (Mean)
#  14  BuildingArea   7130 non-null   float64 ----> Numeric: Continuous (Mean)
#  15  YearBuilt      8205 non-null   float64 ----> Numeric:Discrete (Median)
#  16  CouncilArea    12211 non-null  object  ----> Numeric: Continuous (Mean)
#  17  Lattitude      13580 non-null  float64  ----> Numeric: Continuous (Mean)
#  18  Longtitude     13580 non-null  float64  ----> Numeric: Continuous (Mean)
#  19  Regionname     13580 non-null  object ----> Categorical (Mode)
#  20  Propertycount  13580 non-null  float64 ----> Numeric:Discrete (Median)

# method is to handle the null discrete data for the provided col:


# Drop the duplicate columns : Type
data.drop_duplicates()
def handleDiscreteData(colname):
    print(f'Total number of NAN col in {colname} : {data[colname].isna().sum()}')
    data[colname].fillna(data[colname].median(), inplace= True)


discreteDataCols = ['Rooms', 'Postcode','Bedroom2', 'Bathroom','Car', 'YearBuilt', 'Propertycount']
for i in (discreteDataCols):
    handleDiscreteData(i)
    print(f'After Processing Total number of NAN col in {i} : {data[i].isna().sum()}')


# Method to hande the nan continuous data for the provided col:
def handleContinuous(colname):
    print(f'Total number of NAN col in {colname} : {data[colname].isna().sum()}')
    data[colname].fillna(data[colname].mean(), inplace= True)


contDataCols = ['Distance','Landsize','Landsize','BuildingArea','Lattitude','Longtitude']
for i in (contDataCols):
    handleContinuous(i)
    print(f'After Processing Total number of NAN col in {i} : {data[i].isna().sum()}')

# Method to hande the nan Categorical data for the provided col:
def handleCategorical(colname):
    print(f'Total number of NAN col in {colname} : {data[colname].isna().sum()}')
    data[colname].fillna(data[colname].mode()[0], inplace= True)


categoricalDataCols = ['Suburb','Address','Type','Method','SellerG','Date','Type','CouncilArea','Regionname']
for i in (categoricalDataCols):
    handleCategorical(i)
    print(f'After Processing Total number of NAN col in {i} : {data[i].isna().sum()}')

print(data['Car'].isna().sum())

# all the NA data is handeled now, now we need to convert the categorical data into numerical and then append in final data:
# Following are some categorical cols: we can Use OHE and use:
# Avoiding 0th col and 6th col
print(data['Address'].unique())
print(data['Type'].unique())
print(data['Method'].unique())
print(data['CouncilArea'].unique())
print(data['Regionname'].unique())

feature = data.iloc[:,:].values
print(feature)

encoder = OneHotEncoder(sparse=False)
encoder.fit_transform(feature[:,[1,3,5,16,19]])
values = encoder.transform(feature[:,[1,3,5,16,19]])

# Concat the feature values with the encoded values
final_feature = np.concatenate((values, feature[:,[2,8,9,10,11,12,13,14,15,17,18,20]]), axis = 1)
# print(final_feature)
label = data.iloc[:,[4]].values


'''
This divides the data into training and testing set:
here x_train will have training dataset values from features, and y_train will have training dataset values from labels
and x_test, y_test will have testing feature and label dataset values
#  x_train, x_test, y_train, y_test = train_test_split(final_feature, label,
#                                                     train_size=0.2,
#                                                     random_state=4)
'''
# CL = 0.5
# x_train, x_test, y_train, y_test = train_test_split(final_feature, label,
#                                                     train_size=0.2,
#                                                     random_state=4)

# model = LinearRegression()
# model.test(x_test,y_test)

# trainScore = model.score(x_train, y_train)
# testScore = model.score(x_test, y_test)

# model.predict()

CL = 0.50
for rs in range(1,101):
  X_train,X_test,y_train,y_test = train_test_split(final_feature,
                                                   label,
                                                   test_size=0.2,
                                                   random_state=rs
                                                   )

  model = LinearRegression()

  model.fit(X_train,y_train)

  trainScore = model.score(X_train,y_train)
  testScore = model.score(X_test,y_test)

  if testScore > trainScore and testScore >= CL :
    print(f"Test Score : {testScore} TrainScore : {trainScore} for RandomState {rs}")