# Creating a model that can Predict if a customer is a good or Bad, based on if he is going to make a Purchase or not. 
# This prediction will be made on the basis of Age and Salary of an individual

import pandas as pd
import numpy as np


data = pd.read_csv('Social_Network_Ads.csv')
print(data.head(5))
print(data.info())