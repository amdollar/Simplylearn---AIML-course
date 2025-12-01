# Create a model that can predict the profit of the company based on company's
# spending pattern and company's location
#
# SL = 0.15
#
# 50_Startups.csv (dataset)
import pandas as pd
import numpy as np


data = pd.read_csv('50_Startups.csv')
print(data.head())