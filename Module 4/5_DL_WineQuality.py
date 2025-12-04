import pandas as pd
import numpy as np

data = pd.read_csv('winequality-red.csv')

print(data.info())
print(data.head(2))