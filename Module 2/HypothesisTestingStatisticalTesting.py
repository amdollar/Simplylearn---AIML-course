import pandas as pd
import numpy as np
import seaborn as sns
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

data = pd.read_csv('50_Startups.csv')

print(data.info())
#  #   Column           Non-Null Count  Dtype  
# ---  ------           --------------  -----  
#  0   R&D Spend        50 non-null     float64
#  1   Administration   50 non-null     float64
#  2   Marketing Spend  50 non-null     float64
#  3   State            50 non-null     object
#  4   Profit           50 non-null     float64

print(data.head())


'''------------------------------------Correlation Test using Hypothesis Testing Methodology------------------------------'''
#Goal: Check if rdSpend and Profit has any linear relationship
#      Check if rdSpend has any impact on profitability of the company.

# Check via visual EDA:
sns.scatterplot(data=data, x='R&D Spend', y= 'Profit')
# plt.show()
# Visually Ican see data is looking as linearly related.
''' Lets check Numerically as well:

 How to infer the correlation: 
 Use pd.corr() method, it internally uses Pearson method.
Range of corr() is -1 to +1

1. If the output positive and near to 1 : then corr is Positive
2. If the output is negative and near to 1 : then corr is Negative
3. If output is +ve or -ve but near to 0: then no - corr

Another set of rules / Guidence (Prashant):
1. If data is b/w 0.45 to 1 then: Positive correlation
2. If data is b/w -0.45 to +0.45 then: No co-relation
3. If data is b/w -1 to -0.45 then: Negative corelation

# Always consult the domain expert else follow below to define strong or weak
# If value is near 1 (+0.75 and +1) --------> strong +ve corr
# If value is near -1 (-1 and -0.75) -------> strong -ve corr

'''

print(data['R&D Spend'].corr(data['Profit']))
# 0.9729004656594832
#  Data is very near to +1 we can conclude that it has +ve corelation.

'''
Perform Hypothesis Testing:

1. Create a Viable question goal.
2. Convert that question into a Hypothesis on that goal.
3. Select a statical formula to validate the Hypothesis. 
4. Get the SL (Statical Limit) value of the project.
5. Calculate the p-value from the statical test formula and compare this with the SL value

Perform Hypothesis Testing for correlation of Rd and Profit
'''

# 1. Check if markSpend  has any impact on company profit? 

# i. visual EDA
sns.scatterplot(data, x = 'Marketing Spend', y = 'Profit')
plt.show()

# ii. Numerical EDA
print(data['Marketing Spend'].corr(data['Profit']))
# 0.7477657217414767
#  since data is near to +1 we can say it has reltion ship

# iii. Hypothesis Testing
# 2. Create Hypothesis out of question: 
# H0: No liner relationship
# H1: linear relationship

# 3. Select the Statical formula: Pearson

# 4. Define the SL value:
SL = 0.5

curr, pvalue = pearsonr(data['Marketing Spend'], data['Profit'])

if pvalue <= SL:
    print('There is a Liner relationship b/w MS and P')
else:
    print('There is no positive Liner relationship b/w MS and P')