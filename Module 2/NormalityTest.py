# HypothesisTestingStaticalTesting -> FinalAssignmentHypothesis -> NormalityTest -> HypothesisTestingPart2

import pandas as pd

import numpy as np

data = pd.read_csv('50_Startups.csv')

print(data.info())

# Example: # Goal: Check if the given column has a Normal Distribution |< Gaussian Dis i.e: R & D Spend

'''
#Visual EDA

# sns.displot(data['R&D Spend'] , kind="kde")


# Numerical EDA:
# Skewness and Kurtosis:

# The above metrics can help us to identify how much deviation from normality:
# a. Skewness : Measures the asymmetry in normal distribution (if value is close to 0, its normal dist)
# b. Kurtosis: Measuring the peakedness (Should be close to 3 for Normal Dist)

'''
print("Skewness is ",data['R&D Spend'].skew())
print("Kurtosis is ",data['R&D Spend'].kurt())

# Skewness is  0.164002172321177
# Kurtosis is  -0.7614645568424674

'''
# Perform Hypothesis Testing for Normality test of rdSpend
Steps: 
        1. Create a viable question.
        2. Create a Hypothesis of that question
            i. H0: The column is not Normally distributed.
            ii. H1: The column is Normally distributed.
        3. Select the SL value.
        4. Select the statical forumla for measurement (To calculate the p-value)
            Shapiro Test
        5. Calculte the p-value and take decision


'''
from scipy.stats import shapiro

SL = 0.05

curr, pvalue = shapiro(data['R&D Spend'])

if pvalue >= SL:
    print('H1: the column is Normally distributed')
else:
    print('H0: The column is not normally distributed.')



