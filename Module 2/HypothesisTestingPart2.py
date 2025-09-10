import pandas as pd
import numpy as np

from scipy.stats import pearsonr, shapiro, ttest_ind, chi2_contingency

data = pd.read_csv('50_Startups.csv')

print(data.info())

#  #   Column           Non-Null Count  Dtype
# ---  ------           --------------  -----
#  0   R&D Spend        50 non-null     float64
#  1   Administration   50 non-null     float64
#  2   Marketing Spend  50 non-null     float64
#  3   State            50 non-null     object
#  4   Profit           50 non-null     float64
'''
#Perform Hypothesis Testing
#
# 1. Create a Viable Question Goal (The question must result in a Binary Outcome)
# 2. Convert the question into Hypothesis (H0 and H1)
# 3. Select the Statistical Test/ Formula/ Tool/ Method to validate the Hypothesis (Who wins?)
# 4. Select/Determine/Decide the SL of the project
# 5. Calc the p-value from the statistical test formula and compare the same with the SL to decide who wins?
#



# TEST FOR FEATURE ELIMINATION | Feature Comparison (In terms of Inferential Stats | Model Training | Model Building)
# Compare two columns/variables to check if they are statistically similar (Data Analysis Perspective)

                                                        Test for feature elimination:
                        Numerical and Numerical Pair                                        Numerical and categorical Pair | Categorical and Categorical Pair
                    |                                  |                                                                   |
            Normally Distributed            Not Normally Distributed                                                    Chi-Squre Test
                |                                      |
            Parametric Test                     Non-Parametrica Test                

            
if Col1 and Col2 pass Normality TEst:
        Use Parametric Test
else:
    Use Non-Parametric Test

    
2. Create the Hypothesis out of this:
    H0: Eliminate one column
    H1: Do not Eliminate column
# '''


'''
1. Goal: Check if RDSpend and Administration are statiscally significant | similar | twins
#  
# 
# Step 1: Check if the Column is Normally Distributed  : Using Shaprio Test
#       1. if Yes: Parametric
#       2. else: Non Parametric
# Step 2: Convert the question into Hypothesis: 
#           H0: If Statically Same then we need to remove one of them
#           H1: If Not Statically Same then we need to preserve them.
# 
# Step 3: Use/Select any of the Test from the Parametri or Non-Parametric category and apply that to get pvalue
# Step 4: Select SL value.
# Step 5: Compare SL with the pvalue to get the result
'''
# print(data['R&D Spend'])

# 1. Calculation of If the cols are statically same or not and 
# 2. Hypothesis:
SL = 0.05
curr, pval = shapiro(data['R&D Spend'])
if pval >= SL:
    print('H1: Yes Normally Distributed')
else:
    print('H0: No, Not Normally Distributed')

# print(data['Administration'])
SL = 0.05
curr, pval = shapiro(data['Administration'])
if pval >= SL:
    print('H1: Yes Normally Distributed')
else:
    print('H0: No, Not Normally Distributed')

# 3. Test: Parametric: Student tTest
# from scipy.stats import ttest_ind

# 4. Get the value of SL
SL_temp = 0.05


curr, pvalue = ttest_ind(data['R&D Spend'], data['Administration'])
print(pvalue)

if pvalue <= SL_temp:
    print('H1: These cols are not statically Significant, we can preseve them')
else:
    print('H0: These cols are statically Significant, we can remove them')


'''
# 2. Test for Feature Elimination for RDSpend and MArketing

Step 1: Check if both the cols are Normally Distributed or not ? 

Step 2: Create a Hypothesis:
        H1: Both are Normally Distributed.
        H0: Both are not Normally Distributed

Step 3: Get the SL value

Step 4: Select statical test for calculating the pvalue

Step 5: Calculate the pvalue and get main result
'''

#1.
SL = 0.05
curr, pvalue = shapiro(data['R&D Spend'])
if pvalue >= SL:
    print('H1: Column is Normally Distributed')
else:
    print('H0: Column is not normally Distributed')

curr, pvalue = shapiro(data['Marketing Spend'])
if pvalue >= SL:
    print('H1: Column is Normally Distributed')
else:
    print('H0: Column is not normally Distributed')

# 2. Since both are Normally Distributed: We can use Parametric test, ex: Student Ttest.

curr, pvalue = ttest_ind(data['R&D Spend'], data['Marketing Spend'])

if pvalue <= SL:
    print('H1: Both cols are not statically different, we can Preserve them')
else:
    print('H0: Both cols are statically same, we can remove one of them.')

# In case of non-parametric dataset we can use wilcoxon algo.

'''
#Chi-square Test for Feature Elimination
 - It is meant to deal with feature (Indenpendent variables) and label (Dependent variables) pair.

 a. Feature is categorical and labeled as Numerical
 b. Feature is categorical and labeled as categorical
 c. Feature is Numerical and labeled as Catrogorical

# The goal of this test is to check if there exist a statical relatioship b/w two columns or not. 
'''

ct = pd.crosstab( data['State'], data['Profit'])
print(ct)

SL = 0.05

_,pvalue,_,_ = chi2_contingency(ct)
if pvalue <= SL:
    print('H1: Both the cols are statically different, we can preserve them')
else:
    print('H0: Both the columns are statically same, we can remove State column. ')



#1. Create notes of Hypothesis 
# 2. Normality TEstsing, usage of normality testing
# 3. Paramatric and non parametric test cases,
# 4. Feature elimination.
# 5. End to end flow notes on notebook + Revision