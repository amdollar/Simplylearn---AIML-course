import pandas as pd
import numpy as np


marketing_data = pd.read_csv('marketing_data.csv')


print(marketing_data.info())
# Replace $ from the salary col:
marketing_data['Income'] = marketing_data['Income'].replace(r'^\s*$', np.nan, regex=True)
marketing_data['Income'] = marketing_data['Income'].str.replace(r'[$,]', '', regex=True)

print(f'Data Type: {marketing_data['Dt_Customer'].dtype}')

# Convert the cleaned string column to a numeric type
marketing_data['Income'] = pd.to_numeric(marketing_data['Income'])

print(marketing_data.info())
marketing_data['Income'] = pd.to_numeric(marketing_data['Income'])

print(marketing_data.info())

# Consider customers with similar education, and marital_status
# Data is correct without type errors and no Null

# Education: 
marketing_data['Education'] = marketing_data['Education'].str.strip()
# Have the consistence case:
marketing_data['Education'] = marketing_data['Education'].str.lower()
print(marketing_data['Education'])
# ['Graduation' 'PhD' '2n Cycle' 'Master' 'Basic']

marketing_data['Marital_Status'] = marketing_data['Marital_Status'].str.strip()
marketing_data['Marital_Status'] = marketing_data['Marital_Status'].str.lower()

print(marketing_data['Marital_Status'])
# ['Divorced' 'Single' 'Married' 'Together' 'Widow' 'YOLO' 'Alone' 'Absurd']

# Customer with same Marital status and Education may have same avg income: This means we need to get the median of group of ppl, with this common interest

# avg_income_data = marketing_data.groupby(['Marital_Status', 'Education'])[' Income '].mean()
average_income_map = marketing_data.groupby(['Education', 'Marital_Status'])['Income'].mean()

print(average_income_map)