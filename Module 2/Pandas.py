import pandas as pd
import numpy as np

'''

# A series in Pandas is nothing but like column in a Table. 
# 

# Create a series from the List:

ls = [21,22,23,24,25]
pdls = pd.Series(ls)

print(f'Pandas series: {pdls}')

# Return the first value from this list:

print(f'First value from the series: {pdls[0]}')

# We can create our own lables (i.e.: 0,1,2,3) by passing another 'index' varible in the series creation

labled_sr = pd.Series(ls, index= ['a','b','c','d','e'])
print(labled_sr)

# Key value objs as series:
seq = {"a": 21, "b": 22, "c": 23}
myvar = pd.Series(seq)
print(myvar)


# Dataframe: These are multi-dimensional tables. 
# Create a DataFrame
# 1. Using Collection Object
# 2. Loading a file

first_df = pd.DataFrame({
    "Calories": [120,122,124],
    "Duration": [50,45,60]
})

print(f'First Dataframe: {first_df}')
data = [
    [9, 'Akshita', 290],
    [10, 'Anurag', 200],
    [11, 'Sucheta', 232],
    
]

simple_df = pd.DataFrame(data)

print(f'Simple dataframe: {simple_df}')

# Dataframes will have two types of indexes: 
# 1. Row index, 2. Column indexes. 

# Index will start from 0

# It is recommended to name these columns, for easier use.
simple_named_df = pd.DataFrame(data, columns=['Roll number', 'Name', 'Marks'])
print(f'Simple column named data frame: {simple_named_df}')

simple_named_df.info()

# Load a Delimited File
# Delimited Files referes to how column values are seperated
# csv,tsv,pipe sep

#To load any delimited file,
# read_csv()
# header --- used to specify header in file (none means no header)


emp_data_csv = pd.read_csv('employee.csv', names=['eId, eName', 'eSal'])
print(emp_data_csv)

emp_data_csv.info()

emp_with_header = pd.read_csv('employeeWithHeaders.csv')
print(emp_with_header)

emp_special_csv = pd.read_csv('empSpecial.csv', sep='\t')
print(emp_special_csv)

emp_da = pd.read_csv('empSpecial.csv', sep='\t', header = None, names=['EmployeeId', 'EmployeeName', 'EmployeeSalary'], skiprows=1)
print(emp_da)


# Reading the outside file: 


outside_d = pd.read_csv("https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv")
# outside_d.info()
# print(outside_d)


# head() returns top 5 rows: default

print(outside_d.head())

# tail() retuns least 5 rows. default
print(outside_d.tail())

#--------------------------- Operations in Pandas:-----------------
 
'''

emp_header = pd.read_csv('employeeWithHeaders.csv')
# print(emp_header.info())

# printing headers of file:
# print(emp_header.columns)

# convering the Data Frame to a NumPy Array --- values

empVals = emp_header.values
# print(empVals)

# Convert Numpy Array to a Data Frame: 
# arra = np.arange(10,30)
# print(arra)

# df = pd.DataFrame(arra)
# print(df)


emp_df = pd.DataFrame(empVals, columns= ['Seq', 'Name', 'Sal'])
# print(emp_df)

# Get values of specific columns:

print(emp_df['Sal'])

#Goal: Create a new column named yearlySalary
#      Derive the values of yearlySalary from salary column

print(emp_df)
emp_df['Yearly Salary'] = emp_df['Sal'] * 12

print(emp_df)