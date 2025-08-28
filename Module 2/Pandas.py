import pandas as pd
import numpy as np


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
 

emp_header = pd.read_csv('employeeWithHeaders.csv')
print(emp_header.info())

# printing headers of file:
print(emp_header.columns)

# convering the Data Frame to a NumPy Array --- values

empVals = emp_header.values
print(empVals)

# Convert Numpy Array to a Data Frame: 
arra = np.arange(10,30)
print(arra)

df = pd.DataFrame(arra)
print(df)


emp_df = pd.DataFrame(empVals, columns= ['Seq', 'Name', 'Sal'])
print(emp_df)

# Get values of specific columns:

print(emp_df['Sal'])

#Goal: Create a new column named yearlySalary
#      Derive the values of yearlySalary from salary column

print(f'Employee table before Y sal column: \n {emp_df}')
emp_df['Yearly Salary'] = emp_df['Sal'] * 12

print(f'Employee table after Y sal column : \n {emp_df}')

#Create a function that can accept salary and return the bonus of the salary
# salary less than or equal to 1500 -----> 10% bonus
# salary between 1501 and 6000 ----------> 5% bonus
# salary between 6001 and 9900 ----------> 2.5% bonus
# salary greater than 9900 --------------> 0 (no bonus)

#Pandas also allows you to broadcast function logic
#
# To broadcast a function in pandas, you will use apply() method

def bonus_cals(salary):
  if salary <= 1500:
    return round(salary * 0.10,3)
  elif salary > 1500 and salary <= 6000:
    return round(salary * 0.05,3)
  elif salary > 6000 and salary <= 9900:
    return round(salary * 0.025,3)
  else:
    return 0

emp_df['bonus'] = emp_df['Sal'].apply(bonus_cals)
print(f'Each employee salary after applying bonus: \n{emp_df}')

# turn all the names in the enam col to upper case

emp_df['Name'] = emp_df['Name'].apply(str.upper)

print(f' Each employee name after turning their upper :\n {emp_df}')

# Adding new column name department:

emp_df['Department'] = pd.Series(['IT', 'HR', 'Ops', 'RIS', 'Life', 'Bank'])
print(f'Table structure after adding new column department: \n {emp_df}')

# --------------------------------------------------Data traversal ---- Access the data
#
# Accessing data using Index location ------ iloc
#
# 1. iloc[ rowIndex,colIndex ]
# 2. iloc[ rowRange,colRange ]
# 3. iloc[ rowIndexList, colIndexList ]



employee_with_header = pd.read_csv('employeeWithHeaders.csv')
print(employee_with_header)

print(employee_with_header.iloc[ [0,1] , [0,1,2]])
#    eid     ename  esal
# 0    1  Prashant  1000
# 1    2      Amar  2000

# read data in range:

print(employee_with_header.iloc[ 0:6 , 0:3])   

#Goal: Extract those records whose monthly salary is greater than 5000

sal_condition = employee_with_header[employee_with_header['esal'] > 5000]
print(f'List of employees where salary is > 5000: \n {sal_condition}')

# Adding new column name department:

employee_with_header['Department'] = pd.Series(['IT', 'HR', 'RIS', 'RIS', 'Life', 'RIS'])
print(employee_with_header)

#Goal: Extract those records who belong to RIS Department

ris_employees = employee_with_header[employee_with_header['Department']== 'RIS']
print(ris_employees)

#Goal: Extract those records who belong to RIS dept and has monthly salary greater than 3000
ris_rich_emp = employee_with_header[(employee_with_header['Department']== 'RIS') & (employee_with_header['esal'] > 3000)]
print(ris_rich_emp)

# Extract the eid and ename only of these rich ris employees:

ris_rich_ename_eid = employee_with_header[(employee_with_header['Department']== 'RIS') & (employee_with_header['esal'] > 3000)].iloc[:, [0,1]]
print(ris_rich_ename_eid)

# Extarct the employees who are in RIS and HR

ris_hr_emp= employee_with_header[employee_with_header['Department'].isin(['RIS','HR'])]
print(ris_hr_emp)

# Print the name of employee whos salary is maximum in the table:

max_sal_name = employee_with_header[employee_with_header['esal'] == employee_with_header['esal'].max()]['ename']
print(max_sal_name)

salaries = pd.DataFrame([[1000],[2000],[3000],[4000],[5000],[6000],[7000],[8000],[9000],[2000000],[100000000]] , columns=['salary'])
print(f'Data frame: \n {salaries}')
print(salaries['salary'].mean)