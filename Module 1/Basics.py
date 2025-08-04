
# take input from the user to print name

# fName = input('Enter your first name : ')
# lName = input('Enter your last name : ')
# fullname = fName + ' ' + lName

# print(f'Your full name is : {fullname}')

# simple addition progam by taking inputs from user

# num1 = int(input('Enter first number: '))
# num2 = int(input('Enter second number: '))

# sum = num1 + num2
# print(f'Addition of two provided numbers {num1} and {num2} is : {sum}')
'''
# decision control 

1. if statement
2. else if statement
3. elif statement
'''

# Write a program to take input from user and dertermine if the number is greater than 10 or not
'''
num = int(input('Enter any number: '))
if num > 10:
    print('Number is greater than 10')


if else:
# write a program to check if input is even or odd


num  = int(input('Enter any number : '))

if num % 2 == 0:
    print('Number is even.')
else:
    print('Number is odd.')



# elif condition:

num = int(input('Enter a number : '))

if num > 10:
    print('Number is greater than 10')
elif num < 10:
    print('Number is less than 10')
elif num == 10:
    print('Number is 10')


# Accept the number from the user and display message based on given conditions
# Number between 1 to 10 -----> Good
# Number between 11 to 20 ----> Better
# Number between 21 to 30 ----> Best
# Number above 30 ------------> Outstanding

num = int(input('Enter any number: '))

if num >=1 and num <= 10:
    print('Good')
elif num >= 11 and  num<= 20:
    print('Better')
elif num >= 21 and num <= 30:
    print('Best')
elif num > 30:
    print('Outstanding')


#Question: Loan App Example

# Approve the loan of the customer if the sal is greater than 8000 and the credit score is greater than 700
# The applicant must not be a student
# if the applicant is a student reject the loan irrespective of the sal and credit condition

#User input

# Enter Salary:
# Enter Credit Score (0-800):
# Are you a student (Y/N):


salary = int(input('Enter your salary: '))
credit  = int(input('Enter your credit score: '))
isStudent = input('Are your studet ?  (Y/N): ').upper()


if isStudent == 'Y':
    print('Sorry your application can not be processes.')
elif(credit > 700 and salary > 8000):
    print('Loan approved..')
else:
    print('Loan Rejected')



'''
