'''
  error resistant, 
  in case of error: graceful termination of program


# 1. try..except block
# 2. else block
# 3. finally block

try:
    va = 5/0
except:
    print('Can not do devision by 0.')

try:
    num1 = int(input('Please enter number1: '))
    num2 = int(input('Please enter number2: '))

    print(f'Addition of {num1} and {num2} is : {num1 + num2}. ')

except:
    print('Please enter numbers only.')

# Using else block
# statements of else block will be executed when there is no exception.
# In the normal executions, expect will get ignored and else will get executed. 


try:
    num1 = int(input('Please enter number1: '))
    num2 = int(input('Please enter number2: '))

except:
    print('Please enter numbers only.')
else: 
    print(f'Addition of {num1} and {num2} is : {num1 + num2}. ')
    '''

# Finally block: Execute code irrespective of errors.

try:
    num1 = int(input('Please enter number1: '))
    num2 = int(input('Please enter number2: '))

except Exception as e :
    print('Please enter numbers only. ', e)
else: 
    print(f'Addition of {num1} and {num2} is : {num1 + num2}. ')
finally:
    print('Thanks for using my program..')
