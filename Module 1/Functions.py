'''
# Function declearation and default values: 

def greetings(name, daytime= 'Night'):
    print(f'Hello {name}, wishing u a vey good {daytime}.')

greetings('Rahul', 'Morning')
greetings('anurag', 'evening')
greetings('Suche')

# Assigning function value to a variable:


def addition(num1, num2):
    return num1 + num2

res = addition(2,3)
print(res)

# Preprocession and Transformation function:

1.  map: This allows to map a value to iterable Object (Collection object)
map(functionLogic, iterableObject)

With the help of this we can supply function as an operation:


# squre each element in the list

def squre(num):
    return num * num

lst = [1,2,3,4,5]
sqlst = list(map(squre, lst))
print(sqlst)
[1, 4, 9, 16, 25]

2. filter: allows to filter elements that satisfied the condition
Note: It only retuns values that have True bool values.


def isEven(num):
    return num % 2 == 0
lst = [1,3,2,4,5,6,6,3,3,2,3,4,4]

print(list(map(isEven, lst)))
#[False, False, True, True, False, True, True, False, False, True, False, True, True]
print(list(filter(isEven, lst)))
#[2, 4, 6, 6, 2, 4, 4]


###### 
lambda function


squareLogis = lambda num : num * num

print(squareLogis(2))

addThreeNums = lambda num1, num2, num3 : num1+num2+num3
addition = addThreeNums(1,2,3)

print(addition)


'''