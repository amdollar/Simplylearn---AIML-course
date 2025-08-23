import numpy as np
import pandas as pd
from numpy import random
ls = [1,2,3,4,5,6,7]

npls = np.array(ls)

twoDArray = np.array([
    [1,2,3],
    [4,5,6]
])
'''
#  ---------------------- 01. Basic creation and printing array------------
# 
# Get the dimentions of the array: ndim
print(npls.ndim)

# Get the max element of the array:
print(npls.max())

# To check the shape of the array: shape(), total 

print(npls.shape)

#Creation of a 2D array


print(twoDArray.shape)
print(twoDArray)
print(twoDArray.ndim)

# Print attributes of an 2D array start and end range
# start:end
print(twoDArray[0:1,0:3])


#--------------------02. Arithmatic Operations: +, -, *, / --------------------------------
#All arithmatic operations are element based operations

addition = npls + npls
print(f'Addition of two 1-D arrays : {addition}')

multiplication = npls * npls
print(f'Multiplication of two 1-D arrays: {multiplication}')

addition_2d = twoDArray + twoDArray
print(f'Addition of two 2-D arrays : {addition_2d}')

multiplication_2d = twoDArray * twoDArray
print(f'Multiplication of two 2-D arrays: {multiplication_2d}')


# In case of Matrix operations:  Mathmetical Matrix

twoDMat = [[1,2,3], [4,5,6], [7,8,9]]

matrix1 = np.matrix(twoDMat)

print(matrix1 + matrix1) 
print(type(matrix1))  # numpy matrix

multiplication2Matrix = matrix1 * matrix1

print(f'Multiplication of 2 numpy Matrix: {multiplication2Matrix}')

# [[ 30  36  42]
#  [ 66  81  96]
#  [102 126 150]]

#----- Changing dimantions of an Array:

 #1D -> 2D : ravel() to convert multidimantions array to 1D array
 #2D -> 1D : reshape(dim, dim), pass -1 in the second index to numpy decide the total number of columns.

 #numpy.reshape(): This is the most common and versatile method for changing the shape of an array. 
 #It returns a new array with the same data but a different shape. The product of the dimensions in the new shape must
 #  equal the total number of elements in the original array. 
 #You can use -1 in one of the dimensions to have NumPy automatically calculate that dimension's size.

onedArray = np.arange(12)
print(f'Orignal 1-D array: {onedArray}')

reshapedknown = onedArray.reshape(3,4)
print(f'Reshaped to one 2 D array: {reshapedknown}')

# ------- Extract elements from the array:

 

print(npls)

# Extract 6 from last

index_of_six = npls[5] # [1 2 3 4 5 6 7]
print(f'Index of 6 from the given 1D array {index_of_six}')

#                        Extracting series : Extracting the elements from the array in a series

# arrayElement [ rowSeries: colSeries ]

# WHERE,  rowSeries -> startIndex:upperBound --------- upperBound = endIndex + 1
#         colSeries -> startIndex:upperBound

extract_arr = np.arange(25)
two_d_extract_arr = extract_arr.reshape(5,5)

print(two_d_extract_arr)

print(two_d_extract_arr[1,2])
print(two_d_extract_arr[2:4, 1:4])
# [[11 12 13]
#  [16 17 18]]

 
#                        Fancy Extracting:   Extracting elements from the array in random/not in series


extract_arr_fancy = np.arange(25).reshape(5,5)
print(extract_arr_fancy)

# from the given array print the following values: 3, 7, 11, 16, 23
#      element       rowIndex      colIndex
#    ------------------------------------------
#         3            0             3
#         7            1             2
#         11           2             1
#         16           3             1
#         23           4             3
#

#  variableName[ rowIndexList, colIndexList]

print(f'Elements 3 from the list: {extract_arr_fancy[0,3]}')
print(f'Elements 7 from the list: {extract_arr_fancy[1,2]}')
print(f'Elements 11 from the list: {extract_arr_fancy[2,1]}')
print(f'Elements 16 from the list: {extract_arr_fancy[3,1]}')
print(f'Elements 23 from the list: {extract_arr_fancy[4,3]}')

# Same data intrepreted in 2D array format:
print(extract_arr_fancy[[0,1,2,3],[3,2,1,1]].reshape(2,2))


#    ------------------03. Logical operations and Conditional Operations -----------------

searchable_array = np.arange(25)
# print(searchable_array)

print(23 in searchable_array) # True
# Comparision operations: & and , | or, ~ not
print(223 not in searchable_array) # True


# Find those elements whose value is greater than 11
# print(searchable_array > 11)
# [False False False False False False False False False False False False
#   True  True  True  True  True  True  True  True  True  True  True  True
#   True]

elements_greater_than_certain = searchable_array[searchable_array > 11]
print(f'Elements that are greater than 11 : {elements_greater_than_certain}')
# Elements that are greater than 11: [12 13 14 15 16 17 18 19 20 21 22 23 24]

two_d_Searchable_array = searchable_array.reshape(5,5)
print(f'2-D searchable array : {two_d_Searchable_array}')

# Extract values that are even in the array:
even_in_two_d = two_d_Searchable_array[two_d_Searchable_array % 2 ==0]
print(f'Even values from the 2-D searchable array : {even_in_two_d}')

# Extract values that are odd in the array:
add_values = two_d_Searchable_array[two_d_Searchable_array % 2 !=0]
print(f'Odd values from the 2-D searchable array : {add_values}')


# Return those elements that are less than 11 and is an even number
conditional_vals = two_d_Searchable_array[(two_d_Searchable_array > 11) & (two_d_Searchable_array % 2 !=0)]
print(f'Values that are greater than 11 and even : {conditional_vals}')

# Extract all even numbers except 10 from twodarray: 
print(searchable_array[(searchable_array != 10) & (searchable_array % 2== 0)])

-----------------------------------04. Numpy Initialiser Functions

# arange ------------- Generating arrays containing a sequence of numbers
# arange( start, stop(upperBoundValue) , step, dtype)

step_array= np.arange(1,10,2,dtype= float)
print(step_array)
# [1. 3. 5. 7. 9.]

# array of zeros or ones:

zero_arr = np.zeros(12).reshape(2,-1)
print(f'Zero containing array : {zero_arr}')

ones_arr = np.ones(12).reshape(2,-1)
print(f'One containing array : {ones_arr}')

#-------------------------- Random:------------------

# rand() ---- Returns random numbers which follow Uniform Distribution
# (Uniform Distribution means the number will always be in the range of 0 and 1)
print(f'Rand() impl: {random.rand()}')

# Get array of 10 random float values
print(f'Rand() arr impl: {random.rand(10)}')

# randn() --- Retuns the values which follows Normal Distribution: values mean will be b/w 0-1

print(f'Randn() impl: {random.randn()}')

print(f'Randn() arr impl: {random.randn(10)}')

             
# randint(n): gives random int by taking 1 arg for range 0 : n
# while True:
#     rand_num = random.randint(200)
#     print(rand_num)

# Generate 10 random values between 1 to 100
random_ten = random.randint(100, size = 10)
print(f'10 random values form 1 to 100 : {random_ten}')

random_2_last_row = random.randint(90, 100, size = 2)

print(f'2 random values b/w 90 to 100 : {random_2_last_row}')

# Guess the number game: User will enter any number b/w 1 to 10 and if that is returned then win else loss
'''


while True:
    guess = int(input('Generate any number from 1-10 : '))
    if guess in range(1,11):
        print(f'Entered number: {guess}')
        break
    else:
        print(f'Invalid number, try again.')

if random.randint(1,10) == guess:
    print('Yay! You won the game.')
else:
    print('Better luck next time.')