import numpy as np
import pandas as pd
ls = [1,2,3,4,5,6,7]

npls = np.array(ls)

twoDArray = np.array([
    [1,2,3],
    [4,5,6]
])

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


#------------Arithmatic Operations: +, -, *, /
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

#----------------- Changing dimantions of an Array:

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

# ------------------------- Extract elements from the array:

 

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


