import numpy as np
from numpy import random
# 1. Create an array of 10 zeros

zeros_arr = np.zeros(10)
print(f'Array of 10 zeros: {zeros_arr}')

# Create an array of 10 ones
ones_arr = np.ones(10)
print(f'Array of 10 ones: {ones_arr}')

# 3. Create an array of 10 fives
# Combination of : onces and distributaion
ten_five_arr = np.ones(10) * 5
print(f'Array of 10 fives: {ten_five_arr}')

# 4. Create an array of the integers from 10 to 50
ten_to_fiftine_arr = np.arange(10,51)
print(f'Array that has values from 10 to 50: {ten_to_fiftine_arr}')

# 5. Create an array of all the even integers from 10 to 50
ten_to_fiftine_ar = np.arange(10,51)
ten_fiftine_even = ten_to_fiftine_ar[ten_to_fiftine_ar % 2 == 0]
print(f'Array that has even values from 10 to 50: {ten_fiftine_even}')

# 6. Create a 3x3 matrix with values ranging from 0 to 8

three_three_arr = np.arange(9).reshape(3,3)
print(f'Array in range from 0 to 8 in 3X3: {three_three_arr}')

# 7. Create a 3x3 identity matrix

iden_mat = np.eye(3)
print(f'Identity matrix: {iden_mat}')

# 8. Use NumPy to generate a random number between 0 and 1

random_num = random.rand()
print(f'A random number between 0 and 1: {random_num}')


# 9. Use NumPy to generate an array of 25 random numbers sampled from a standard normal distribution
random_arr = random.randn(25)
print(random_arr)

# arr = np.arange(0.1,1.01, dtype=float, step=0.1).reshape(10,10)
print(np.arange(0.1,1.01,step=0.01))

# 10. Create an array of 20 linearly spaced points between 0 and 1:

liner_arr = np.linspace(0 , 1, 20)
print(liner_arr)

''' 
-------------------- Numpy Indexing and Selection: 
                                        Exercise based on given matrix:

array([[ 1,  2,  3,  4,  5],
       [ 6,  7,  8,  9, 10],
       [11, 12, 13, 14, 15],
       [16, 17, 18, 19, 20],
       [21, 22, 23, 24, 25]])

# WRITE CODE HERE THAT REPRODUCES THE OUTPUT OF THE CELL BELOW
# BE CAREFUL NOT TO RUN THE CELL BELOW, OTHERWISE YOU WON'T
# BE ABLE TO SEE THE OUTPUT ANY MORE

11. array([[12, 13, 14, 15],
       [17, 18, 19, 20],
       [22, 23, 24, 25]])

'''
search_arr = np.arange(1, 26).reshape(5,5)
print(search_arr)
res_arr = search_arr[2:5,1:5]
print(res_arr)

# 12 . WRITE CODE HERE THAT REPRODUCES THE OUTPUT OF THE CELL BELOW
# 20

# search_arr = np.arange(1, 26).reshape(5,5)
# print(search_arr)
res_20 = search_arr[3,4]
print(res_20)

#  13. array([[ 2],
#        [ 7],
#        [12]])

res_ag = search_arr[0:3,1].reshape(3,1)
print(res_ag)

# 14. array([21, 22, 23, 24, 25])
print(search_arr[4,0:5])

# 15. array([[16, 17, 18, 19, 20],
#        [21, 22, 23, 24, 25]])

print(search_arr[3:5, 0:5])

# 15. Get the sum of all the values in mat
all_sum = search_arr.sum()
print(all_sum)

# 16. Get the standard deviation of the values in mat
std = search_arr.std()
print(std)

# 17. Get the sum of all the columns in mat

col_sum = search_arr.sum(axis=0)
print(col_sum)

# 18. Get the sum of all the rows in mat:

row_sum = search_arr.sum(axis=1)
print(row_sum)