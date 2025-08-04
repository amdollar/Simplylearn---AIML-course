'''

1. List
2. Tuple
3. Set
4. Dictionary

Slicing technique is used to extract the list elements in a range i.e: 4:8


lst = [1,3,5,2,4,2,4,5,6]

print(lst[2:7])

print(lst[3:7])

print(lst[len(lst)-1])


# Dictionary:

stateAndCapital = {
    'UP': 'Lko',
    'MP': 'Ind',
    'TS': 'Hyd'
}
print(stateAndCapital['UP']) 


#tuple

nums = (1,2,3,4,5,6)
print(nums[2:6])

nums[2] = 3 # not permitted
# Set:

1. Only unique values.
2. Helps to perform Math Set operations.
'''


lst = [2112,4,2,3,34,232,32,34,23,232,3343,43232,3,2,1,1,1,1,3,3]
print(set(lst))

set1 = {1,2,3,4}
set2 = {4,3,5,6}

print(set1.union(set2))
print(set1.intersection(set2))