'''
# acts as an iterator


list1 = [2,3,5,2,6,7]

for num in list1:
    print(num)

# Extract even numbers only from the given list

nums = [1,2,3,4,5,6,7,8,9,10]
evenResult = []
 
for num in nums:
    if num % 2 == 0:
        evenResult.append(num)

print(evenResult)



# Enumerate the String:

text = 'You are in Hyderabad'
    
for char in text:
        print(char)

lst = text.split(' ')

for wor in lst:
    print(wor)

# Iterating the dictionary:


dic = {'k1': 1, 'k2': 2, 'k3': 3}

print(dic) # {'k1': 1, 'k2': 2, 'k3': 3}

for element in dic:
    print(element)
# k1
# k2
# k3

for element in dic.values():
    print(element)
# 1
# 2
# 3

for element in dic.items():
    print(element)
# ('k1', 1)
# ('k2', 2)
# ('k3', 3)

'''