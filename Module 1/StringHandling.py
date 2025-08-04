'''
Accessing elements/chars in the String:

Note: EVERY THING IN STRINGS IS CASE SENSITIVE

# Extract welcome from the following String: 

test = 'welcome to python class.'

print(test[0:7])

# Basic string ops: 
# 1. concatenation: Merging to or more string 


var1 = 'Hello'
var2 = 'World'

print(var1 + ' '+ var2)

# 2. Repetition: (*)

text = 'Hello ' *3 
print(text) #Hello Hello Hello

# 3. Search: if a word exists in the test.


text = 'welcome to hyderbad'
exist = 'Hyderbad'

if exist.lower() in text.lower():
    print('yes')
else:
    print('no')


# 4. Find and replace:
Find: gives start index of string if not available then -1
Replace: replaces the give word with the provided one. source string is not changed. 



text = 'I live in in hyderabad'
toFind = 'in'

print(text.find(toFind))

textnew = text.replace('hyderabad', 'Lucknow')
print(textnew)

# 5. Stripping whitespaces: whitespaces means the space in the starting and the ending of the string only

.strip(), lstrip(), rstrip()

test = "   text  "
print(test)
print(test.strip())

   text  
text

'''