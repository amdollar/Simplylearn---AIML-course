'''
File handling:

1. reading a file 
2. writing a file


#Build-in function is there in python: open()

# Syntax: open(fileLocation, mode)

Mode:

r ---------- read
w ---------- write
a ---------- apend
r+ --------- read and write





data = 'Welcome to python, without with statement.'
f = open('test.txt', 'w')
try:
    f.write(data)
except Exception as e:
    print(e.message()) 
finally:
    f.close()


f = open('test.txt', 'r')
try:
    data = f.read()
    print(data)
finally:
    f.close()

File handeling is same as Java, and in order to let python internally handle the resource management, 
With Keyworld is being used:

'''
writeOne = 'Init if data'
with open('text.txt', 'w') as f:
    f.write(writeOne)

writeDate = 'This is example of appending the content in exsiting file.'
with open('text.txt', 'a') as f:
    f.write('\n This is new line and looks good.')

data = 'welcome to file handeling...!'
with open('test.txt', 'r') as f:
    fi = f.read()
    print(fi)
