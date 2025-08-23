import numpy as np
import pandas as pd
ls = [1,2,3,4,5,6,7]

npls = np.array(ls)

print(npls.ndim)
print(npls.max())

twoDArray = np.array([
    [1,2,3],
    [4,5,6]
])

print(twoDArray)
print(twoDArray.ndim)

print(twoDArray[0:1,0:3])