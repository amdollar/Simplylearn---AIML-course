# Targets:
# Shuffling, Sampling with Replacement, Sampling without replacement


import numpy as np

dataset = [1,2,3,4,5,6]

# Sampling with replacement:
replacement_dataset = np.random.choice(dataset, size=3, replace=True)

print(replacement_dataset)

# Sampling without replacement (All values will be unique)
unique_values_dataset = np.random.choice(dataset, size = 3, replace=False)
print(unique_values_dataset)
# [1 2 2]
# [3 6 4]