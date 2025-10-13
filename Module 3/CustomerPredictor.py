import pickle
import numpy as np

model = pickle.load(open('CustomerPredictor.pkl', 'rb'))

age = input('Enter age: ')
salary = input('Enter salary: ')
feature = np.array([[age, salary]])

print(feature)
values = model.predict(feature)
print(values)
