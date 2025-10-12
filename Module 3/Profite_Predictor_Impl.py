import numpy as np
import pickle

profit_predictor = pickle.load(open('ProfitPredictor.pkl', 'rb'))
state_imputer = pickle.load(open('StateConvertor.obj', 'rb'))

state_name = input("Enter the state name: ")
r_d_spend = input('Enter the R&D Spend values: ')
administration = input('Enter the Administration Spend values: ')
marketing_spend = input('Enter the Marketing Spend Spend values: ')

print(f'State name: {state_name}, R&D spend value: {r_d_spend}, Adminstration value: {administration}, Marketing spend value: {marketing_spend}')

if state_name in state_imputer.categories_[0]:
    state = state_imputer.transform(np.array([[state_name]]))
    feature_value = np.concatenate((state, np.array([[r_d_spend, administration, marketing_spend]])), axis = 1)
    predicted_value = profit_predictor.predict(feature_value)
    print(f'Predicted profit based on provided values is: {np.round(predicted_value, 2)}')
else:
    print(f'State is not recognized by ai model!')

