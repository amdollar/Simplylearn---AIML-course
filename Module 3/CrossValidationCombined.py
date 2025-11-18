# This demo is to find out the best model among different ML algorithms that can produce the best result


import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

models = [{'model_name':   'KNeighborClassification', 'instance' : KNeighborsClassifier()}, 
          {'model_name': 'DecisionTreeClassifier', 'instance': DecisionTreeClassifier()},
          {'model_name': 'RandomForestClassifier', 'instance': RandomForestClassifier()},
          {'model_name': 'BaggingClassifier (KNeighborsClassifier)', 'instance': BaggingClassifier(KNeighborsClassifier())},
          {'model_name': 'BaggingClassifier(LogisticRegression)', 'instance': BaggingClassifier(LogisticRegression())},
          {'model_name': 'SVC', 'instance': SVC()}]


data = pd.read_csv('Social_Network_Ads.csv')
features = data.iloc[:,[0,1]].values
labels = data.iloc[:,[2]].values

cv_vals = [5,10]
results = []
for model in models:
    for cv in cv_vals:
        values = cross_val_predict(model['instance'], 
                          features,
                          labels, 
                          cv= cv)
        
        # Get the score of each model
        model_name = model['model_name']
        cvs_str = f'cvs: {str(cv)}'
        results.append({'Model': model_name, 'CVS': cvs_str, 'Scores': values.mean()})

results_df = pd.DataFrame(results)

'''
Model: KNeighborClassification, CVS: 5, Scores: 0.305
Model: KNeighborClassification, CVS: 10, Scores: 0.3225
Model: DecisionTreeClassifier, CVS: 5, Scores: 0.33
Model: DecisionTreeClassifier, CVS: 10, Scores: 0.3425
Model: RandomForestClassifier, CVS: 5, Scores: 0.3525
Model: RandomForestClassifier, CVS: 10, Scores: 0.3525
Model: BaggingClassifier (KNeighborsClassifier), CVS: 5, Scores: 0.29
Model: BaggingClassifier (KNeighborsClassifier), CVS: 10, Scores: 0.305
Model: BaggingClassifier(LogisticRegression), CVS: 5, Scores: 0.31
Model: BaggingClassifier(LogisticRegression), CVS: 10, Scores: 0.295
Model: SVC, CVS: 5, Scores: 0.185
Model: SVC, CVS: 10, Scores: 0.185

'''

best_model = results_df.groupby('Model')['Scores'].mean().idxmax()

best_cvs =results_df['CVS'][ results_df[results_df['Model'] == best_model].iloc[:,[1,2]]['Scores'].idxmax() ]
print(f'the best model is: {best_model} with a cvs of: {best_cvs}')