import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class HRAnalyzer:

    def __init__(self, data_path):
        "Init with data path"
        self.data_path = data_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.model_scores = {}

    def load_data(self):
        print('='*60)
        print('Loding dataset')
        print('='*60)

        self.df = pd.read_csv(self.data_path)
        print(f'Dataset shape: {self.df.shape}')
        print(f'\n First first 5 rows: ')
        print(self.df.head(5))
        print('\n Dataset information: ')
        print(self.df.info())


    def data_quality_check(self):
        '''Perform complete data quality check'''
        print("=" * 60)
        print('Data quality checks')
        print("=" * 60)

        # Check for missing values
        missing_values = self.df.isnull().sum()
        print(f'Number of missing values per column: {missing_values}')

        if missing_values.sum() == 0:
            print('No missing values found in the dataset!')
        else:
            print('Missing values found in the dataset!')

        # Basic statistics analysis
        print(f'\n Basic Statistics: ')
        print(self.df.describe())

        # Check for duplicates:
        print(f'Check for duplicates: ')
        print(self.df.duplicated().sum())

        return missing_values
    
    def exploratory_data_analytics(self):
        'Peform comprehensive EDA as per requirement'
        print('=' * 60)
        print('2. Exploratory Data Analysis')
        print('=' * 60)

        # Corelation Heatmap
        print(f' 2.1. Creation of corelation heatmap:')

        plt.figure(figsize=(12,8))
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        correlation_matrix = self.df[numeric_cols].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap = 'coolwarm', center=0, square=True, fmt= '.2f')
        plt.title('Correlation Matrix - Numerical Features', fontsize= 14, fontweight = 'bold')
        plt.tight_layout()
        plt.savefig('Correlationheatmap.png', dip = 300, bbox_inches= 'tight')
        plt.show()



    def run_complete_analysis(self):
        '''Complete program startup code.'''
        print('='* 60 )
        print('Starting HR Analysis code')
        print('='* 60 )

        # Loading the data from CSV file
        self.load_data()

        # Step 1: Perform Quality check:
        self.data_quality_check()


        # Step 2: EDA
        self.exploratory_data_analytics()

        # Step 3: Clustering
        # self.employee_clustering()



if __name__ == '__main__':
    analyzer = HRAnalyzer('HR_Comma_sep.csv')



    analyzer.run_complete_analysis()
