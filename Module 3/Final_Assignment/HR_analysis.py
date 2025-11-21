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
        plt.savefig('Correlationheatmap.png', dpi = 300, bbox_inches= 'tight')
        plt.show()

        # 2.2 Distribution Plot:
        print('\n 2.2. Creation of distribution plots..')
        fig, axes = plt.subplots(1,3, figsize = (18, 5))

        # Employee satisfaction:
        axes[0].hist(self.df['satisfaction_level'], bins=30, alpha= 0.7, color= 'skyblue', edgecolor = 'black')
        axes[0].set_title('Distribution of employee satisfaction level', fontweight = 'bold')
        axes[0].set_xlabel('Satisfaction level')
        axes[0].set_ylabel('Frequency')

        # Employee Evaluation:
        axes[1].hist(self.df['last_evaluation'], bins=30, alpha= 0.7, color= 'lightgreen', edgecolor = 'black')
        axes[1].set_title('Distribution of last evaluation', fontweight = 'bold')
        axes[1].set_xlabel('Last evaluation score')
        axes[1].set_ylabel('Frequency')

        # Employee Evaluation:
        axes[2].hist(self.df['average_montly_hours'], bins=30, alpha= 0.7, color= 'salmon', edgecolor = 'black')
        axes[2].set_title('Distribution of average monthly hours', fontweight = 'bold')
        axes[2].set_xlabel('Average monthly hours')
        axes[2].set_ylabel('Frequency')

        plt.tight_layout()
        plt.savefig('distribution_plots.png', dpi= 300, bbox_inches = 'tight')
        plt.show()

        # 2.3. Bar plot of project count by employee status
        print('\n Creating bar plot for Project count analysis..')
        plt.figure(figsize=(10,6))
        sns.countplot(data= self.df, x='number_project', hue = 'left', palette=['lightblue', 'salmon'])
        plt.title('Employee Project Count Distribution (Stayed vs Left)', fontsize=14, fontweight = 'bold')
        plt.xlabel('Number of Projects')
        plt.ylabel('Count')
        plt.legend(title='Employee Status', labels=['Stayed', 'Left'])
        plt.tight_layout()
        plt.savefig('project_count_analysis.png', dpi = 300, bbox_inches= 'tight')
        plt.show()

        self._print_eda_insights()

    def _print_eda_insights(self):
        'Print insight from EDA'
        print('\n Insights From EDA: ')
        print('-' * 40)

        # Project count insights
        project_left = self.df[self.df['left']==1]['number_project'].value_counts().sort_index()
        project_stayed = self.df[self.df['left']==0]['number_project'].value_counts().sort_index()

        print('Peoject count analysis: ')
        print(f'Employees who left most commonly worked on {project_left.idxmax()} projects')
        print(f'Employees who stayed most commonly worked on {project_stayed.idxmax()} projects')

        # Satisfaction insights
        avg_satisfaction_left = self.df[self.df['left']== 1]['satisfaction_level'].mean()
        avg_satisfaction_stayed = self.df[self.df['left']== 0]['satisfaction_level'].mean()

        print(f'\n Satisfaction Analysis: ')
        print(f'Average satisfaction of emplyees who left: {avg_satisfaction_left:.2f}')
        print(f'Average satisfaction of emplyees who stayed: {avg_satisfaction_stayed:.2f}')

    


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
