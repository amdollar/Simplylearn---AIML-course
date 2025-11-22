import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier


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
        # plt.show()

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
        # plt.show()

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
        # plt.show()

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

    def employee_clustering(self):
        'Plateform k-Mean clustering of employees who left'
        print('\n' + '=' * 60)
        print('Employee clustering analysis')
        print('\n' + '=' * 60)

        # 3.1 Select relevant cols to filter employees who left
        print('\n 3.1 Selecting employees who left company..')
        left_employees = self.df[self.df['left']==1][['satisfaction_level', 'last_evaluation', 'left']]
        print(f'Employees who left: {left_employees}')
        print(f'Number of Employees who left: {len(left_employees)}')

        # 3.2 Perform K-Means clustering
        print('\n 3.2 Perform K-means clustering into 3 clusters..')
        clustering_data = left_employees[['satisfaction_level', 'last_evaluation']]

        k_means = KMeans(n_clusters=3, random_state=42, n_init=10)
        clusters = k_means.fit_predict(clustering_data)

        # Add cluster labels to the data
        left_employees_clustered = left_employees.copy()
        left_employees_clustered['cluster'] = clusters

        # Visualize clusters
        plt.figure(figsize=(12,8))
        scatter = plt.scatter(left_employees_clustered['satisfaction_level'], left_employees_clustered['last_evaluation'],
                              c = clusters, cmap = 'viridis', alpha=0.6, s=50)
        
        plt.scatter(k_means.cluster_centers_[:,0], k_means.cluster_centers_[:,1], c='red', marker= 'x', s=200, linewidths=3, label='Centorids')

        plt.xlabel('Satisfaction level', fontsize=12)
        plt.ylabel('Last Evaluation', fontsize=12)
        plt.title('K-mean clusterig of Employees who left \n (Based on Satisfaction & Evaluation)', fontsize=12, fontweight='bold')
        plt.colorbar(scatter, label='Cluster')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('employee_clustering.png', dpi=300, bbox_inches='tight')
        plt.show()

        # 3.3 Analyze clsuter
        self._analyze_clusters(left_employees_clustered, k_means.cluster_centers_)
        return left_employees_clustered, k_means

    def _analyze_clusters(self, clustered_data, centroids):
        '''Analyze and interpret the clusters'''
        print('\n 3.3 Cluster Analysis: ')
        print('-' * 40)

        for i in range(3):
            cluster_data = clustered_data[clustered_data['cluster'] ==i]
            avg_satisfaction = cluster_data['satisfaction_level'].mean()
            avg_evaluation = cluster_data['last_evaluation'].mean()

            count= len(cluster_data)

            print(f'\n Cluster {i}:')
            print(f'Count : {count } employees ({count/len(cluster_data) * 100 :.1f}%)')
            print(f'Average Satisfaction: {avg_satisfaction:.2f}')
            print(f'Average evaluation: {avg_evaluation:.2f}')

            # Interpretation

            if avg_satisfaction < 0.5 and avg_evaluation < 0.7:
                interpretation = 'Low performers with low satisfaction'
            elif avg_satisfaction <0.5 and avg_evaluation >=0.7:
                interpretation = 'High performers but un-satisfied (burnout risk)'
            else: 
                interpretation = 'Moderately satisfied employees'

            print(f'Interpretation: {interpretation}')   

    def handle_class_imbalance(self):
        '''Handle the class imbalance using SMOTE technique'''

        print('\n' + '=' * 60)
        print('4. Handling class Imbalance with SMOTE')
        print('=' * 60)

        # 4.1 Preprocess data
        print('4.1 Preprocessing data')
        
        # Saperate the categorical and numerical variables
        categorical_cols = self.df.select_dtypes(include=['object']).columns.tolist()
        numerical_cols = self.df.select_dtypes(include=['np.number']).columns.tolist()

        print(f'Categorical columns: {categorical_cols}')
        print(f'Numerical columns: {numerical_cols}')

        # Remove target variable from the numerical cols:
        if 'left' in numerical_cols:
            numerical_cols.remove('left')

        # Apply get_dummies to categorical variables
        df_encoded = pd.get_dummies(self.df[categorical_cols], drop_first=True)

        # combined with numerical valriables
        X = pd.concat([self.df[numerical_cols], df_encoded], axis=1)
        y = self.df['left']

        print(f'Final feature matrix shape: {X.shape}')
        print(f'Target distribution: \n{y.value_counts()}')

        # 4.2 Startified train-test split

        print('\n 4.2 Performing stratified train-test split (80-20)...')
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size = 0.2, random_state= 123, startify= y)
        
        print(f'Train set shape: {self.X_train.shape}')
        print(f'Test set shape: {self.X_test.shape}')
        print(f'Train target distribution: \n {self.y_train.value_counts()}')

        # 4.3
        print('\n 4.3 Applying SMOTE to balance the training set...')
        smote = SMOTE(random_state=123)
        self.X_train_balanced, self.y_train_balanced = smote.fit_resample(self.X_train, self.y_train)

        print(f'After SMOTE - Train set shape: {self.X_train_balanced.shape}')
        print(f'After SMOTE - Target distribution: \n {pd.Series(self.y_train_balanced).value_counts()}')

        return X, y
    
    def train_and_evaluate_models(self):
        '''Train models with 5-fold corss-validation'''
        print('\n' + '=' *60)
        print('5. Model training and evaluation')
        print('=' * 60)

        # Initialize models

        models = {'Logistic Regression': LogisticRegression(random_state=123, max_iter=1000),
                  'Random Forest': RandomForestClassifier(random_state=123, n_estimators=100),
                   'Gradient Boosting': GradientBoostingClassifier(random_state=123, n_estimators=100) }
        
        # 5-fold cross-validation

        from sklearn.model_selection import StratifiedKFold
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)

        for name, model in models.items():
            print(f'\n 5.{list(models.keys()).index(name) + 1} Training {name}...')

            # Fit model on balanced training data
            model.fit(self.X_train_balanced, self.y_train_balanced)
            self.models[name] = model

            # cross validation scores
            from sklearn.model_selection import cross_val_score
            cv_score = cross_val_score(model, self.X_train_balanced, self.y_train_balanced, cv = cv, scoring= 'accuracy')

            print(f'5-fold CV Accuracy: {cv_score.mean():.4f} (+/- {cv_score.std() * 2:.4f})')

            # Prediction on test set
            y_pred = model.predict(self.X_test)
            # Classification report
            print(f'\n Classification report for {name}: ')
            from sklearn.metrics import classification_report
            print(classification_report(self.y_test, y_pred))

            # Store model performance
            self.model_score[name] = {
                'cv_mean': cv_score.mean(),
                'cv_std': cv_score.std(),
                'predictions': y_pred
            }

                
            



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
        self.employee_clustering()

        # Step 4: Handle class imbalance
        self.handle_class_imbalance()

        # Step 5: Model training
        self.train_and_evaluate_models()

        # Step 6: Best Model identification:



if __name__ == '__main__':
    analyzer = HRAnalyzer('HR_Comma_sep.csv')



    analyzer.run_complete_analysis()
