import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from scipy.stats import mannwhitneyu
import warnings

warnings.filterwarnings('ignore')

def task_1_data_import_verification():
    """Task 1: Examine variables like Dt_Customer and Income to verify importation. """
    print("=" * 60)
    print("Task 1 : Data import verification:")
    print("=" * 60)
    
    df = pd.read_csv('marketing_data.csv')

    print(f"\n Dataset shape: {df.shape}")
    print(f"\n Dt_Customer examination: ")
    print(f" Data Type: {df['Dt_Customer'].dtype}")
    print(f" Sample values: {df['Dt_Customer'].head(3).tolist()}")
    print(f" Date Range: {df['Dt_Customer'].min()} to {df['Dt_Customer'].max()}")

    print(f"\n Income examination: ")
    print(f" Data type: {df[' Income '].dtype}")
    print(f"Sample Values: {df[' Income '].head(3).tolist()}")
    print(f"Missing Values: {df[' Income '].isna().sum()}")

    print(f"\n Total missing values: {df.isnull().sum().sum()}")

    return df

def task_2_missing_data_imputation(df):
    """Task 2: Handle the missing values and clean categorical data:"""
    print("\n" + "=" * 60)
    print("Task2 : Missing value imputation")
    print("=" * 60)

    df_clean = df.copy()

    print(f"\n Education categories: {df_clean['Education'].unique().tolist()}")
    print(f"\n Marital status categories: {df_clean['Marital_Status'].unique().tolist()}")


    # Clean unusual Marital status:
    status_mapping = {'YOLO':'Single', 'Absurd': 'Single', 'Alone': 'Single'}
    df_clean['Marital_Status'] = df_clean['Marital_Status'].replace(status_mapping)
    print(f"Cleaned martial status mapping: {status_mapping}")
    print(f"\n Marital status categories: {df_clean['Marital_Status'].unique().tolist()}")

    # Process Income:
    df_clean['Income'] = df_clean[' Income '].str.replace('$', '').str.replace(',', '').str.strip()
    df_clean['Income'] = pd.to_numeric(df_clean['Income'], errors='coerce')

    missing_before = df_clean['Income'].isnull().sum()
    income_median= df_clean.groupby(['Education', 'Marital_Status'])['Income'].median()

    
    def impute_income(row):
        if pd.isna(row['Income']):
            key = (row['Education'], row['Marital_Status'])
            return income_median.get(key, df_clean.groupby('Education')['Income'].median()[row['Education']])
        
        return row['Income']
    
    # Understand this more:
    df_clean['Income'] = df_clean.apply(impute_income, axis=1)
    
    missing_after = df_clean['Income'].isnull().sum()

    print(f"\nMissing income values - Before: {missing_before}, After: {missing_after}")
    print(f"Income Range: ${df_clean['Income'].min():,.0f} -  ${df_clean['Income'].max():,.0f}")

    # Removed space's col name from data set 
    df_clean = df_clean.drop(' Income ', axis = 1)
 
    return df_clean


def task_3_feature_engineering(df):
    """Task 3: Create new variables: """
    print("\n" + "=" * 60)
    print("Task 3: Feature Engineering")
    print("=" * 60) 

    df['Total_Children']  = df['Kidhome'] + df['Teenhome']
    children_dist = df['Total_Children'].value_counts().sort_index()
    print(f"\n Total Children: {dict(children_dist)} (0= {children_dist[0]/len(df) * 100:.1f}%, 1+={100-children_dist[0]/len(df)*100:.1f}%)")


    #Age, Spending and purchase
    df['Age'] = 2014-df['Year_Birth']
    spending_cols = ['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
    df['Total_Spending'] = df[spending_cols].sum(axis = 1)
    purchase_cols= ['NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases']
    df['Total_Purchases'] = df[purchase_cols].sum(axis= 1)

    print(f"Age: {df['Age'].min()}-{df['Age'].max()} (avg: {df['Age'].mean():.1f})")
    print(f"Spendings: $ {df['Total_Spending'].min():,.0f} - ${df['Total_Spending'].max():,.0f} (avg: {df['Total_Spending'].mean():.0f})")
    print(f"Purchases: {df['Total_Purchases'].min()}-{df['Total_Purchases'].max()} (avg:{df['Total_Purchases'].mean():.1f})")

    return df


def task_4_distributed_analysis_outliers(df):
    """Task 4: Generate box plots and Histrograms, identify outliers"""
    print("\n" + "=" * 60)
    print("Task 4: Distributaion analysis and Outliers")
    print("=" * 60) 

    numerical_cols = ['Age', 'Income', 'Total_Children','Total_Spending', 'Total_Purchases', 'Recency']
    print(f"\nStatical summary:")
    print(df[numerical_cols].describe().round(2))

    # Create visualization:
    fig, axes = plt.subplots(2, 3, figsize= (15, 10))
    fig.suptitle('Distribution Analysis', fontsize = 1)

    for i, col in  enumerate(numerical_cols):
        row, col_ind = i//3, i%3


        # Histrogram:
        if row == 0:
            axes[row, col_ind].hist(df[col], bins = 20, alpha = 0.7, edgecolor = 'black')
            axes[row, col_ind].set_title(f'{col} Distribution')
        else:
            axes[row, col_ind].boxplot(df[col])
            axes[row, col_ind].set_title(f'{col} Box Plot')

    plt.tight_layout()
    plt.savefig('task4_distributions.png', dpi = 300, bbox_inches = 'tight')
    plt.close()

    # Outliner detection and treatement:
    numerical_cols = ['Age', 'Income', 'Total_Children','Total_Spending', 'Total_Purchases', 'Recency']
    outlier_counts = {}
    for col in numerical_cols:
        Q1, Q3 = df[col].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)]
        outlier_counts[col] = f"{len(outliers)} ({len(outliers) / len(df) * 100:.1f}%)"

    print(f"\n Outliners detected: {outlier_counts}")
    
    # Outliner Treatement:
    for col in ['Age', 'Income']:
        Q1, Q3 = df[col].quantile([0.25,0.75])
        IQR = Q3- Q1
        lower_bound = Q1 - 1 * IQR
        upper_bound = Q3 + 1 * IQR

        df.loc[df[col] < lower_bound, col] = lower_bound
        df.loc[df[col] > upper_bound, col] = upper_bound

    print("Applied outlier treatement for Age (18-90) and Income (99th percentile cap)")

    return df

def task5_categorical_encoding(df):
    """TAsk 5: Apply ordianl and one-hot encoding."""
    print("\n " + "=" * 60) 
    print("Task 5: Categorical encoding")
    print("=" * 60)

    df_encoded = df.copy()

    # Ordinal encoding:
    education_order = ['Basic', '2n Cycle', 'Graduation', 'Master', 'PhD']
    education_mapping = {
        edu: i for i , edu in enumerate(education_order)
    }
    df_encoded['Education_Ordinal'] = df['Education'].map(education_mapping)

    print(f"Education ordinal encoding: {education_mapping}")

    # One-hot encoding for Marital Status
    marital_dummies = pd.get_dummies(df['Marital_Status'], prefix='Marital')
    df_encoded = pd.concat([df_encoded, marital_dummies], axis = 1)

    print(f"Marital status One-Hot-Encoded columns: {marital_dummies.columns.tolist()}")

    #Label encoding for Country:

    le_country = LabelEncoder()
    df_encoded['Country_Encoded'] = le_country.fit_transform(df['Country'])

    # Understand this:
    print(f"Country Encoding: {dict(zip(le_country.classes_, range(len(le_country.classes_))))}")
    print(f"Final Dataset shape: {df_encoded.shape}")

    return df_encoded

def task_6_correlation_analysis(df):
    """Task 6: Generate Correlation heatmap."""
    print("\n " + "=" * 60) 
    print("Task 6: Correlation heatmap")
    print("=" * 60)

    numerical_cols = [col for col in df.select_dtypes(include = [np.number]).columns if col not in ['Id', 'Year_Birth']]
    correlation_matrix = df[numerical_cols].corr()

    # Create Heatmap:
    plt.figure(figsize=(12,10))
    mask = np.triu(np.ones_like(correlation_matrix, dtype = bool))
    sns.heatmap(correlation_matrix, mask = mask, annot = True, cmap = 'RdYlBu_r', center=0, fmt = '.2f')
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('task6_correlation_heatmap.png', dpi = 300, bbox_inches = 'tight')
    plt.close()

    # Find and display strong correlation:
    strong_correlation = [(correlation_matrix.columns[i], correlation_matrix.columns[j], correlation_matrix.iloc[i,j])
                            for i in range(len(correlation_matrix.columns))
                            for j in range(i+1, len(correlation_matrix.columns))
                            if abs(correlation_matrix.iloc[i,j]) > 0.7]

    print(f"Strong Correlation (|r| > 0.7): {len(strong_correlation)} found")
    for var1, var2, corr in sorted(strong_correlation, key = lambda x: abs(x[2]), reverse= True):
        print(f" {var1} -> {var2}: {corr:.3f}")

    return correlation_matrix

def task_7_test_hypothesis(df):
    """Task 7: Test specified hypothesis."""
    print("\n " + "=" * 60) 
    print("Task 7: Hypothesis Testing")
    print("=" * 60)

    median_age = df['Age'].median()

    #7.a. Older vs Younger store purchase:
    older= df[df['Age']> median_age]['NumStorePurchases']
    younger = df[df['Age'] < median_age]['NumStorePurchases']

    _,p_val = mannwhitneyu(older, younger, alternative = 'greater')

    result = "Supported" if p_val < 0.05 else "Non Supported"
    print(f"7.a. Older ({older.mean():.1f}) vs Younger ({younger.mean():.1f}) store Purchase: {result}")

    #7b. Children vs no Children web purchase:
    with_kids = df[df['Total_Children'] > 0]['NumWebPurchases']
    without_kids = df[df['Total_Children'] == 0]['NumWebPurchases']
    _,p_val = mannwhitneyu(with_kids, without_kids, alternative = 'greater')

    result = "Supported" if p_val < 0.05 else "Non Supported"
    print(f"7.b. With kid ({with_kids.mean():.1f}) vs No Kids ({without_kids.mean():.1f}) web purchase: {result}")

    # 7.c. Store cannibalization (Negative correlation indicate cannibalization:
    
    web_corr = df['NumStorePurchases'].corr(df['NumWebPurchases'])
    catalog_corr = df['NumStorePurchases'].corr(df['NumCatalogPurchases'])
    cannibalization = "Yes" if (web_corr < 0 or catalog_corr < 0) else "No"
    print(f"7 c. Store Cannibalization: {cannibalization} (web: {web_corr:.3f}, catalog: {catalog_corr:.3f})")

    # 7d. Us vs Non US purchases:
    us_purchases = df[df['Country'] == 'US']['Total_Purchases']
    no_us_purchases = df[df['Country'] != 'US']['Total_Purchases']
    _, p_val = mannwhitneyu(us_purchases, no_us_purchases, alternative= 'greater')
    result = "Supported" if p_val < 0.05 else "Not Supported"
    print(f"7d. Us ({us_purchases.mean():.1f}) vs Non-Us ({no_us_purchases.mean():.1f}) purchases: {result}")

    return df

def task_8_specific_visualization(df):
    """Task 8: Create Specific Visualization."""
    print("\n " + "=" * 60) 
    print("Task 8: Specific Visualization")
    print("=" * 60)

    fig, axes = plt.subplots(3,2, figsize =(15,18))
    fig.suptitle('Specific Business Analysis', fontsize = 14)

    # 8.a. Product Performance:
    product_cols = ['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds']
    
    # Calculate total revenue for each Product Category:
    product_revenue = {}
    for col in product_cols:
        product_name = col.replace('Mnt', '')
        total_revenue = df[col].sum()
        product_revenue[product_name] = total_revenue

    # Create bar chart showing revenue of above cols:
    product_name = list(product_revenue.keys())
    revenue_value = list(product_revenue.values())
    axes[0,0].bar(product_name, revenue_value)
    axes[0,0].set_title('8a. Product Revenue Performance')
    axes[0,0].tick_params(axis = 'x', rotation= 45)

    # Find and print top three products by revenue:
    sorted_products = sorted(product_revenue.items(), key = lambda x: x[1], reverse = True)
    top_3_products = dict(sorted_products[:3])
    print(f'Top  Products: {top_3_products}')

    # 8.b: Age vs Campaign acceptance
    age_bins, age_labels = [18,30,40,50,60,100], ['18-29','30-39', '40-49','50-59','60+']
    df['Age_Bin'] = pd.cut(df['Age'], bins = age_bins, labels = age_labels)
    age_acceptance = df.groupby('Age_Bin')['Response'].mean() * 100
    axes[0,1].bar(range(len(age_acceptance)), age_acceptance.values)
    axes[0,1].set_title('8b. Campain Acceptance by Age.')
    axes[0,1].set_xticks(range(len(age_acceptance)))
    axes[0,1].set_xticklabels(age_acceptance.index)
    print(f"Age acceptance (%): {dict(zip(age_acceptance.index, age_acceptance.values.round(1)))}")

    # 8.c: Country compaign acceptance:
    top_countries = df.groupby('Country')['Response'].sum().nlargest(5)
    axes[1,0].bar(range(len(top_countries)), top_countries.values)
    axes[1,0].set_title('8c. Top countries by campaign acceptance')
    axes[1,0].set_xticks(range(len(top_countries)))
    axes[1,0].set_xticklabels(top_countries.index)
    print(f"Top countries: {top_countries.to_dict()}")

    # 8d: Chindren vs spending
    children_spending = df.groupby('Total_Children')['Total_Spending'].mean()
    axes[1,1].plot(children_spending.index, children_spending.values, marker = 'o')
    axes[1,1].set_title('8d. Spendings by number of childrens')
    axes[1,1].set_xlabel('Number of children')
    print(f"Spending by children: {children_spending.round().to_dict()}")

    # 8e: Education of complainers
    complainers = df[df['Complain'] == 1]
    if len(complainers) > 0:
        complaint_rate = (complainers['Education'].value_counts()/ df['Education'].value_counts() *100).fillna(0)
        axes[2,0].bar(range(len(complaint_rate)), complaint_rate.values)
        axes[2,0].set_title('8e. Complaint rate by Education.')
        axes[2,0].set_xticks(range(len(complaint_rate)))
        axes[2,0].set_xticklabels(complaint_rate.index, rotation=45)
        print(f"Complaint rates : {complaint_rate.round(1).to_dict()}")
    else:
        axes[2,0].text(0.5,0.5, 'No Complaints', ha = 'center', va = 'center', transform= axes[2,0].transAxes)
        print('No Complaints found')

    
    # Channel preferences by age:
    channel_data = df.groupby('Age_Bin')[['NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases']].mean()
    x_pos, width = np.arange(len(age_labels)), 0.25
    for i, (channel, label) in enumerate(zip(['NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases'],
                                             ['Web', 'Catalog', 'Store'])):
        axes[2,1].bar(x_pos + i*width - width, channel_data[channel], width, label= label)
        axes[2,1].set_title('Channel preferences by Age')
        axes[2,1].set_xticks(x_pos)
        axes[2,1].set_xticklabels(age_labels)
        axes[2,1].legend()

    plt.tight_layout()
    plt.savefig('task8_specific_visualization.png', dpi = 300, bbox_inches = 'tight')
    plt.close()
    return df

def main():
    "Executing the tasks here."
    print('Marketing Campaign Data Analysis')

    df = task_1_data_import_verification()
    df = task_2_missing_data_imputation(df)
    df = task_3_feature_engineering(df)
    df = task_4_distributed_analysis_outliers(df)
    df = task5_categorical_encoding(df)
    correlation_metrix = task_6_correlation_analysis(df)
    df = task_7_test_hypothesis(df)
    df = task_8_specific_visualization(df)

    print("\n" + "=" * 60)
    print("Analysis Completed")
    print("=" * 60)




if __name__ == "__main__":
    main()