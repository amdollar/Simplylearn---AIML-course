import pandas as pd
import numpy as np

#Load a CSV file named sales_data.csv into a pandas DataFrame. Display the first 5 rows of the DataFrame.

sales_fl = pd.read_csv('sales_data.csv')
print(sales_fl.head())

#Filter the DataFrame to show only the rows where the Sales column is greater than 500.
#Display the filtered DataFrame.

df_sale_500 = sales_fl[sales_fl['Sales'] > 500]
print(df_sale_500)

#Filter the DataFrame to show only the rows where the Region is 'North' and the Sales are greater than 700.
#Display the filtered DataFrame.

df_north_700 = sales_fl[(sales_fl['Region'] == 'North') & (sales_fl['Sales'] > 700)]
print(df_north_700)

#Select the Product, Sales, and Region columns from the DataFrame and display them.
selected_cols = sales_fl.iloc[:,1:4]

print(selected_cols)

#Add a new column named SalesTax which is 10% of the Sales column.
#Display the first 5 rows of the DataFrame with the new column.

sales_fl['SalesTax'] = sales_fl['Sales'] * .10
print(sales_fl.head())

#Create a new column named HighSales that contains True if the Sales value is greater than 800 and False otherwise.
#Display the first 5 rows of the DataFrame with the new column

def sales_cal(val):
    if val > 800:
        return True
    else:
        return False

sales_fl['High Sales'] = sales_fl['Sales'].apply(sales_cal)
print(sales_fl.head())

# Filter the DataFrame to show only the rows where the HighSales column is True.
# Display the filtered DataFrame.

high_sales_filtered = sales_fl[sales_fl['High Sales'] == True]
print(high_sales_filtered)

# Add two new columns: Discount which is 5% of Sales, and FinalPrice which is Sales minus Discount.
# Display the first 5 rows of the DataFrame with the new columns.
# gatiman: 

sales_fl['Discount'] = sales_fl['Sales'] * 0.05
print(sales_fl)

sales_fl['FinalPrice'] = sales_fl['Sales'] - sales_fl['Discount']
print(sales_fl)

#Filter the DataFrame to show only the rows where the SalesTax is greater than 50 and FinalPrice is less than 800.
# Display the filtered DataFrame.

print(sales_fl[(sales_fl['FinalPrice'] < 800) & (sales_fl['SalesTax'] > 50)])

#Filter the DataFrame to show only the rows where the SalesTax is greater than 50 and FinalPrice is less than 800.
# Display the filtered DataFrame.

print(sales_fl[(sales_fl['SalesTax'] > 50) & (sales_fl['FinalPrice'] < 800)])


# Filter the DataFrame to show only the rows where the Region is 'South' and Sales are between 600 and 1000 (inclusive).
# Display the filtered DataFrame.

df = sales_fl[(sales_fl['Region'] == 'South') & (sales_fl['Sales'].between(600,1000))]
print(df)