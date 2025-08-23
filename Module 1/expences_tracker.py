
import csv
from datetime import datetime
import os
from typing  import List, Dict, Union




class ExpenseTracker:

    def __init__(self):
        self.csv_file = str = 'expenses.csv'
        self.monthly_budget = float = 0.0
        self.expenses : List[Dict]= []

    def add_expense(self):
        print('---Add new expense ---')
        while True:
            date = input('Enter the expense Date: ')

            try: 
                datetime.strptime(date, '%Y-%m-%d')
                break
            except ValueError:
                print('Invalid date, please provide a valid date.')

        category = input('Enter expense catrgory: ')
        
        if not category.strip():
            print('Category can not be empty.')
            return
        
        while True:
            try:
                amount = float(input('Enter expense amount :'))
                if amount <= 0:
                    print('Amount must be greater than 0.')
                    continue
                break
            except ValueError:
                print('Invalid amount, please enter valid number.')
            
        description = input('Enter expense description: ')
        if not description.strip():
            print('Description can not be empty.')
            return
        
        expense= {
            'date' : date,
            'amount' : amount,
            'category' : category,
            'description' : description
        }

        self.expenses.append(expense)
        print(f'Expense added {amount } successfully for {category}.')

    def view_expense(self):
        print('---Printing all the expenses----')
        print(self.expenses)

        if not self.expenses:
            print('No expenses recorded.')
            return
        
        print(f"{'Date': <12} {'Catrgory:' : <15} {'Amount': <10} {'Description'}")
        print('-' *60)

        total = 0.0
        for expense in self.expenses:
            if all(key in expense and str(expense[key]).strip() for key in ['date', 'category', 'amount', 'description']):
                print(f"{expense['date']:<12} {expense['category'] :<15}"
                        f"{expense['amount']:< 9.2f} {expense['description']}")


                #    print(f'{expense['date']:<12} {expense['category']: <15} {expense['amount']:<9.2f} {expense['description']}')
                total += expense['amount']
            else:
                print('Incomplete expenses entery found and skipped.')

        print('-' *60)
        print(f'Total expenses: {total:.2f}')


    def track_budget(self):

        if self.monthly_budget <= 0:
            while True:
                try:
                    budget = float(input('No monthly budget allocated, Enter your monthly budget: '))
                    if budget <= 0:
                        print('Budget must be greater than 0')
                        continue
                    self.monthly_budget = budget
                    print(f'Monthly budget is set to: {self.monthly_budget}')
                    break
                except Exception as e:
                    print(f'Invalid budget amount {e}') 
        current_month = datetime.now().strftime('%Y-%m')
        monthly_expense = 0.0

        for expense in self.expenses:
            try:
                if expense['date'].startswith(current_month):
                    monthly_expense += expense['amount']

            except (ValueError, TypeError, KeyError):
                continue

        print(f'Monthly Budget: {self.monthly_budget}')
        print(f'Current Month Expense: {monthly_expense}')

        if monthly_expense > self.monthly_budget:
            print('You have exceeded your budget.')
        else:
            remaining = self.monthly_budget - monthly_expense
            print(f'Remaining budget: {remaining} left for month.')


    def save_expense(self):
        print('Save Expense to the CSV file.')

        try:
            with open(self.csv_file, 'w', newline='', encoding= 'utf-8') as file:
                fieldnames = ['date', 'category', 'amount', 'description']
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writeheader()
                if self.expenses:
                    writer.writerows(self.expenses)
                    print(f'{len(self.expenses)} saved in file.')
                else:
                    print('Empty file created with headers.')
        except Exception as e:
            print('Error while saving expenses: {e}')

    
    def display_menu(self):
        print('-'*10)
        print('''
            1. Add Expense.
            2. View Expenses.
            3. Track Expenses.
            4. Save Expenses.
            5. Exit.

            ''')
    def load_expenses(self):
        'Load expenses from CSV file.'
        global load_expenses

        if not os.path.exists(self.csv_file):
            print('No file exists as of now, starting from new...')

        try:
            with open(self.csv_file, 'r', encoding= 'utf-8') as file:
                reader= csv.DistReader(file)
                expenses = []

                for row in reader:
                    try:
                        row['amount'] = float[row['amount']]
                        datetime.strptime(row['date'], '%Y-%m-%d')

                        if all(key in row and str(row[key]).strip() for key in ['date', 'category', 'amount', 'discription']):
                            expenses.append(row)

                    except:
                        print(f'Skipping invaid entry.')
                        continue
                print(f'Loaded {len(expenses)} from the csv file')
            
        except Exception as e:
            print('Error while loading file')
            expenses = []
        


    def starter(self):
        'Main program function'
        print('Welcome to personal expense tracker!')

        self.load_expenses()

        while True:
            self.display_menu()
            input_choice = input('Enter your choice (1-5): ').strip()
            
            if input_choice == '1':
                self.add_expense()
            elif input_choice == '2':
                self.view_expense()
            elif input_choice ==  '3':
                self.track_budget()
            elif input_choice == '4':
                self.save_expense()
            elif input_choice == '5':
                print('Thanks for using the tool.')
                print('---Closing---')
                self.save_expense()
                break
            else:
                print('Invalid choice, plesae select option b/w 1-5')


def main():
    "Initiliazing the Expenses Tracker:"
    tracker = ExpenseTracker()
    tracker.starter()

main();