'''
1. Add an expense:
• Create a function to prompt the user for expense details. Ensure you ask for:
o The date of the expense in the format YYYY-MM-DD
o The category of the expense, such as Food or Travel
o The amount spent
o A brief description of the expense
• Store the expense in a list as a dictionary, where each dictionary includes the
date, category, amount, and description as key-value pairs
Example:
{'date': '2024-09-18', 'category': 'Food', 'amount': 15.50, 'description':
'Lunch with friends'}
'''
import csv
from datetime import datetime
from typing import List, Dict

class ExpensesTracker:
    

    def __init__(self, csv_file= 'expenses.csv'):
        self.csv_file = csv_file
        self.expenses : List[Dict]= []
        self.budget = 0.0
        self.loadFile()


    def loadFile(self):
            try:
                with open(self.csv_file, 'r') as file:
                    data = csv.DictReader(file)
                    loaded_expenses = []
                    for row in data:
                        print(row)
                        loaded_expenses.append(row)
                
                    self.expenses = loaded_expenses
            except Exception as e:
                print('Error reading the expenses file, Starting fresh', e)


    # This is to add in the shared memory, not in file 
    def addExpenses(self, date, category, amount, description):
        print('Add expenses method')
        self.expenses.append({'date': date, 'category': category, 'amount': amount, 'description': description})
    

        '''
        2. View expenses:
        • Write a function to retr3ieve and display all stored expenses
        o Ensure the function loops through the list of expenses and displays the
        date, category, amount, and description for each entry
        • Validate the data before displaying it
        o If any required details (date, category, amount, or description) are
        missing, skip the entry or notify the user that it’s incomplete

        '''

    def viewExpenses(self):
        print('View expenses method')
        print(self.expenses)


        '''
        3. Set and track the budget:
        • Create a function that allows the user to input a monthly budget. Prompt the
        user to:
        o Enter the total amount they want to budget for the month
        • Create another function that calculates the total expenses recorded so far
        o Compare the total with the user’s monthly budget
        o If the total expenses exceed the budget, display a warning (Example:
        You have exceeded your budget!)
        o If the expenses are within the budget, display the remaining balance
        (Example: You have 150 left for the month)

        '''
    def setAndTrackBudget(self):
        print('---Budget Tracking---')

        if self.budget == 0:
            print('No allocated budget, please set budget before proceeding.')
            budget_inp = float(input('Enter your budget amount: '))
            self.budget = budget_inp
            return
        
        month = datetime.now().strftime('%Y-%m')
        monthlyExpenses = 0.0

        for expense in self.expenses:
            if expense['date'].startswith(month):
                amount = float(expense['amount'])
                monthlyExpenses += amount


        print(f'Monthly budget: {self.budget}')
        print(f'Monthly spends: {monthlyExpenses}')

        if monthlyExpenses > self.budget:
            print(f'Exceeded the monthly budget by: {monthlyExpenses - self.budget}')
        elif monthlyExpenses == self.budget:
            print('On Track')
        else:
            print(f'In the monthly budget remaining amount : {self.budget - monthlyExpenses}')
        




        '''
        4. Save and load expenses:
        • Implement a function to save all expenses to a CSV file, with each row
        containing the date, category, amount, and description of each expense
        • Create another function to load expenses from the CSV file. When the
        program starts, it should:
        o Read the saved data from the file
        o Load it back into the list of expenses so the user can see their previous
        expenses and continue from where they left off
        '''
    def saveExpenses(self):
        print('Save expenses method')
        # Open the file then iterate over the expenses and then writet them in the File.

        try:
            with open(self.csv_file, 'w') as File:
                headers = ['date', 'category', 'amount', 'description']
                writer = csv.DictWriter(File, fieldnames= headers)
                writer.writeheader()
                writer.writerows(self.expenses)


        except Exception as e: 
            print('Error while saving the expenses: ' , e)


        '''
        5. Create an interactive menu:
        • Build a function to display a menu with the following options:
        o Add expense
        o View expenses
        o Track budget
        o Save expenses
        o Exit
        • Allow the user to enter a number to choose an option
        • Implement the following conditions:
        o If the user selects option 1, call the function to add an expense
        o If the user selects option 2, call the function to view expenses
        o If the user selects option 3, call the function to track the budget
        o If the user selects option 4, call the function to save expenses to the file
        o If the user selects option 5, save the expenses and exit the program
        '''

    def run(self):

        while True:
            print('''
            =========Menu=========
            Choose your option from:
            1. Add new expense.
            2. View the expense.
            3. Track the Budget.
            4. Save the expense in the File.
            5. Exit.
            ''') 
            option = int(input('Enter your option to proceed: '))
            if option== 1:
                # Give more options to take the expences method parameters
                # o The date of the expense in the format YYYY-MM-DD
                # o The category of the expense, such as Food or Travel
                # o The amount spent
                # o A brief description of the expense
                date = input('Provide date of expense in (YYYY-MM-DD) format: ')
                category = input('Provide expense category: ')
                amount = input('Provide total amount: ')
                description = input('Provide description about the expense: ')
                self.addExpenses(date, category, amount, description)
                
            elif option == 2:
                self.viewExpenses()
                continue
            elif option == 3:
                self.setAndTrackBudget()
                continue
            elif option == 4:
                self.saveExpenses()
                continue
            elif option == 5:
                self.saveExpenses()
                break
            else: 
                print('Invalid option, please try again.')
                continue
            
        
def main():
    "Initiliazing the Expenses Tracker:"
    tracker = ExpensesTracker()
    tracker.run()

main();