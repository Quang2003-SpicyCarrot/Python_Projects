def Hello_World(): # Function print welcome screen after run program
    hello_world = 'Hello World'
    print(hello_world)

def Close_Program(): # Function before program close
    close_program = 'Here is Our Data'
    print(close_program)

def Main_Input(): # Function collect all input we need when typing and calculating
    Number_Employee = 0 # Before typing input, there are no data so far
    while True: 
        Number_Employee = Number_Employee + 1 # Record how many employee in input data
        print('Number of Employee: ', Number_Employee)
        Input_Name_Employee() # Function to typing and recording input employee name
        List_Input_Hours_Work() # Function to typing and recording input hours worked each employee
        List_Input_Hours_Rate() # Function to typing and recording input hours rate each employee
        continue_input = input('Continue Enter Input Information? \n Press C to calculate all input \n Press other key to continute enter input')
        # This line above to show if user want to enter new employee's information
        if continue_input == 'C' or continue_input == 'c':
            Calculate_Function() # Function to calculate all input
            Close_Program() # Funciton print close program
            break

def Input_Name_Employee(): # Function to typing and recording input employee name
    while True:
        user_input_name_employee = input('Enter Employee Name: ') # Enter input employee's name
        if any(char.isdigit() for char in user_input_name_employee) == True or len(user_input_name_employee) == 0:
            # Checking if employee's name is not valid
            print('Not a Valid Name, Please Enter Right Name')
        else:
            # Save input
            List_Name_Employee.append(user_input_name_employee)
            break

def List_Input_Hours_Work(): # Function to typing and recording input hours worked each employee
    while True:
        user_input_hours_work = input('Enter Hours Work: ') # Enter input employee's hours worked
        if all(char.isdigit() for char in user_input_hours_work) == False or len(user_input_hours_work) == 0:
            # Checking if employee's hours worked is not valid
            print('Not a Valid Hours Work, Please Enter Number')
        else:
            # Save input
            List_Hours_Work.append(user_input_hours_work)
            break

def List_Input_Hours_Rate(): # Function to typing and recording input hours rate each employee
    while True:
        user_input_hours_rate = input('Enter Hours Rate: ') # Enter input employee's hours rate
        if all(char.isnumeric() for char in user_input_hours_rate) == False or len(user_input_hours_rate) == 0:
            print('Not a Valid Hours Rate, Please Enter Number')
            # Checking if employee's hours rate is not valid
        else:
            List_Hours_Rate.append(user_input_hours_rate)
            break

def Calculate_Function(): # Function to calculate all input
    print('Calcualte Processing')
    Order_Value = 0 # Before start calculate processing, there are no data so far
                    # and first data start at 0 position in list
    Number_Employee = len(List_Name_Employee) # Count how many rows input when user typing
    while True:
        if Number_Employee - Order_Value != 0: # Checking if all employee input data is processing
            # Calculate employe's income
            Income = float(List_Hours_Work[Order_Value]) * float(List_Hours_Rate[Order_Value])
            # Save employee's income
            List_Income.append(Income)
            # Calculate employee's income tax deduction
            Income_Tax_Deduction = Income * (fixed_tax_rate /100)
            # Save employee's income tax deduction
            List_Income_Tax_Deduction.append(round(Income_Tax_Deduction,2))
            # Calculate employee's superannuation deduction
            Superannuation_Deduction = Income * (superannuation_tax /100)
            # Save employee's superannuation deduction
            List_Superannuation_Deduction.append(round(Superannuation_Deduction,2))
            # Calculate next employee's data
            Order_Value = Order_Value + 1
        else:
            # Display all result running this program
            Display_Information()
            break

def Display_Information():
    # Data input
    data_input = [Header_Input] + list(zip(List_Name_Employee, List_Hours_Work, List_Hours_Rate))
    # Data output
    data_output = [Header_Output] + list(zip(List_Name_Employee, List_Income, List_Income_Tax_Deduction, List_Superannuation_Deduction ))
    
    # Display data input
    print('\n Input Information' )
    for row_index, row_data in enumerate(data_input):
        line = '||'.join(str(column).ljust(12) for column in row_data)
        print(line)
        if row_index == 0:
            print('_' * len(line))
    
    # Display data input
    print('\n Output Information' )
    for row_index, row_data in enumerate(data_output):
        line = '||'.join(str(column).ljust(12) for column in row_data)
        print(line)
        if row_index == 0:
            print('_' * len(line))


def Running_Program(): # Start program function
    Hello_World() # Welcome screen
    user_input_run_program = input('Do you want to start my program \n Press Y for YES \n Press other key for NO: ') #3
    if user_input_run_program == 'Y' or user_input_run_program == 'y':
        return Main_Input() # If user want to use program
    else: # If user not want to use program
        return Close_Program()
    
# List Input and Column Name
List_Hours_Work = []
List_Name_Employee = []
List_Hours_Rate = []

# List Output and Column Name
List_Income = []
List_Income_Tax_Deduction = []
List_Superannuation_Deduction = []

# Header Column
Header_Input = ['Name Employee', 'Hours Worked', 'Hourly Rate']
Header_Output = ['Name Employee', 'Income', 'Income Tax Deduction', 'Superannuation Deduction']

# Rage Tax By Precentage
fixed_tax_rate = 20
superannuation_tax = 10

Running_Program() # This line will start program
