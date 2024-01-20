# 1.Text file in Python
# 1.1. Open Python file using this:
# with open(<file_part>, <mode>) as <variable_name>:
#	<Action_1>
#   <Action_2>

# <mode>: 
# Read Only (‘r’)
# Read and Write (‘r+’)
# Write Only (‘w’)
# Write and Read (‘w+’)
# Append Only (‘a’)
# Append and Read (‘a+’)

# 1.2. Read text file in Python

with open ('Part_1.txt', 'r', encoding = 'utf-8') as P1: #this is only for read file only "r", encoding can be 'utf-8' or latin or more
    # Compare readline() and readlines()
    #print(P1.readline())
    #print(P1.readlines())

    # Or using this method to read all line in one text file
    for line in P1:
        print(line)

# 1.3. Write down in text file with Python
# Using this: <variable_name>.write(<content>)
        
# This line code is just show old text
with open ('Part_1_Old.txt', 'a+', encoding = 'utf-8') as P1_old:
    P1_old.write('''Sóng bắt đầu từ gió
Gió bắt đầu từ đâu?
Em cũng không biết nữa
Khi nào ta yêu nhau''')
    for line in P1_old:
        print(line)

# If you want to see both old text and new text, you need to run code twice
# But you can not append new text in Part_1_Old.txt, your new text is in Part_1, why?
        
# 1. Combine Part_1_NewText into Part_1_Old and export new file 'Merged_Part_1.txt'
# More example problem in Solution_2

