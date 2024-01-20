# Python code to append the content of one file to another

# File names
old_file = 'Part_1_Old.txt'
new_file = 'Part_1_NewText.txt'
merged_file = 'Merged_Part_1.txt'

# Reading the content of the new file
with open(new_file, 'r', encoding='utf-8') as file:
    new_content = file.read()

# Appending the content of the new file to the old file
with open(old_file, 'a', encoding='utf-8') as file:
    file.write('\n' + new_content)

# Creating a separate merged file
with open(old_file, 'r', encoding='utf-8') as file:
    old_content = file.read()

with open(merged_file, 'w', encoding='utf-8') as file:
    file.write(old_content + '\n' + new_content)

print(merged_file)

