import os
import glob

# Assuming all files are in the same directory and follow the naming pattern "Part_*.txt"
directory = 'File_Path_Text'
file_pattern = 'Part_*.txt'
combined_file_path = 'Combined_Text.txt'

# Find all files that match the pattern
file_paths = glob.glob(os.path.join(directory, file_pattern))

# Sort the files to maintain the order (Part_1, Part_2, etc.)
file_paths.sort()

# Combine the contents of all files into one
with open(combined_file_path, 'w', encoding='utf-8') as combined_file:
    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as file:
            # Read and write the content to the combined file
            combined_file.write(file.read() + '\n')

combined_file_path  # Return the path of the combined file

