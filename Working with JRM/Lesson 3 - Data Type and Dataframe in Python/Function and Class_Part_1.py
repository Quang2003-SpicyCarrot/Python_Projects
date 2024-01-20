# 1. Function in Python
# A function is a block of code which only runs when it is called.
# You can pass data, known as parameters, into a function.
# A function can return data as a result.

# Example: 
def my_function():
  print("Hello from a function")

my_function()

# Arguments function in Python
# Example:
def my_name(name, number_day):
  print('My name is', name)
  print('We have', number_day, 'workshop of Data Analyst with Python')

my_name('Tuong', 3)

# 2. Class in Python
#Python is an object oriented programming language.
# Almost everything in Python is an object, with its properties and methods.
# A Class is like an object constructor, or a "blueprint" for creating objects.

class MyClass:
  x = 5

p1 = MyClass()
print(p1.x)