# Understand Class in Python
# To understand the meaning of classes we have to understand the built-in __init__() function.
# All classes have a function called __init__(), which is always executed when the class is being initiated.

# Use the __init__() function to assign values to object properties,
# or other operations that are necessary to do when the object is being created

class Person:
  def __init__(self, name, age):
    self.name = name
    self.age = age

p1 = Person("John", 36)

print(p1.name)
print(p1.age)

#The __str__() function controls what should be returned when the class object is represented as a string.
# If the __str__() function is not set, the string representation of the object is returned:

class Person_2:
  def __init__(self, name, age):
    self.name = name
    self.age = age

# Delete 2 line code below to compare with and without __str__
  def __str__(self):
    return f"{self.name}({self.age})"

p2 = Person_2("John", 36)

print(p2)

# Combine function and class in Python
# Another example:
class Person_3:
  def __init__(self, name, age, gender):
    self.name = name
    self.age = age
    self.gender = gender

  def myfunc(self):
    if self.gender == 'Male':
      gender = 'He'
    else:
      gender = 'She'
    print(gender, 'is', self.name, 'and', gender + ''''s ''', self.age)

p3 = Person_3("Jimy", 30, 'Female')
p3.myfunc()
p3 = Person_3("Adam", 35, 'Male')
p3.myfunc()

# More things about class in Python: Object-Oriented Programming (OOP)