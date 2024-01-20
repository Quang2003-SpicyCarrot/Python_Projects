#===Here is simple of Series===#
import pandas as pd
import numpy as np 
from datetime import datetime
test_data = [0.25, 0.5, datetime(2023,12,31), 1.0]
s = pd.Series(test_data)
print(s)

# How to change label in Series
s_new = pd.Series(data = test_data, index = [101, 102, 103, 104])
print(s_new)

#===Some operations in Series===#

# Get the first element
s4 = pd.Series([1, 2, 3, 4, 5], index = ['a', 'b', 'c', 'd', 'e'])
print(s4)
print('Method 1 : ', s4.iloc[0]) # By order number
print('Method 2 : ', s4.loc['a']) # By index value

# Get the first elements that come before line 3
print('From the beginning to before line 3 :')
print(s4.iloc[:2])

# Get elements by label
print("Elements with label 'a', 'c' : ")
print(s4.loc[['a', 'c']])

# Get the first 2 elements
print('First two elements :')
print(s4.head(2))
# Get the last 2 elements
print('Last two elements :')
print(s4.tail(2))

#===Change the value of an element in pandas.Series===#

# Create Series
s5 = pd.Series([1, 2, 3], index = ['a', 'b', 'c'])
print('Before : ')
print(s5)

# Assign a new value to the element with label 'b'
s5.loc['b'] = 100
print('After : ')
print(s5)

#===Add or remove elements to pandas.Series===#
# Create Series
s6 = pd.Series([1, 2, 3, 4], index = ['a', 'b', 'c', 'd'])
print('Previous : ')
print(s6)
# Add a new element with value 100 and label 'g'
s6.loc['g'] = 100
print('After : ')
print(s6)

#===Delete an existing element in pandas.Series===#
# We will use Series s6 above again
print('Before deletion: ')
print(s6)
# Delete the element with label 'a'
s6 = s6.drop(['a'])
print("After deleting the element with label 'a' : ")
print(s6)

#===Retrieve some information about any pandas.Series===#
# Create Series
s6 = pd.Series(np.random.randint(5))
# Get the size of the Series
print('Size : ', s6.size)
# Get the dimension of the Series
print('Number of dimensions : ', s6.ndim)
# Check if Series is empty
print('Is Empty : ', s6.empty)

#=== Some statistical methods ===#
#.count(): returns the number of elements other than NaN (Not a Number, a special value of pandas).
#.sum(): returns the sum of elements.
#.mean(): returns the average of elements.
#.median(): returns the median.
#.mode(): returns the mode (the element that appears the most times).
#.std(): returns standard deviation.
#.min(): returns the smallest value.
#.max(): returns the maximum value.
#.abs(): returns absolute value.
#.cumsum(): returns the cumulative total.
#.describe(): returns descriptive statistics.

# generate random data
data_1 = np.random.randint(10000, size = 365)
# create DatetimeIndex
index_1 = pd.date_range(start = '2019,01,01', periods = 365, freq = 'D')
# create Time Series
ts1 = pd.Series(data = data_1, index = index_1)
print(ts1)

# get data in May 2019
ts1['2019‐05']
