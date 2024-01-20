import pandas as pd 

pd.DataFrame({
'Product' : ['Apple', 'Banana', 'Cherry'],
'Quantity' : [12, 34, 56],
'Price' : [10, 5, 8]
})

# Create dataframe from list of list
ll = [
['Apple', 100],
['Banana', 25],
['Cherry', 36]
]
# create DataFrame
print(pd.DataFrame(ll))

# Create DataFrame, specifying index and columns
new_dataframe = pd.DataFrame(ll, index = ['a', 'b', 'c'], columns = ['Product', 'Quantity'])
print(new_dataframe)

# Create dataframe from dictionary
d1 = {
'col1' : ['Apple', 'Banana'],
'col2' : [1, 2],
'col3' : ['2019-10-02', '2019-11-01']
}
# create DataFrame
data = pd.DataFrame(d1)
print(data)
