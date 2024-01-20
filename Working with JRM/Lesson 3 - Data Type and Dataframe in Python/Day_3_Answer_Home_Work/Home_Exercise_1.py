from random import sample
A = sample(range(10),4)
B = []
a = 0
print(A)
while len(set(A)) != 1:
    for i in range(len(A)-1):
        y = A[i+1] - A[i]
        B.append(y)
    print(B)
    A = B
    B = []
    a = a + 1
else:
    print(a)