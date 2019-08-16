print('hello python!!')
a = 10.2
b = 3.2
c = a + b
print(c)

a, b = 7 , 3
print(a, b)

# + - * / ** // %
print(a + b)
print(a - b)
print(a * b)
print(a / b)

print(a ** b)
print(a // b)
print(a % b)

# > >= < <= == !=
print( a > b)
print( a < b)
print( a >= b)
print( a <= b)
print( a == b)
print( a != b)


# data tyepe
# list :  [ ],   mutable
a = [1,3,5,7]
print(a[0],a[1],a[2],a[3],a[-1])  # indexing
print(a[1:3])           # slicing
a[0] = 10
print(a)


# tuple : ( ) , immutable
a = (1,3,5,7)
print(a[0],a[1],a[2],a[3],a[-1])  # indexing
print(a[1:3])           # slicing
#a[0] = 10  # Error!!

# dictionary : { }
d = { 'name':'kildong', 'age':40}
print(d['name'], d['age'])

# loop : for
s = ''
for i in range(1,10) :   # 1-9
    s += str(i) + ' '
    print(s)
    
# function
def func_1(a,b) :
    c = a + b
    return c

result = func_1(10,20)
print(result)

