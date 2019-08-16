# Python_data_type.py

# 단축키
# Ctrl + /  :  주석처림 및 해제

# Ctrl + shift + F10 : 새로 작성된 소스를 실행
# shift + F10 : 현재(기존 작업중) 작업중인 소스를 실행

# Tab : Indent , 들여쓰기
# Shift +  Tab : Unindent, 들여쓰기 해제

a = 12
b = 3.14
a, b = 12, 3.14
print(a, b)

c = 12, 3.14  # tuple
print(c)  # tuple 출력
print(c[0], c[1])

print('-' * 30)

# 연산자(operator) : 산술, 관계, 논리
# 산술 : + - * / // ** %
a, b = 13, 6
print(a + b)
print(a - b)
print(a * b)
print(a / b)  # 2.1666, 실수 정수 몫 나눗셈
print(a // b)  # 2  , 정수 몫 나눗셈
print(a ** b)
print(a % b)

# 문제 : 두 자리 양수를 거꾸로 뒤집어 보세요
# 29 -- > 92
n = 29
n = n % 10 * 10 + n // 10
print(n)

# 관계 : > >= < <= == !=
print(a, b)  # 13, 6
print(a > b)
print(a >= b)
print(a < b)
print(a <= b)
print(a == b)
print(a != b)

# 논리 : and  or not
# 다른언어 : && || !
print(True and True)
print(True and False)
print(False and True)
print(False and False)

# data tyepe
# list :  [ ],   mutable
a = [1, 3, 5, 7]
print(a[0], a[1], a[2], a[3], a[-1])  # indexing
print(a[1:3])  # slicing
a[0] = 10
a.append(8)
print(a)

# tuple : ( ) , immutable
a = (1, 3, 5, 7)
print(a[0], a[1], a[2], a[3], a[-1])  # indexing
print(a[1:3])  # slicing
# a[0] = 10  # Error!!

# dictionary : { }
d = {'name': 'kildong', 'age': 40}
print(d['name'], d['age'])

# for loop
for i in range(1, 11):
    print(i, end=' ')
print()
for i in range(1, 11, 2):
    print(i, end=' ')
print()
for i in range(11, 0, -1):
    print(i, end=' ')
print()
for i in range(11):
    print(i, end=' ')
print()

# 문제 0부터 99 까지의 정수를 한 줄에 10개씩 출력해보세요
for i in range(100):
    print(i, end=' ')

    if i % 10 == 9:
        print()


# 0 1 2 3 4 5 6 7 8 9
# 10 11 12 13 14 15 16 17 18 19
#  ...
# 90 91 92 93 94 95 96 97 98 99

# function : 함수

# 반환값이 없고 매개변수(인자) 없고
def f_1():
    print('f_1 is called!!')


f_1()


# 반환값이 없고 매개변수(인자) 있고
def f_2(a, b):
    print('f_2', a, b, a + b)


f_2(12, 34)


def f_22(a1, a2, a3='aaa'):
    print('f_22', a1, a2, a3)


f_2(12, 34)
f_2('hello', 'knu')
f_22('hello', 'knu')


# 반환값이 있고 매개변수(인자) 없고
def f_3():
    print('f_3')
    return 78

print(f_3())

# 반환값이 있고 매개변수(인자) 있고
def f_4(a,b):
    c = a + b
    return c

print(f_4(3,5))

# 문제 : 2개의 정수에 대해 큰 수, 작은 수의 순서로 반환하는 함수를 만드세요
# 3 5 --> 3 5
# 5 3 --> 3 5
def order(a,b):
    if a >= b :
        a,b = b,a
    return a,b
print(order(10,30))
print(order(30,10))
