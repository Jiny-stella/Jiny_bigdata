# 05-4_예외처리_외장함수.PY

# 예외(Exception)
# try :
'''
except : <예외종류> :
'''
'''
finally:
'''


# c = a/b   # ZeroDivisionError: division by zero
# d = a + value     # NameError: name 'value' is not defined
# e = 'Hi' + 5      # TypeError: can only concatenate str (not "int") to str
# f = open('hello.txt', 'r')      # FileNotFoundError: [Errno 2] No such file or directory: 'hello.txt'

a,b = 5,0
try:
     c = a / b
except ZeroDivisionError:  #exept 종류를 모르겠다하면은 비워놔도 됨
    print(b,'로 나누었습니다.(ZeroDivisionError)')
    pass

try:
    d = a + value
except NameError:
    print('잘못된 변수를 사용하였습니다')
    pass

score = '결석'
try:
    e = 'score' + a
except TypeError:
    e = 0 + a
    print(e)
finally:
    print('finally:',0)

try:
    f = open('hello.txt,','r')
except FileNotFoundError:
    pass


try:
    import a_module
except ModuleNotFoundError:
    print(' a_module이 없습니다(ModuleNotFoundError)')

print('정상종료되었습니다.')

#예외 종료를 출력
def classtree(cls, indent = 0):
    print(',' * indent, cls.__name__)
    for subcls in cls.__subclasses__():
        classtree(subcls, indent = 3)

print(classtree(BaseException))

#재귀함수
# def add(a,b):
#     c = a + b
#     add(a,c)
# print(add(1,2))

# 예외 만들기
# raise<예외종류> : 예외를 강제로 발생시킴
try:
    raise FileNotFoundError
except FileNotFoundError:
    print('미안')
    pass

# 사용자 예외 만들기
class MyError(Exception):
    pass

def say_nick(nick):
    if nick == '바보':
        raise MyError()
    print(nick)
try:
    say_nick('천사')
    say_nick('얼짱')
    say_nick('바보')
except MyError:
    print('제발 바보라고 부르지 마세요(MyError)')
    pass
print('정상종료되었습니다.')

# 파이썬 외장함수
# sys 모듈 : 터미널 명령모드에서 인자 정보와 시스템 경로 설정

# import sys
# print(sys.argv)
#터미널에서 아래와 같이 실행
# python 05-4_예외처리_외장함수.py hello python sys module
# ['05-4_예외처리_외장함수.py', 'hello', 'python', 'sys', 'module']
# print(sys.argv[0])  # 05-4_예외처리_외장함수.py
# print(sys.argv[1])  #hello

# print(sys.path)
# sys.path.append('C:/myfolder/hello/')     # 시스템 환경변수에 경로


# os 모듈 : 파일, 디렉토리, 환경변수, os 의 자원

import os
# os.system("dir")    # MS -DOS 명령어를 직접 실행 할 수 있다.
# os.system("cls")

# Time 모듈 : 시간 관련 모듈

# import time
# print(time.time())
# start_time = time.time()
# time.sleep(2)   # 지연 함수
# for k in range(1000):
#     print(k,end=' ')
# end_timne = time.time()
# print()
#
# duration = end_timne - start_time
# print(duration,'(sec)')
#
# print(time.ctime())    # Tue Jul 16 11:14:23 2019
#
# time_now = time.ctime()
# print(time_now)            # Tue Jul 16 11:14:23 2019
# only_time = time_now[11:19]
# print(only_time)
#
# print(time.strftime('%Y-%b-%m-%d',time.localtime(time.time())))
#
# time.sleep(0.1)
# print('10ms 가 경과 되었습니다.')
#
# import calendar
# print(calendar.calendar(2019))
# print(calendar.calendar(2020,7))
# calendar.prmonth(2020,7)
# print(calendar.monthrange(2019,7))


#random 모듈
import random

# random()함수    : 0 < = num <
num = random.random()
print(num)

for k in range(100):
    num = round(random.random(),3)
    print(num, end = ',')
print()

# random.seed(3)   # seed값을 고정시키면 항상 동일한 값을 생성
num = random.random()
print(num)

num = random.randint(1,24)
print('int:', num)


#choice(시퀀스형)
mylist = [1,2,4,6,7,8,9]
mylist2 = ['b','a,','d','q','r']

num = random.choice(mylist)
string = random.choice(mylist2)
print(num,string)

# sample(시퀀스형, 갯수) : 중복되지 않는 요소를 추출
num_list = random.sample(mylist,3)
string_list = random.sample(mylist2,3)
print(num_list,string_list)

# shuffle()
num_shuffle = random.shuffle(mylist)
print(mylist)
