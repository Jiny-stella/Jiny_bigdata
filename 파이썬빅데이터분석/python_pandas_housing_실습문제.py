
# 6
import pandas as pd

# sunspot = pd.read_csv('sunspots1.csv')
# date1 = sunspot['Date']
# pd.to_datetime(sunspot['Date'])
# sunspot['Date'] = pd.to_datetime(sunspot['Date'])
# print(sunspot.dtypes)
# print(sunspot[sunspot['Date']> sunspot['Date'].mean()])
#

import numpy as np

# a = np.arange(12).reshape(3,4)
# # print(a)
# #
# # # print(a[1:2,1:2])
# # # print(a[::2,::2])
# # print(a[::2,::2])
# # print(a[::3,::2])

# 입력기능
# while True:
#     select_list = {1:'입력',2:'출력',3:'검색',9:'종료중에 메뉴를 선택하쇼'}
#     print(select_list)
#     select = int(input('입력: '))
#     if select ==1 :
#     pro_name = input('제품명: ')
#     quantity = int(input('수량: '))
#     print(select)
#     index_list = ['입력값','제품명','수량','계속입력?']
#     going = input('Y/N: ')
#     if select == 2:
#         while going == 'Y':
#             index_list = ['입력값','제품명','수량','계속입력?']
#             ser = pd.Series([1,pro_name,quantity, going],index = index_list)
#             T_list = T_list.append(ser)
#             if going == 'N':
#                 break

#             print(T_list)

#1
# index_list=[]

ser = pd.Series([], index=[])
while True:
    select_list = {1: '입력', 2: '출력', 3: '검색', 9: '종료중에 메뉴를 선택하쇼'}
    print(select_list)
    select = int(input('입력: '))

    if select == 1:
        while True:
            pro_name = input('제품명: ')
            quantity = int(input('수량: '))
            ser= ser.append( pd.Series([quantity], index = [pro_name] ))
            going = input('Y/N:')
            if going == 'N':
                break

    if select == 2:
        print(ser)

    if select == 3:
        search = input('제품명: ')
        print(ser[ser.index.str.contains(search)])

    if select == 9:
        print('종료합니당')
        break


print('{0:=^50}'.format('2번'))

