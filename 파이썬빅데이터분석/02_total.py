import pandas as pd
# 파일 불러오기
data1 = pd.read_csv('data.csv')
print(data1)

# 필요 컬럼 추출
# print(data1.columns) #컬럼이름검색
data2 = data1[['기준년월','사업체id','건물번호','공정위업종중분류명칭','사업체명칭','공정위업종중분류코드','시군구명칭','행정동명칭','유동인구합계','주간상주인구_합계','야간상주인구_합계','예상매출액']]
print(data2)

# 건물 별로 데이터 묶기
df_buil = data2.groupby('건물번호')
print(df_buil.count())
print(df_buil.describe())

# 컬럼 조건 검색 코드
data_x = data2[(data2['건물번호'] == '3014010100100660009013799')]
print(data_x)

data_y = data2[(data2['사업체id'] == 1547966)]
print(data_y)

# 그래프 그리기
import numpy as np
import matplotlib.pyplot as plt

# 그래프 그리기 - 건물별 데이터
df_buil = df_buil.describe()
df_buil = pd.DataFrame(df_buil)
df_buil = df_buil.replace(to_replace = '_', value = 0, regex = True)

# 데이터 타입 변경
df_buil = df_buil.astype({'예상매출액':float})

# 매출액
df_buil = df_buil[df_buil['예상매출액']< 50000]

plt.boxplot(df_buil, labels = '예상매출액')
plt.show()

# 유동인구와 건물별 매출의 상관관계
df_popul = data2.groupby(['건물번호']).describe()
df_popul= df_popul[['유동인구합계','야간상주인구_합계']]
df_popul = df_popul.iloc[[1,9]]
print(df_popul)

