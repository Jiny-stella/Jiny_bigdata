from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import urllib.request as REQ
import urllib
# import json
# import folium

# 1.1
print('1.1','-'*50)
baseurl= 'http://www.menupan.com'
murl = '/restaurant/bestrest/bestrest.asp?pt=rt&areacode=dj201'

response = REQ.urlopen(baseurl+murl)
soup = BeautifulSoup( response, 'html.parser')

listData=[]
href=[]

ulList = soup.select_one( '.rankingList .list')
for li in ulList.select('li'):
    rank = li.select_one('.numTop,.rankNum').string
    href.append( li.select_one('.listName a')['href'] )
    listName = li.select_one('.listName').string
    listType = li.select_one('.listType').string
    listArea = li.select_one('.listArea').string
    listData.append({'랭킹':rank,
                     "상호명":listName,
                     "업종":listType,
                     "지역":listArea})

df= pd.DataFrame(listData)
print(df)


# 1.2
print('1.2','-'*50)
tel=[]
addr=[]
for h in href:
    response = REQ.urlopen(baseurl+h)
    sMenu = BeautifulSoup( response, 'html.parser')
    stel = sMenu.select_one('.tel1').string
    saddr = sMenu.select_one('.add1').string
    tel.append( stel )
    addr.append(saddr)
    # print(stel, saddr)

df['전화번호'] = tel
df['주소']=addr
print(df)

# 1.3
print('1.3','-'*50)
df.set_index('랭킹',inplace=True)
# inplace : bool, default False
# Modify the DataFrame in place (do not create a new object).

print(df)
df.to_excel("menu.xlsx")


#  1.4
print('1.4','-'*50)
m = df[ df['업종'].str.contains('한식')]
print(m)


#  1.5
print('1.5','-'*50)
area = df[ df['지역'].str.contains('대흥')]
print(area)

# 1.6
print('1.6','-'*50)
import googlemaps
gmaps_key = "AIzaSyCcw89SLkOaJ5r8xoL3aFkUx0kpu7baLRg" # 자신의 key를 사용합니다.
gmaps = googlemaps.Client(key=gmaps_key)

def get_lat_lng(addr):
    area = gmaps.geocode(addr, language='ko')
    latlng = area[0].get("geometry")
    lat = latlng['location']['lat']
    lng = latlng['location']['lng']
    # print(lat,lng)
    return lat,lng


import folium

lat, lng = get_lat_lng(df['주소'].values[0])
map = folium.Map(location=[lat, lng], zoom_start=12)
for addr in range(len(df['주소'].values)):
    lat, lng = get_lat_lng(df['주소'].values[addr])
    #     print(lat,lng)
    #     print(df['주소'].values[addr])
    #     print(df['상호명'].values[addr])
    m = folium.Marker([lat, lng], popup=str(addr) + df['상호명'].values[addr],
                      icon=folium.Icon(icon='cloud')).add_to(map)
    m.add_to(map)

map.save('Daejoen_맛집.html')
