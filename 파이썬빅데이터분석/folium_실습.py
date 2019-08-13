#folium_test.py
# folium : 지도 시각화
# googlemap : API key를 인증받아서 사용


import folium

Daejeon_city = folium.Map(location=[36.351781, 127.423237], ZOOM_START=19)

print(Daejeon_city)

Daejeon_city.save('Daejoen.html')
