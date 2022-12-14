import streamlit as st
from PIL import Image
from pygam import LinearGAM, s, f
import datetime
import pandas as pd
import numpy as np

st.set_page_config(layout = 'wide')

total = pd.read_csv('./chicken_total.csv')
X, y = total[total.columns[1:]].values, total['주문건수'].values
gam = LinearGAM(s(0, n_splines = 15) + s(1, n_splines = 15) + s(2, n_splines = 15) + f(3) + f(4) + s(5, n_splines = 15)).gridsearch(X, y)

def main():
    st.title("오늘의 치킨 판매양은?")
    selected_gu = st.selectbox("어느 구에서 장사하시나요? warning : 이외의 지역은 업데이트가 예정되어있습니다. 조금만 기다려주세요!!", ('구로구', '영등포구', '은평구'))
    selected_date = st.date_input("오늘의 날짜")
    selected_dday = selected_date - datetime.date(selected_date.year, 1, 1)
    selected_dday = selected_dday.days + 1
    selected_day_of_week = selected_date.weekday()
    selected_temp = st.number_input("오늘의 평균 기온(°C)")
    selected_rain = st.number_input("오늘의 평균 강수량(mm)")
    selected_wind = st.number_input("오늘의 평균 풍속(m/s)")

    gu = {'영등포구' : 0, '은평구' : 1, '구로구' : 2}
    input = np.array([[selected_temp, selected_rain, selected_wind, gu[selected_gu],  selected_day_of_week, selected_dday]])
    output = gam.predict(input)
    
    st.header(f"===== Date : {selected_date} =====")
    st.metric("Region", f"{selected_gu}")
    st.metric("Temperature", f"{selected_temp} °C")
    st.metric("Wind", f"{selected_wind} m/s")
    st.metric("Precipication", f"{selected_rain} mm")
    st.header("=========================")

    st.write("\n\n\n")
    col1, col2 = st.columns([3, 15])
    col1.image(Image.open("./chicken-leg.png"), width = 200)
    col2.header(f"오늘의 예측 배달 건수는 {int(output[0])}건 입니다.")


main()
