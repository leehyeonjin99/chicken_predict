import streamlit as st
from PIL import Image
from pygam import LinearGAM, s, f
import datetime
import pandas as pd
import numpy as np

st.set_page_config(layout = 'wide')

total_seoul = pd.read_csv('./total_seoul.csv')
total_seoul = total_seoul.drop('평균 풍속(m/s)', axis = 1)
print(total_seoul.columns)
X, y = total_seoul[total_seoul.columns[1:]].values, total_seoul['주문건수'].values
gam = LinearGAM(s(0, n_splines = 15) + s(1, n_splines = 15) + f(2) + s(3, n_splines = 15)).gridsearch(X, y)

def main():
    st.title("오늘의 치킨 판매양은?")
    selected_date = st.date_input("오늘의 날짜")
    selected_dday = selected_date - datetime.date(selected_date.year, 1, 1)
    selected_dday = selected_dday.days + 1
    selected_day_of_week = selected_date.weekday()
    selected_temp = st.number_input("오늘의 평균 기온(°C)")
    # selected_wind = st.number_input("오늘의 평균 풍속(m/s)")
    selected_rain = st.number_input("오늘의 평균 강수량(mm)")

    input = np.array([[selected_temp, selected_rain, selected_day_of_week, selected_dday]])
    output = gam.predict(input)
    
    st.header(f"===== Date : {selected_date} =====")
    st.metric("Temperature", f"{selected_temp} °C")
    # st.metric("Wind", f"{selected_wind} m/s")
    st.metric("Precipication", f"{selected_rain} mm")
    st.header("=========================")

    st.write("\n\n\n")
    col1, col2 = st.columns([3, 15])
    col1.image(Image.open("./chicken-leg.png"), width = 200)
    col2.header(f"오늘의 예측 배달 건수는 {int(output[0])}건 입니다.")


main()