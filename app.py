"""
Rossmann 매장 매출 예측 앱
--------------------------------
채은님의 XGBoost 모델을 불러와서, 매장 정보와 날짜를 입력하면
예상 매출을 보여주는 Streamlit 웹앱입니다.

실행 방법 (터미널에서):
    pip install streamlit xgboost pandas numpy
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import json
from datetime import date

@st.cache_resource
def load_model():
    model = xgb.XGBRegressor()
    model.load_model("rossmann_model.json")
    with open("feature_columns.json") as f:
        columns = json.load(f)
    return model, columns

@st.cache_data
def load_store_info():
    return pd.read_csv("store_info.csv")

model, feature_columns = load_model()
store_info = load_store_info()

st.set_page_config(page_title="Rossmann 매출 예측", page_icon="🛒")
st.title("🛒 Rossmann 매장 매출 예측")
st.caption("매장과 날짜 정보를 입력하면, 학습된 XGBoost 모델이 예상 매출을 예측합니다.")

st.divider()

col1, col2 = st.columns(2)

with col1:
    store_id = st.selectbox("매장 번호", sorted(store_info["Store"].unique()))
    selected_date = st.date_input("예측할 날짜", value=date(2015, 8, 1))
    promo = st.checkbox("오늘 프로모션 진행", value=True)

with col2:
    school_holiday = st.checkbox("학교 방학 기간")
    state_holiday = st.selectbox(
        "공휴일 종류",
        options=["0", "a", "b", "c"],
        format_func=lambda x: {"0": "평일", "a": "공식 공휴일", "b": "부활절", "c": "크리스마스"}[x],
    )

row = store_info[store_info["Store"] == store_id].iloc[0]

with st.expander("이 매장의 상세 정보 (자동 반영됨)"):
    st.write(row)

def build_input_row(store_id, selected_date, promo, school_holiday, state_holiday, store_row):
    day_of_week = selected_date.isoweekday()
    year = selected_date.year
    month = selected_date.month
    day = selected_date.day
    week_of_year = selected_date.isocalendar()[1]

    data = {
        "Store": store_id,
        "DayOfWeek": day_of_week,
        "Promo": int(promo),
        "SchoolHoliday": int(school_holiday),
        "CompetitionDistance": store_row["CompetitionDistance"],
        "CompetitionOpenSinceMonth": store_row["CompetitionOpenSinceMonth"],
        "CompetitionOpenSinceYear": store_row["CompetitionOpenSinceYear"],
        "Promo2": store_row["Promo2"],
        "Promo2SinceWeek": store_row["Promo2SinceWeek"],
        "Promo2SinceYear": store_row["Promo2SinceYear"],
        "CompetitionOpenKnown": store_row["CompetitionOpenKnown"],
        "Year": year,
        "Month": month,
        "Day": day,
        "WeekOfYear": week_of_year,
        "Holiday_0": 1 if state_holiday == "0" else 0,
        "Holiday_a": 1 if state_holiday == "a" else 0,
        "Holiday_b": 1 if state_holiday == "b" else 0,
        "Holiday_c": 1 if state_holiday == "c" else 0,
        "Type_a": 1 if store_row["StoreType"] == "a" else 0,
        "Type_b": 1 if store_row["StoreType"] == "b" else 0,
        "Type_c": 1 if store_row["StoreType"] == "c" else 0,
        "Type_d": 1 if store_row["StoreType"] == "d" else 0,
        "Assort_a": 1 if store_row["Assortment"] == "a" else 0,
        "Assort_b": 1 if store_row["Assortment"] == "b" else 0,
        "Assort_c": 1 if store_row["Assortment"] == "c" else 0,
    }
    return pd.DataFrame([data])[feature_columns]

st.divider()

if st.button("매출 예측하기", type="primary"):
    input_row = build_input_row(store_id, selected_date, promo, school_holiday, state_holiday, row)
    prediction = model.predict(input_row)[0]

    st.metric("예상 매출", f"{prediction:,.0f} 원")
    st.caption("※ 검증 데이터 기준 평균 오차율(MAPE) 약 14.8% — 참고용 추정치입니다.")

st.divider()
st.subheader("모델이 중요하게 본 변수 Top 10")
st.caption("XGBoost가 학습 과정에서 각 변수를 얼마나 자주, 효과적으로 활용했는지를 보여줍니다.")

importance = pd.Series(model.feature_importances_, index=feature_columns)
top10 = importance.sort_values(ascending=False).head(10)

st.bar_chart(top10.sort_values())
