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

# ---------------------------------------------------------
# 1. 모델과 참고 데이터 불러오기
#    @st.cache_resource: 앱이 새로고침 될 때마다 모델을 다시
#    불러오지 않고, 한 번만 불러온 걸 재사용하게 해줌 (속도 개선)
# ---------------------------------------------------------
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

# ---------------------------------------------------------
# 2. 화면 구성
# ---------------------------------------------------------
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

# 선택한 매장의 정보를 store_info.csv에서 자동으로 가져옴
row = store_info[store_info["Store"] == store_id].iloc[0]

with st.expander("이 매장의 상세 정보 (자동 반영됨)"):
    st.write(row)

# ---------------------------------------------------------
# 3. 입력값을 모델이 학습했던 것과 '똑같은 형태'로 변환
#    -> 이게 실수하기 제일 쉬운 부분! 학습 때 만든 컬럼과
#       순서/이름이 정확히 일치해야 모델이 제대로 예측함
# ---------------------------------------------------------
def build_input_row(store_id, selected_date, promo, school_holiday, state_holiday, store_row):
    day_of_week = selected_date.isoweekday()  # 1=월 ... 7=일
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
        # StateHoliday 원-핫 인코딩 (9교시에서 배운 방식)
        "Holiday_0": 1 if state_holiday == "0" else 0,
        "Holiday_a": 1 if state_holiday == "a" else 0,
        "Holiday_b": 1 if state_holiday == "b" else 0,
        "Holiday_c": 1 if state_holiday == "c" else 0,
        # StoreType 원-핫 인코딩 (선택한 매장의 실제 유형 기준)
        "Type_a": 1 if store_row["StoreType"] == "a" else 0,
        "Type_b": 1 if store_row["StoreType"] == "b" else 0,
        "Type_c": 1 if store_row["StoreType"] == "c" else 0,
        "Type_d": 1 if store_row["StoreType"] == "d" else 0,
        # Assortment 원-핫 인코딩
        "Assort_a": 1 if store_row["Assortment"] == "a" else 0,
        "Assort_b": 1 if store_row["Assortment"] == "b" else 0,
        "Assort_c": 1 if store_row["Assortment"] == "c" else 0,
    }
    # 학습 때 썼던 컬럼 순서 그대로 정렬
    return pd.DataFrame([data])[feature_columns]

st.divider()

if st.button("매출 예측하기", type="primary"):
    input_row = build_input_row(store_id, selected_date, promo, school_holiday, state_holiday, row)
    prediction = model.predict(input_row)[0]

    st.metric("예상 매출", f"{prediction:,.0f} 원")
    st.caption("※ 검증 데이터 기준 평균 오차율(MAPE) 약 14.8% — 참고용 추정치입니다.")
