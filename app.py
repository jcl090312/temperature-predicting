# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(
    page_title="기온 예측기",
    page_icon="🌡️",
    layout="wide"
)

DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)


@st.cache_data
def load_and_process_data():
    # 데이터 불러오기
    df = pd.read_csv(DATA_URL, encoding="utf-8")

    # 날짜를 날짜 형식으로 변환
    df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")

    # 평균기온을 숫자 형식으로 변환
    df["평균기온"] = pd.to_numeric(df["평균기온"], errors="coerce")

    # 날짜와 평균기온이 없는 행 제거
    df = df.dropna(subset=["날짜", "평균기온"]).copy()

    # 연도 열 만들기
    df["연도"] = df["날짜"].dt.year

    # 수업 기준 기간: 2025년까지
    df = df[df["연도"] <= 2025]

    # 연도별 평균기온과 관측일 수 계산
    yearly = (
        df.groupby("연도")
        .agg(
            연평균기온=("평균기온", "mean"),
            관측일수=("날짜", "nunique")
        )
        .reset_index()
    )

    # 관측일이 300일 미만인 해 제거
    yearly = yearly[yearly["관측일수"] >= 300].copy()
    yearly = yearly.sort_values("연도").reset_index(drop=True)

    return yearly


st.title("🌡️ 서울 기온 예측기")
st.write("서울의 연도별 평균기온을 이용하여 선형 회귀 직선으로 미래 기온을 예측합니다.")

try:
    yearly_data = load_and_process_data()
except Exception as e:
    st.error(f"데이터를 불러오는 중 오류가 발생했습니다: {e}")
    st.stop()

if len(yearly_data) < 2:
    st.error("회귀 분석에 필요한 연도별 데이터가 충분하지 않습니다.")
    st.stop()

# 회귀 직선 계산: y = 기울기 × 연도 + 절편
x = yearly_data["연도"].to_numpy()
y = yearly_data["연평균기온"].to_numpy()

slope, intercept = np.polyfit(x, y, 1)
correlation = yearly_data["연도"].corr(yearly_data["연평균기온"])

start_year = int(yearly_data["연도"].min())
end_year = int(yearly_data["연도"].max())
number_of_years = len(yearly_data)

st.subheader("분석에 사용한 데이터")
col1, col2, col3 = st.columns(3)
col1.metric("회귀 직선을 만든 해의 개수", f"{number_of_years}개")
col2.metric("시작 연도", f"{start_year}년")
col3.metric("끝 연도", f"{end_year}년")

st.info(
    f"2025년 이후 자료와 관측일이 300일 미만인 해를 제외했습니다. "
    f"연도와 연평균기온의 상관계수는 **{correlation:.3f}**입니다."
)

# 연도 선택 슬라이더
selected_year = st.slider(
    "예상 기온을 확인할 연도를 선택하세요.",
    min_value=1900,
    max_value=2100,
    value=2025,
    step=1
)

predicted_temperature = slope * selected_year + intercept

st.subheader(f"📅 {selected_year}년 예상 평균기온")
st.metric(
    label="회귀 직선으로 예측한 연평균기온",
    value=f"{predicted_temperature:.2f} ℃"
)

# 회귀선 범위: 슬라이더 범위와 동일하게 1900~2100년
line_years = np.arange(1900, 2101)
line_temperatures = slope * line_years + intercept

# Plotly 그래프 만들기
fig = go.Figure()

# 산점도
fig.add_trace(
    go.Scatter(
        x=yearly_data["연도"],
        y=yearly_data["연평균기온"],
        mode="markers",
        name="실제 연평균기온",
        marker=dict(color="#1f77b4", size=8),
        hovertemplate="연도: %{x}<br>연평균기온: %{y:.2f} ℃<extra></extra>"
    )
)

# 회귀 직선
fig.add_trace(
    go.Scatter(
        x=line_years,
        y=line_temperatures,
        mode="lines",
        name="회귀 직선",
        line=dict(color="#e74c3c", width=3),
        hovertemplate="연도: %{x}<br>예측기온: %{y:.2f} ℃<extra></extra>"
    )
)

# 선택한 연도 표시
fig.add_trace(
    go.Scatter(
        x=[selected_year],
        y=[predicted_temperature],
        mode="markers+text",
        name="선택한 연도 예측값",
        marker=dict(color="#f39c12", size=15, symbol="star"),
        text=[f"{predicted_temperature:.2f} ℃"],
        textposition="top center",
        hovertemplate=(
            f"선택 연도: {selected_year}<br>"
            f"예측기온: {predicted_temperature:.2f} ℃<extra></extra>"
        )
    )
)

fig.update_layout(
    title="서울 연도별 평균기온과 회귀 직선",
    xaxis_title="연도",
    yaxis_title="연평균기온 (℃)",
    template="plotly_white",
    hovermode="closest",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
)

fig.update_xaxes(range=[1900, 2100])

st.plotly_chart(fig, use_container_width=True)

st.caption(
    f"회귀식: 예상 평균기온 = {slope:.5f} × 연도 + ({intercept:.2f})"
)

with st.expander("연도별 분석 데이터 보기"):
    display_data = yearly_data.copy()
    display_data["연평균기온"] = display_data["연평균기온"].round(2)
    st.dataframe(display_data, use_container_width=True)
