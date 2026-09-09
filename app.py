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
    """서울 기온 자료를 불러와 연도별 평균기온 자료를 만든다."""
    df = pd.read_csv(DATA_URL, encoding="utf-8")

    # 날짜와 평균기온을 알맞은 자료형으로 변환
    df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")
    df["평균기온"] = pd.to_numeric(df["평균기온"], errors="coerce")

    # 날짜 또는 평균기온이 없는 행 제거
    df = df.dropna(subset=["날짜", "평균기온"]).copy()

    # 연도 열 생성
    df["연도"] = df["날짜"].dt.year

    # 수업 기준 기간: 2025년까지
    df = df[df["연도"] <= 2025]

    # 연도별 평균기온과 실제 관측일 수 계산
    yearly = (
        df.groupby("연도")
        .agg(
            연평균기온=("평균기온", "mean"),
            관측일수=("날짜", "nunique")
        )
        .reset_index()
    )

    # 관측일이 300일 미만인 해 제외
    yearly = yearly[yearly["관측일수"] >= 300].copy()

    # 연도순 정렬
    yearly = yearly.sort_values("연도").reset_index(drop=True)

    return yearly


def calculate_regression(data):
    """
    연도와 연평균기온을 이용하여 회귀식과 상관계수를 계산한다.
    반환값: 기울기, 절편, 상관계수
    """
    x = data["연도"].to_numpy()
    y = data["연평균기온"].to_numpy()

    slope, intercept = np.polyfit(x, y, 1)
    correlation = data["연도"].corr(data["연평균기온"])

    return slope, intercept, correlation


st.title("🌡️ 서울 기온 예측기")
st.write("서울의 연도별 평균기온 자료를 이용해 기온 변화 추세와 미래 기온을 예측합니다.")

try:
    yearly_data = load_and_process_data()
except Exception as e:
    st.error(f"데이터를 불러오는 중 오류가 발생했습니다: {e}")
    st.stop()

if len(yearly_data) < 2:
    st.error("회귀 분석에 필요한 연도별 데이터가 충분하지 않습니다.")
    st.stop()

# 전체 기간 회귀 분석
all_slope, all_intercept, all_correlation = calculate_regression(yearly_data)

start_year = int(yearly_data["연도"].min())
end_year = int(yearly_data["연도"].max())
number_of_years = len(yearly_data)

# 최근 20년: 마지막 분석 연도를 기준으로 최근 20개 달력 연도 사용
recent_start_year = end_year - 19
recent_data = yearly_data[yearly_data["연도"] >= recent_start_year].copy()

if len(recent_data) >= 2:
    recent_slope, recent_intercept, recent_correlation = calculate_regression(recent_data)
else:
    recent_slope = None
    recent_intercept = None
    recent_correlation = None

# 1년당 기울기를 100년당 변화량으로 변환
all_rise_per_100_years = all_slope * 100

if recent_slope is not None:
    recent_rise_per_100_years = recent_slope * 100
else:
    recent_rise_per_100_years = None

st.subheader("분석에 사용한 데이터")

info_col1, info_col2, info_col3 = st.columns(3)
info_col1.metric("회귀 직선을 만든 해의 개수", f"{number_of_years}개")
info_col2.metric("시작 연도", f"{start_year}년")
info_col3.metric("끝 연도", f"{end_year}년")

st.info(
    f"2025년 이후 자료와 관측일이 300일 미만인 해는 제외했습니다. "
    f"전체 기간의 연도와 연평균기온 상관계수는 **{all_correlation:.3f}**입니다."
)

st.subheader("📈 기온 상승 추세 비교")

slope_col1, slope_col2 = st.columns(2)

with slope_col1:
    st.metric(
        label=f"전체 기간 ({start_year}~{end_year}년)",
        value=f"{all_rise_per_100_years:+.2f} ℃",
        delta="100년에 변화하는 평균기온"
    )
    st.caption(f"분석에 사용한 연도 수: {number_of_years}개")

with slope_col2:
    if recent_rise_per_100_years is not None:
        st.metric(
            label=f"최근 20년 ({recent_start_year}~{end_year}년)",
            value=f"{recent_rise_per_100_years:+.2f} ℃",
            delta="100년에 변화하는 평균기온"
        )
        st.caption(f"분석에 사용한 연도 수: {len(recent_data)}개")
    else:
        st.warning("최근 20년 기울기를 계산할 데이터가 부족합니다.")

st.caption(
    "값이 양수(+)이면 기온이 올라가는 추세이고, "
    "음수(-)이면 기온이 내려가는 추세입니다."
)

# 연도 선택 슬라이더
selected_year = st.slider(
    "예상 기온을 확인할 연도를 선택하세요.",
    min_value=1900,
    max_value=2100,
    value=2025,
    step=1
)

# 전체 기간 회귀식으로 예상 기온 계산
predicted_temperature = all_slope * selected_year + all_intercept

st.subheader(f"📅 {selected_year}년 예상 평균기온")
st.metric(
    label="전체 기간 회귀 직선으로 예측한 연평균기온",
    value=f"{predicted_temperature:.2f} ℃"
)

# 회귀 직선용 연도: 슬라이더 전체 범위
line_years = np.arange(1900, 2101)
all_line_temperatures = all_slope * line_years + all_intercept

# Plotly 그래프 생성
fig = go.Figure()

# 실제 연도별 평균기온 산점도
fig.add_trace(
    go.Scatter(
        x=yearly_data["연도"],
        y=yearly_data["연평균기온"],
        mode="markers",
        name="실제 연평균기온",
        marker=dict(
            color="#1f77b4",
            size=8
        ),
        hovertemplate=(
            "연도: %{x}<br>"
            "연평균기온: %{y:.2f} ℃"
            "<extra></extra>"
        )
    )
)

# 전체 기간 회귀 직선
fig.add_trace(
    go.Scatter(
        x=line_years,
        y=all_line_temperatures,
        mode="lines",
        name="전체 기간 회귀 직선",
        line=dict(
            color="#e74c3c",
            width=3
        ),
        hovertemplate=(
            "연도: %{x}<br>"
            "예측기온: %{y:.2f} ℃"
            "<extra></extra>"
        )
    )
)

# 최근 20년 회귀 직선도 점선으로 표시
if recent_slope is not None:
    recent_line_years = np.arange(recent_start_year, 2101)
    recent_line_temperatures = (
        recent_slope * recent_line_years + recent_intercept
    )

    fig.add_trace(
        go.Scatter(
            x=recent_line_years,
            y=recent_line_temperatures,
            mode="lines",
            name="최근 20년 회귀 직선",
            line=dict(
                color="#27ae60",
                width=3,
                dash="dash"
            ),
            hovertemplate=(
                "연도: %{x}<br>"
                "최근 20년 기준 예측기온: %{y:.2f} ℃"
                "<extra></extra>"
            )
        )
    )

# 선택한 연도의 전체 기간 기준 예측값 표시
fig.add_trace(
    go.Scatter(
        x=[selected_year],
        y=[predicted_temperature],
        mode="markers+text",
        name="선택한 연도 예측값",
        marker=dict(
            color="#f39c12",
            size=15,
            symbol="star"
        ),
        text=[f"{predicted_temperature:.2f} ℃"],
        textposition="top center",
        hovertemplate=(
            f"선택 연도: {selected_year}<br>"
            f"예측기온: {predicted_temperature:.2f} ℃"
            "<extra></extra>"
        )
    )
)

fig.update_layout(
    title="서울 연도별 평균기온과 회귀 직선",
    xaxis_title="연도",
    yaxis_title="연평균기온 (℃)",
    template="plotly_white",
    hovermode="closest",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="center",
        x=0.5
    )
)

fig.update_xaxes(range=[1900, 2100])

st.plotly_chart(fig, use_container_width=True)

st.caption(
    f"전체 기간 회귀식: 예상 평균기온 = {all_slope:.5f} × 연도 + ({all_intercept:.2f})"
)

if recent_slope is not None:
    st.caption(
        f"최근 20년 회귀식: 예상 평균기온 = "
        f"{recent_slope:.5f} × 연도 + ({recent_intercept:.2f})"
    )

with st.expander("연도별 분석 데이터 보기"):
    display_data = yearly_data.copy()
    display_data["연평균기온"] = display_data["연평균기온"].round(2)
    st.dataframe(display_data, use_container_width=True)
