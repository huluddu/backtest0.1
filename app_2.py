# ✅ 완성형: 안정 기반에 전체 기능 통합 (시그널 체크 + 백테스트 + 그리드서치)

import streamlit as st
import pandas as pd
import yfinance as yf
import datetime
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import itertools
import random
from pykrx import stock


def get_mdd(asset_curve):
    peak = asset_curve.cummax()
    drawdown = (asset_curve - peak) / peak
    return drawdown.min() * 100


def get_krx_data(ticker, start_date, end_date):
    df = stock.get_etf_ohlcv_by_date(start_date.strftime("%Y%m%d"), end_date.strftime("%Y%m%d"), ticker)
    df = df[["종가"]].reset_index()
    df.columns = ["Date", "Close"]
    return df


def get_data(ticker, start_date, end_date):
    try:
        if ticker.lower().endswith(".ks") or ticker.isdigit():
            return get_krx_data(ticker, start_date, end_date)
        else:
            df = yf.download(ticker, start=start_date, end=end_date)

            # ✅ MultiIndex일 경우 보정
            if isinstance(df.columns, pd.MultiIndex):
                if ("Close", ticker.upper()) in df.columns:
                    df = df[("Close", ticker.upper())]
                elif "Close" in df.columns.get_level_values(0):
                    df = df["Close"]
                df = df.to_frame(name="Close")
            elif isinstance(df, pd.Series):
                df = df.to_frame(name="Close")

            # ✅ 마지막으로 Close만 남기고 정리
            df = df[["Close"]].dropna().reset_index()
            return df
    except Exception as e:
        st.error(f"❌ 데이터 로딩 실패: {e}")
        return pd.DataFrame()


def check_signal_today(df, ma_buy, offset_ma_buy, ma_sell, offset_ma_sell,
                       offset_cl_buy, offset_cl_sell,
                       ma_compare_short=None, ma_compare_long=None,
                       offset_compare_short=1, offset_compare_long=1):

    df = df.copy()
    df = df.sort_values("Date").reset_index(drop=True)
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df["MA_BUY"] = df["Close"].rolling(ma_buy).mean()
    df["MA_SELL"] = df["Close"].rolling(ma_sell).mean()

    if ma_compare_short and ma_compare_long:
        df["MA_SHORT"] = df["Close"].rolling(ma_compare_short).mean()
        df["MA_LONG"] = df["Close"].rolling(ma_compare_long).mean()

    i = -1
    try:
        cl_b = float(df["Close"].iloc[i - offset_cl_buy])
        ma_b = float(df["MA_BUY"].iloc[i - offset_ma_buy])
        cl_s = float(df["Close"].iloc[i - offset_cl_sell])
        ma_s = float(df["MA_SELL"].iloc[i - offset_ma_sell])
        ref_date = df["Date"].iloc[i].strftime('%Y-%m-%d')
    except Exception as e:
        st.warning(f"❗오늘 시그널 판단에 필요한 데이터가 부족합니다: {e}")
        return

    st.subheader("📌 오늘 시그널 판단")
    st.write(f"📆 기준일: {ref_date}")

    trend_ok = True
    trend_msg = "비활성화"
    if ma_compare_short and ma_compare_long:
        try:
            ma_short = float(df["MA_SHORT"].iloc[i - offset_compare_short])
            ma_long = float(df["MA_LONG"].iloc[i - offset_compare_long])
            trend_ok = ma_short >= ma_long
            trend_msg = f"{ma_short:.2f} vs {ma_long:.2f} → {'통과 ✅' if trend_ok else '미통과 ❌'}"
        except:
            trend_msg = "❗데이터 부족"
            trend_ok = False

    st.write(f"📈 추세 조건: {trend_msg}")

    buy_ok = cl_b > ma_b and trend_ok
    sell_ok = cl_s < ma_s

    st.write(f"💡 매수판단: 종가({cl_b:.2f}) {'>' if cl_b > ma_b else '<='} MA({ma_b:.2f}) → {'매수조건 ✅' if buy_ok else '조건부족 ❌'}")
    st.write(f"💡 매도판단: 종가({cl_s:.2f}) {'<' if cl_s < ma_s else '>='} MA({ma_s:.2f}) → {'매도조건 ✅' if sell_ok else '조건부족 ❌'}")

    if buy_ok:
        st.success("📈 오늘은 매수 시그널입니다!")
    elif sell_ok:
        st.error("📉 오늘은 매도 시그널입니다!")
    else:
        st.info("⏸ 매수/매도 조건 모두 만족하지 않음")

    last_buy_date = None
    last_sell_date = None

    for j in range(len(df) - max(offset_cl_buy, offset_ma_buy), 0, -1):
        try:
            cb = df["Close"].iloc[j - offset_cl_buy]
            mb = df["MA_BUY"].iloc[j - offset_ma_buy]
            cs = df["Close"].iloc[j - offset_cl_sell]
            ms = df["MA_SELL"].iloc[j - offset_ma_sell]

            trend_pass = True
            if ma_compare_short and ma_compare_long:
                ms_short = df["MA_SHORT"].iloc[j - offset_compare_short]
                ms_long = df["MA_LONG"].iloc[j - offset_compare_long]
                trend_pass = ms_short >= ms_long

            if last_buy_date is None and cb > mb and trend_pass:
                last_buy_date = df["Date"].iloc[j]
            if last_sell_date is None and cs < ms:
                last_sell_date = df["Date"].iloc[j]

            if last_buy_date and last_sell_date:
                break
        except:
            continue

    if last_buy_date:
        st.write(f"🗓 마지막 매수 조건 만족: {last_buy_date.strftime('%Y-%m-%d')}")
    if last_sell_date:
        st.write(f"🗓 마지막 매도 조건 만족: {last_sell_date.strftime('%Y-%m-%d')}")
    if not last_buy_date and not last_sell_date:
        st.warning("❗최근 매수/매도 조건에 부합한 날이 없습니다.")

# ✅ 전략 프리셋 목록 정의
PRESETS = {
    "SOXL 최고 전략": {
        "ma_buy": 25, "offset_ma_buy": 1, "offset_cl_buy": 25,
        "ma_sell": 25, "offset_ma_sell": 1, "offset_cl_sell": 1,
        "ma_compare_short": 25, "ma_compare_long": 25,
        "offset_compare_short": 25, "offset_compare_long": 1,
        "stop_loss_pct": 0.0, "take_profit_pct": 0.0
    },

    "SOXL 익절 포함 최고 전략": {
        "ma_buy": 25, "offset_ma_buy": 5, "offset_cl_buy": 25,
        "ma_sell": 25, "offset_ma_sell": 1, "offset_cl_sell": 1,
        "ma_compare_short": 25, "ma_compare_long": 25,
        "offset_compare_short": 25, "offset_compare_long": 1,
        "stop_loss_pct": 0.0, "take_profit_pct": 50.0
    }
}

# ✅ UI 구성
st.set_page_config(page_title="전략 백테스트", layout="wide")
st.title("📊 전략 백테스트 웹앱")

col1, col2 = st.columns(2)
with col1:
    signal_ticker = st.text_input("시그널 판단용 티커", value="SOXL")
with col2:
    trade_ticker = st.text_input("실제 매매 티커", value="SOXL")

col3, col4 = st.columns(2)
with col3:
    start_date = st.date_input("시작일", value=datetime.date(2021, 1, 1))
with col4:
    end_date = st.date_input("종료일", value=datetime.date.today())

with st.expander("📈 전략 조건 설정"):

# ✅ 전략 행동 방식 선택

# 📌 프리셋 선택 UI
    selected_preset = st.selectbox("🎯 전략 프리셋 선택", ["직접 설정"] + list(PRESETS.keys()))

    if selected_preset != "직접 설정":
        preset_values = PRESETS[selected_preset]
    else:
        preset_values = {}


    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**📥 매수 조건**")
        offset_ma_buy = st.number_input("□일 전", key="offset_ma_buy", value=preset_values.get("offset_ma_buy", 1))
        ma_buy = st.number_input("□일 이동평균선보다", key="ma_buy", value=preset_values.get("ma_buy", 25))
        offset_cl_buy = st.number_input("□일 전 종가가 크면 **매수**", key="offset_cl_buy", value=preset_values.get("offset_cl_buy", 25))
        st.markdown("---") 
        st.markdown("근데, 필요시 조건을 더 해")
        offset_compare_short = st.number_input("□일 전", key="offset_compare_short", value=preset_values.get("offset_compare_short", 1))
        ma_compare_short = st.number_input("□일 이동평균선보다 (0=비활성)", key="ma_compare_short", value=preset_values.get("ma_compare_short", 0))
        offset_compare_long = st.number_input("□일 전", key="offset_compare_long", value=preset_values.get("offset_compare_long", 1))
        ma_compare_long = st.number_input("□일 이동평균선이 커야 **매수**", key="ma_compare_long", value=preset_values.get("ma_compare_long", 0))

    with col2:
        st.markdown("**📤 매도 조건**")
        offset_ma_sell = st.number_input("□일 전", key="offset_ma_sell", value=preset_values.get("offset_ma_sell", 1))
        ma_sell = st.number_input("□일 이동평균선보다", key="ma_sell", value=preset_values.get("ma_sell", 25))
        offset_cl_sell = st.number_input("□일 전 종가가 작으면 매도", key="offset_cl_sell", value=preset_values.get("offset_cl_sell", 1))

        stop_loss_pct = st.number_input("손절 기준 (%)", key="stop_loss_pct", value=preset_values.get("stop_loss_pct", 0.0), step=0.5)
        take_profit_pct = st.number_input("익절 기준 (%)", key="take_profit_pct", value=preset_values.get("take_profit_pct", 0.0), step=0.5)

    strategy_behavior = st.selectbox(
        "⚙️ 매수/매도 조건 동시 발생 시 행동",
        options=[
            "1. 포지션 없으면 매수 / 보유 중이면 매도",
            "2. 포지션 없으면 매수 / 보유 중이면 HOLD",
            "3. 포지션 없으면 HOLD / 보유 중이면 매도"
        ]
    )


# ✅ 시그널 체크
if st.button("📌 오늘 시그널 체크"):
    df_today = get_data(signal_ticker, start_date, end_date)
    if not df_today.empty:
        check_signal_today(df_today,
            ma_buy=ma_buy,
            offset_ma_buy=offset_ma_buy,
            ma_sell=ma_sell,
            offset_ma_sell=offset_ma_sell,
            offset_cl_buy=offset_cl_buy,
            offset_cl_sell=offset_cl_sell,
            ma_compare_short=ma_compare_short if ma_compare_short > 0 else None,
            ma_compare_long=ma_compare_long if ma_compare_long > 0 else None,
            offset_compare_short=offset_compare_short,
            offset_compare_long=offset_compare_long
        )

######### 주요 코드 [백테스트] #########
def backtest_strategy_with_ma_compare(signal_ticker, trade_ticker,
                                      ma_buy, offset_ma_buy, ma_sell, offset_ma_sell,
                                      offset_cl_buy, offset_cl_sell,
                                      ma_compare_short=None, ma_compare_long=None,
                                      offset_compare_short=1, offset_compare_long=1,
                                      initial_cash=5_000_000,
                                      start_date=None, end_date=None,
                                      min_days_between_trades=0,
                                      stop_loss_pct=0.0,
                                      take_profit_pct=0.0):

    signal_df = get_data(signal_ticker, start_date, end_date)
    trade_df = get_data(trade_ticker, start_date, end_date)

    if signal_df.empty or trade_df.empty:
        st.warning("❗데이터가 부족합니다.")
        return {}

    signal_df["MA_BUY"] = signal_df["Close"].rolling(ma_buy).mean()
    signal_df["MA_SELL"] = signal_df["Close"].rolling(ma_sell).mean()

    if ma_compare_short and ma_compare_long:
        signal_df["MA_SHORT"] = signal_df["Close"].rolling(ma_compare_short).mean()
        signal_df["MA_LONG"] = signal_df["Close"].rolling(ma_compare_long).mean()
    else:
        signal_df["MA_SHORT"] = signal_df["MA_LONG"] = None

    cash = initial_cash
    buy_price = None
    position = 0.0
    asset_curve = []
    logs = []

    for i in range(max(ma_buy, ma_sell,
                       offset_ma_buy, offset_ma_sell,
                       offset_cl_buy, offset_cl_sell,
                       offset_compare_short or 0, offset_compare_long or 0), len(signal_df)):

        cl_b = float(signal_df["Close"].iloc[i - offset_cl_buy])
        ma_b = float(signal_df["MA_BUY"].iloc[i - offset_ma_buy])
        cl_s = float(signal_df["Close"].iloc[i - offset_cl_sell])
        ma_s = float(signal_df["MA_SELL"].iloc[i - offset_ma_sell])
        close_today = float(trade_df["Close"].iloc[i])
        current_date = trade_df["Date"].iloc[i]

        trend_ok = True
        if ma_compare_short and ma_compare_long:
            ma_short = signal_df["MA_SHORT"].iloc[i - offset_compare_short]
            ma_long = signal_df["MA_LONG"].iloc[i - offset_compare_long]
            trend_ok = ma_short >= ma_long

        profit_pct = (close_today - buy_price) / buy_price * 100 if buy_price else 0

        signal = "HOLD"

        buy_condition = cl_b > ma_b and trend_ok
        sell_condition = cl_s < ma_s
        stop_hit = stop_loss_pct > 0 and profit_pct <= -stop_loss_pct
        take_hit = take_profit_pct > 0 and profit_pct >= take_profit_pct

        # ✅ 전략 1 / 2 / 3 분기 처리
        if strategy_behavior.startswith("1"):
            if buy_condition and sell_condition:
                if position == 0:
                    position = cash / close_today
                    cash = 0.0
                    signal = "BUY"
                    buy_price = close_today
                else:
                    cash = position * close_today
                    position = 0.0
                    signal = "SELL"
                    buy_price = None
            elif position == 0 and buy_condition:
                position = cash / close_today
                cash = 0.0
                signal = "BUY"
                buy_price = close_today
            elif position > 0 and (sell_condition or stop_hit or take_hit):
                cash = position * close_today
                position = 0.0
                signal = "SELL"
                buy_price = None

        elif strategy_behavior.startswith("2"):
            if buy_condition and sell_condition:
                if position == 0:
                    position = cash / close_today
                    cash = 0.0
                    signal = "BUY"
                    buy_price = close_today
                else:
                    signal = "HOLD"
            elif position == 0 and buy_condition:
                position = cash / close_today
                cash = 0.0
                signal = "BUY"
                buy_price = close_today
            elif position > 0 and (sell_condition or stop_hit or take_hit):
                cash = position * close_today
                position = 0.0
                signal = "SELL"
                buy_price = None

        elif strategy_behavior.startswith("3"):
            if buy_condition and sell_condition:
                if position == 0:
                    signal = "HOLD"  # 매수/매도 모두 만족, 포지션 없으면 HOLD
                else:
                    cash = position * close_today
                    position = 0.0
                    signal = "SELL"
                    buy_price = None

            elif buy_condition and position == 0:
                position = cash / close_today
                cash = 0.0
                signal = "BUY"
                buy_price = close_today

            elif position > 0 and (sell_condition or stop_hit or take_hit):
                cash = position * close_today
                position = 0.0
                signal = "SELL"
                buy_price = None

        total = cash + (position * close_today if position > 0 else 0)
        asset_curve.append(total)

        logs.append({
            "날짜": current_date.strftime("%Y-%m-%d"),
            "종가": round(close_today, 2),
            "신호": signal,
            "자산": round(total),
            "매수시그널": buy_condition,
            "매도시그널": sell_condition,
            "매수이유": (
                f"종가({cl_b:.2f}) > MA_BUY({ma_b:.2f})"
                + (f" + 추세필터 통과" if trend_ok else " + 추세필터 불통과")
                if buy_condition else ""
            ),
            "매도이유": (
                f"종가({cl_s:.2f}) < MA_SELL({ma_s:.2f})"
                if sell_condition else ""
            ),
            "양시그널": buy_condition and sell_condition 
        })


    df = trade_df.iloc[-len(asset_curve):].copy()
    df["Asset"] = asset_curve
    mdd = get_mdd(df["Asset"])
    peak = df["Asset"].cummax()
    drawdown = df["Asset"] / peak - 1
    mdd_pos = drawdown.values.argmin()
    mdd_date = df["Date"].iloc[mdd_pos]

    recovery_date = None
    for i in range(mdd_pos, len(df)):
        if df["Asset"].iloc[i] >= peak.iloc[mdd_pos]:
            recovery_date = df["Date"].iloc[i]
            break

    trade_pairs = []
    current_buy = None
    for log in logs:
        if log["신호"] == "BUY":
            current_buy = log
        elif log["신호"] == "SELL" and current_buy:
            trade_pairs.append((current_buy, log))
            current_buy = None

    win_trades = sum(1 for buy, sell in trade_pairs if sell["종가"] > buy["종가"])
    total_trades = len(trade_pairs)
    win_rate = round((win_trades / total_trades) * 100, 2) if total_trades > 0 else 0

    return {
        "최종 자산": round(asset_curve[-1]),
        "수익률 (%)": round((asset_curve[-1] - initial_cash) / initial_cash * 100, 2),
        "승률 (%)": win_rate,
        "MDD (%)": round(mdd, 2),
        "MDD 발생일": mdd_date.strftime("%Y-%m-%d"),
        "MDD 회복일": recovery_date.strftime("%Y-%m-%d") if recovery_date else "미회복",
        "회복 기간 (일)": (recovery_date - mdd_date).days if recovery_date else None,
        "총 매매 횟수": total_trades,
        "매매 로그": logs
    }

def run_random_simulations(n_simulations=30):
    results = []
    for _ in range(n_simulations):
        # 랜덤 파라미터 생성
        ma_buy = random.choice([5, 10, 15, 25, 50])
        offset_ma_buy = random.choice([1, 5, 15, 25])
        offset_cl_buy = random.choice([1, 5, 15, 25])

        ma_sell = random.choice([5, 10, 15, 25])
        offset_ma_sell = random.choice([1, 5, 15, 25])
        offset_cl_sell = random.choice([1, 5, 15, 25])

        ma_compare_short = random.choice([0, 5, 15, 25, 50])
        ma_compare_long = random.choice([0, 5, 15, 25, 50])
        offset_compare_short = random.choice([1, 5, 15, 25, 50])
        offset_compare_long = 1

        stop_loss_pct = random.choice([0, 10])
        take_profit_pct = random.choice([0, 10, 25, 50])

        result = backtest_strategy_with_ma_compare(
            signal_ticker=signal_ticker,
            trade_ticker=trade_ticker,
            ma_buy=ma_buy,
            offset_ma_buy=offset_ma_buy,
            ma_sell=ma_sell,
            offset_ma_sell=offset_ma_sell,
            offset_cl_buy=offset_cl_buy,
            offset_cl_sell=offset_cl_sell,
            ma_compare_short=ma_compare_short if ma_compare_short > 0 else None,
            ma_compare_long=ma_compare_long if ma_compare_long > 0 else None,
            offset_compare_short=offset_compare_short,
            offset_compare_long=offset_compare_long,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
            start_date=start_date,
            end_date=end_date
        )

        if result:
            result_clean = {k: v for k, v in result.items() if k != "매매 로그"}
            results.append({
                **result_clean,
                "ma_buy": ma_buy,
	    "offset_ma_buy": offset_ma_buy,
                "ma_sell": ma_sell,
	    "offset_ma_sell": offset_ma_sell,
                "offset_cl_buy": offset_cl_buy,
                "offset_cl_sell": offset_cl_sell,
                "ma_compare_short": ma_compare_short if ma_compare_short > 0 else None,
                "ma_compare_long": ma_compare_long if ma_compare_long > 0 else None,
                "offset_compare_short": offset_compare_short,
                "offset_compare_long": offset_compare_long,
                "stop_loss": stop_loss_pct,
                "take_profit": take_profit_pct,
                "승률": result["승률 (%)"],
                "수익률": result["수익률 (%)"]
            })
    return pd.DataFrame(results)


# ✅ UI 버튼 및 시각화
if st.button("✅ 백테스트 실행"):
    result = backtest_strategy_with_ma_compare(
        signal_ticker=signal_ticker,
        trade_ticker=trade_ticker,
        ma_buy=ma_buy,
        offset_ma_buy=offset_ma_buy,
        ma_sell=ma_sell,
        offset_ma_sell=offset_ma_sell,
        offset_cl_buy=offset_cl_buy,
        offset_cl_sell=offset_cl_sell,
        ma_compare_short=ma_compare_short if ma_compare_short > 0 else None,
        ma_compare_long=ma_compare_long if ma_compare_long > 0 else None,
        offset_compare_short=offset_compare_short,
        offset_compare_long=offset_compare_long,
        start_date=start_date,
        end_date=end_date,
        stop_loss_pct=stop_loss_pct,
        take_profit_pct=take_profit_pct
    )

    if result:
        st.subheader("📊 백테스트 결과 요약")
        st.json({k: v for k, v in result.items() if k != "매매 로그"})

        df_log = pd.DataFrame(result["매매 로그"])
        df_log["날짜"] = pd.to_datetime(df_log["날짜"])
        df_log.set_index("날짜", inplace=True)

############# 그래프 그리기 ###########
        fig = go.Figure()

        # 자산 곡선 (왼쪽 y축)
        fig.add_trace(go.Scatter(
            x=df_log.index,
            y=df_log["자산"],
            mode="lines",
            name="Asset",
            yaxis="y1"
        ))

        # 종가 (오른쪽 y축)
        fig.add_trace(go.Scatter(
            x=df_log.index,
            y=df_log["종가"],
            mode="lines",
            name="Price",
            yaxis="y2"
        ))

        # 매수/매도 시점 필터
        buy_points = df_log[df_log["신호"] == "BUY"]
        sell_points = df_log[df_log["신호"] == "SELL"]

        # 동시 만족 필터
        both_buy = buy_points[buy_points["양시그널"] == True]
        both_sell = sell_points[sell_points["양시그널"] == True]

        # 일반 BUY 마커
        fig.add_trace(go.Scatter(
            x=buy_points.index,
            y=buy_points["종가"],
            mode="markers",
            name="BUY",
            yaxis="y2",
            marker=dict(
                color="green",
                size=6,
                symbol="triangle-up"
            )
        ))

        # 일반 SELL 마커
        fig.add_trace(go.Scatter(
            x=sell_points.index,
            y=sell_points["종가"],
            mode="markers",
            name="SELL",
            yaxis="y2",
            marker=dict(
                color="red",
                size=6,
                symbol="triangle-down"
            )
        ))

        # 동시 BUY 마커 (노란 테두리)
        fig.add_trace(go.Scatter(
            x=both_buy.index,
            y=both_buy["종가"],
            mode="markers",
            name="BUY (양시그널)",
            yaxis="y2",
            marker=dict(
                color="green",
                size=9,
                symbol="triangle-up",
                line=dict(color="yellow", width=2)
            )
        ))

        # 동시 SELL 마커 (노란 테두리)
        fig.add_trace(go.Scatter(
            x=both_sell.index,
            y=both_sell["종가"],
            mode="markers",
            name="SELL (양시그널)",
            yaxis="y2",
            marker=dict(
                color="red",
                size=9,
                symbol="triangle-down",
                line=dict(color="yellow", width=2)
            )
        ))

        # 레이아웃 설정
        fig.update_layout(
            title="📈 자산 & 종가 흐름 (BUY/SELL 시점 포함)",
            yaxis=dict(title="Asset"),
            yaxis2=dict(title="Price", overlaying="y", side="right"),
            hovermode="x unified",
            height=800
        )

        st.plotly_chart(fig, use_container_width=True)

#############
        with st.expander("🧾 매매 로그"):
            st.dataframe(df_log)

        # 다운로드 버튼
        csv = df_log.reset_index().to_csv(index=False).encode("utf-8-sig")
        st.download_button("⬇️ 백테스트 결과 다운로드 (CSV)", data=csv, file_name="backtest_result.csv", mime="text/csv")

if st.button("🧪 랜덤 전략 시뮬레이션 (50회 실행)"):
    df_sim = run_random_simulations(50)
    st.subheader("📈 랜덤 전략 시뮬레이션 결과")
    st.dataframe(df_sim.sort_values(by="수익률", ascending=False).reset_index(drop=True))
