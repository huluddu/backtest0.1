# ✅ 완성형: 안정 기반에 전체 기능 통합 (시그널 체크 + 백테스트 + 그리드서치)

import streamlit as st
import pandas as pd
import yfinance as yf
import datetime
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import itertools
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
    st.markdown("## 🔹 매수 조건 설명")
    st.markdown(
        """
        - `매수 종가 조건일` 전 종가가  
        - `매수 이평선 조건일' 전 `매수 이평선`일 이동평균선보다 크고  
        - 동시에 `추세 옛날 조건일` 전 `추세 옛날 이평선` 이동평균선이  
        - `추세 최근 조건일` 전 `추세 최근 이평선` 이동평균선보다 크거나 같을 때 **매수**
        """
    )
    st.markdown("## 🔹매도 조건 설명")
    st.markdown(
        """
        - `매도 종가 조건일` 전 종가가  
        - `매도 이평선 조건일' 전 `매도 이평선`일 이동평균선보다 클 때 **매도**
        """
    )
    ma_buy = st.number_input("매수 이평선", value=25)
    offset_ma_buy = st.number_input("매수 이평선 조건일", value=1)
    offset_cl_buy = st.number_input("매수 종가 조건일", value=25)

    ma_sell = st.number_input("매도 MA", value=25)
    offset_ma_sell = st.number_input("매도 MA 오프셋", value=1)
    offset_cl_sell = st.number_input("매도 종가 오프셋", value=1)

    ma_compare_short = st.number_input("추세 옛날 이평선 (0=비활성)", value=0)
    ma_compare_long = st.number_input("추세 최근 이평선 (0=비활성)", value=0)
    offset_compare_short = st.number_input("추세 옛날 조건일", value=1)
    offset_compare_long = st.number_input("추세 최근 조건일", value=1)

    stop_loss_pct = st.number_input("손절 기준 (%)", value=0.0, step=0.5, help="예: 10 입력 시 -10% 하락 시 손절")
    take_profit_pct = st.number_input("익절 기준 (%)", value=0.0, step=0.5, help="예: 20 입력 시 +20% 상승 시 익절")

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
        trend_desc = "비활성화"
        if ma_compare_short and ma_compare_long:
            ma_short = signal_df["MA_SHORT"].iloc[i - offset_compare_short]
            ma_long = signal_df["MA_LONG"].iloc[i - offset_compare_long]
            trend_ok = ma_short >= ma_long
            trend_desc = f"{ma_short:.2f} vs {ma_long:.2f}"

        allow_trade = True
        if logs:
            last_trade_date = datetime.datetime.strptime(logs[-1]["날짜"], "%Y-%m-%d")
            if (current_date - last_trade_date).days < min_days_between_trades:
                allow_trade = False

        signal = "HOLD"
        if position == 0 and cl_b > ma_b and trend_ok and allow_trade:
            position = cash / close_today
            cash = 0.0
            signal = "BUY"
            buy_price = close_today
        elif position > 0:
            profit_pct = (close_today - buy_price) / buy_price * 100 if buy_price else 0
            sell_condition = cl_s < ma_s
            stop_hit = stop_loss_pct > 0 and profit_pct <= -stop_loss_pct
            take_hit = take_profit_pct > 0 and profit_pct >= take_profit_pct

            if (sell_condition or stop_hit or take_hit) and allow_trade:
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
            "자산": round(total)
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
        "MDD (%)": round(mdd, 2),
        "MDD 발생일": mdd_date.strftime("%Y-%m-%d"),
        "MDD 회복일": recovery_date.strftime("%Y-%m-%d") if recovery_date else "미회복",
        "회복 기간 (일)": (recovery_date - mdd_date).days if recovery_date else None,
        "총 매매 횟수": total_trades,
        "승률 (%)": win_rate,
        "매매 로그": logs
    }

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

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_log.index, y=df_log["자산"], mode="lines", name="Asset", yaxis="y1"))
        fig.add_trace(go.Scatter(x=df_log.index, y=df_log["종가"], mode="lines", name="Price", yaxis="y2"))

        fig.update_layout(
            title="자산 & 종가 시각화",
            yaxis=dict(title="Asset"),
            yaxis2=dict(title="Price", overlaying="y", side="right"),
            hovermode="x unified",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("🧾 매매 로그"):
            st.dataframe(df_log)

        # 다운로드 버튼
        csv = df_log.reset_index().to_csv(index=False).encode("utf-8-sig")
        st.download_button("⬇️ 백테스트 결과 다운로드 (CSV)", data=csv, file_name="backtest_result.csv", mime="text/csv")
