import streamlit as st
import pandas as pd
import yfinance as yf
import datetime
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import itertools
import random
from pykrx import stock
from functools import lru_cache
import numpy as np


# ===== Fast helpers =====
def _fast_ma(x: np.ndarray, w: int) -> np.ndarray:
    """단순이동평균을 numpy.convolve로 빠르게 계산"""
    if w is None or w <= 1:
        return x.astype(float)
    kernel = np.ones(w, dtype=float) / w
    y = np.full(x.shape, np.nan, dtype=float)
    if len(x) >= w:
        conv = np.convolve(x, kernel, mode="valid")
        y[w-1:] = conv
    return y

@st.cache_data(show_spinner=False, ttl=3600)
def get_krx_data_cached(ticker: str, start_date, end_date):
    """KRX(숫자티커)용: pykrx에서 종가만 가져와 정리"""
    df = stock.get_etf_ohlcv_by_date(start_date.strftime("%Y%m%d"),
                                     end_date.strftime("%Y%m%d"),
                                     ticker)
    df = df[["종가"]].reset_index().rename(columns={"날짜": "Date", "종가": "Close"})
    return df

@st.cache_data(show_spinner=False, ttl=3600)
def get_yf_data_cached(ticker: str, start_date, end_date):
    """야후파이낸스용: Close만 단일 컬럼으로 정리"""
    df = yf.download(ticker, start=start_date, end=end_date)
    if isinstance(df.columns, pd.MultiIndex):
        # 티커 멀티컬럼 보정
        if ("Close", ticker.upper()) in df.columns:
            df = df[("Close", ticker.upper())]
        elif "Close" in df.columns.get_level_values(0):
            df = df["Close"]
        df = df.to_frame(name="Close")
    elif isinstance(df, pd.Series):
        df = df.to_frame(name="Close")
    df = df[["Close"]].dropna().reset_index()
    df.columns = ["Date", "Close"]
    return df

def get_data(ticker: str, start_date, end_date) -> pd.DataFrame:
    """티커 타입에 따라 KRX/yf 로더 분기"""
    try:
        if ticker.lower().endswith(".ks") or ticker.isdigit():
            return get_krx_data_cached(ticker, start_date, end_date)
        return get_yf_data_cached(ticker, start_date, end_date)
    except Exception as e:
        st.error(f"❌ 데이터 로딩 실패: {e}")
        return pd.DataFrame()



# ===== Base prepare =====
@st.cache_data(show_spinner=False, ttl=1800)
def prepare_base(signal_ticker, trade_ticker, start_date, end_date, ma_pool):
    """한 번에 머지 + 필요한 모든 MA(신호용) 미리 계산"""
    sig = get_data(signal_ticker, start_date, end_date).sort_values("Date")
    trd = get_data(trade_ticker, start_date, end_date).sort_values("Date")
    base = pd.merge(sig, trd, on="Date", suffixes=("_sig", "_trd"), how="inner").dropna().reset_index(drop=True)

    x_sig = base["Close_sig"].to_numpy(dtype=float)
    x_trd = base["Close_trd"].to_numpy(dtype=float)

    ma_dict_sig = {}
    for w in sorted(set([w for w in ma_pool if w and w > 0])):
        ma_dict_sig[w] = _fast_ma(x_sig, w)

    return base, x_sig, x_trd, ma_dict_sig


def get_mdd(asset_curve):
    peak = asset_curve.cummax()
    drawdown = (asset_curve - peak) / peak
    return drawdown.min() * 100


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

st.markdown("KODEX미국반도체 390390, KODEX미국나스닥100 379810, ACEKRX금현물 411060, ACE미국30년국채액티브(H) 453850, ACE미국빅테크TOP7Plus 465580")

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
    # 📌 프리셋 선택 UI
    selected_preset = st.selectbox("🎯 전략 프리셋 선택", ["직접 설정"] + list(PRESETS.keys()))

    if selected_preset != "직접 설정":
        preset_values = PRESETS[selected_preset]
    else:
        preset_values = {}

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("**📥 매수 조건**")
        offset_cl_buy = st.number_input("□일 전 종가", key="offset_cl_buy", value=preset_values.get("offset_cl_buy", 25))
        buy_operator = st.selectbox("매수 조건 부호", [">", "<"], index=0)
        offset_ma_buy = st.number_input("□일 전", key="offset_ma_buy", value=preset_values.get("offset_ma_buy", 1))
        ma_buy = st.number_input("□일 이동평균선", key="ma_buy", value=preset_values.get("ma_buy", 25))
   
        st.markdown("---")
        st.markdown("근데, 필요시 조건을 더 해")
        offset_compare_short = st.number_input("□일 전", key="offset_compare_short", value=preset_values.get("offset_compare_short", 1))
        ma_compare_short = st.number_input("□일 이동평균선보다 (0=비활성)", key="ma_compare_short", value=preset_values.get("ma_compare_short", 0))
        offset_compare_long = st.number_input("□일 전", key="offset_compare_long", value=preset_values.get("offset_compare_long", 1))
        ma_compare_long = st.number_input("□일 이동평균선이 커야 **매수**", key="ma_compare_long", value=preset_values.get("ma_compare_long", 0))

    with col_right:
        st.markdown("**📤 매도 조건**")
        offset_cl_sell = st.number_input("□일 전 종가", key="offset_cl_sell", value=preset_values.get("offset_cl_sell", 1))
        sell_operator = st.selectbox("매도 조건 부호", ["<", ">"], index=0)
        offset_ma_sell = st.number_input("□일 전", key="offset_ma_sell", value=preset_values.get("offset_ma_sell", 1))
        ma_sell = st.number_input("□일 이동평균선", key="ma_sell", value=preset_values.get("ma_sell", 25))
        stop_loss_pct = st.number_input("손절 기준 (%)", key="stop_loss_pct", value=preset_values.get("stop_loss_pct", 0.0), step=0.5)
        take_profit_pct = st.number_input("익절 기준 (%)", key="take_profit_pct", value=preset_values.get("take_profit_pct", 0.0), step=0.5)
        min_hold_days = st.number_input("매수 후 최소 보유일", key="min_hold_days", value=0, min_value=0, step=1)

    strategy_behavior = st.selectbox(
        "⚙️ 매수/매도 조건 동시 발생 시 행동",
        options=[
            "1. 포지션 없으면 매수 / 보유 중이면 매도",
            "2. 포지션 없으면 매수 / 보유 중이면 HOLD",
            "3. 포지션 없으면 HOLD / 보유 중이면 매도"
        ]
    )

with st.expander("⚙️ 체결/비용 & 기타 설정"):
    initial_cash_ui = st.number_input("초기 자본", value=5_000_000, step=100_000)
    fee_bps = st.number_input("거래수수료 (bps)", value=25, step=1)
    slip_bps = st.number_input("슬리피지 (bps)", value=0, step=1)
    seed = st.number_input("랜덤 시뮬 Seed (재현성)", value=0, step=1)
    if seed:
        random.seed(int(seed))

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


######### 주요 코드 [백테스트] ###########
# ===== Fast Backtest =====

def backtest_fast(
    base, x_sig, x_trd, ma_dict_sig,
    ma_buy, offset_ma_buy, ma_sell, offset_ma_sell,
    offset_cl_buy, offset_cl_sell,
    ma_compare_short=None, ma_compare_long=None,
    offset_compare_short=1, offset_compare_long=1,
    initial_cash=5_000_000,
    stop_loss_pct=0.0, take_profit_pct=0.0,
    strategy_behavior="1. 포지션 없으면 매수 / 보유 중이면 매도",
    min_hold_days=0,
    fee_bps=0, slip_bps=0,
):
    n = len(base)
    if n == 0:
        return {}

    ma_buy_arr  = ma_dict_sig.get(ma_buy)
    ma_sell_arr = ma_dict_sig.get(ma_sell)
    ma_s_arr = ma_dict_sig.get(ma_compare_short) if ma_compare_short else None
    ma_l_arr = ma_dict_sig.get(ma_compare_long)  if ma_compare_long  else None

    idx0 = max(
        (ma_buy or 1), (ma_sell or 1),
        offset_ma_buy, offset_ma_sell, offset_cl_buy, offset_cl_sell,
        (offset_compare_short or 0), (offset_compare_long or 0)
    )

    # ===== 변수 =====
    cash = float(initial_cash)
    position = 0.0
    buy_price = None
    asset_curve, logs = [], []
    sb = strategy_behavior[:1]
    hold_days = 0

    def _fill_buy(px: float) -> float:
        return px * (1 + (slip_bps + fee_bps) / 10000.0)

    def _fill_sell(px: float) -> float:
        return px * (1 - (slip_bps + fee_bps) / 10000.0)

    for i in range(idx0, n):
        just_bought = False

        # 값 가져오기 (iloc 금지, 배열 인덱싱)
        try:
            cl_b = float(x_sig[i - offset_cl_buy])
            ma_b = float(ma_buy_arr[i - offset_ma_buy])
            cl_s = float(x_sig[i - offset_cl_sell])
            ma_s = float(ma_sell_arr[i - offset_ma_sell])
        except Exception:
            asset_curve.append(cash + position * x_trd[i] if position else cash)
            continue

        trend_ok = True
        if (ma_s_arr is not None) and (ma_l_arr is not None):
            ms = ma_s_arr[i - offset_compare_short] if i - offset_compare_short >= 0 else np.nan
            ml = ma_l_arr[i - offset_compare_long]  if i - offset_compare_long  >= 0 else np.nan
            trend_ok = (np.isfinite(ms) and np.isfinite(ml) and ms >= ml)

        close_today = x_trd[i]
        profit_pct = ((close_today - buy_price) / buy_price * 100) if buy_price else 0.0

        # ===== 조건 계산 =====
        signal = "HOLD"


        if buy_operator == ">":
            buy_condition = (cl_b > ma_b) and trend_ok
        else:
            buy_condition = (cl_b < ma_b) and trend_ok

        if sell_operator == "<":
            sell_condition = (cl_s < ma_s)
        else:
            sell_condition = (cl_s > ma_s)
        
        stop_hit = (stop_loss_pct > 0 and profit_pct <= -stop_loss_pct)
        take_hit = (take_profit_pct > 0 and profit_pct >= take_profit_pct)

        base_sell = (sell_condition or stop_hit or take_hit)
        can_sell = (position > 0.0) and base_sell and (hold_days >= min_hold_days)
        if stop_hit or take_hit:
            can_sell = True

        if sb == "1":
            if buy_condition and sell_condition:
                if position == 0.0:
                    fill = _fill_buy(close_today)
                    position = cash / fill; cash = 0.0
                    signal = "BUY"; buy_price = fill
                    hold_days = 0; just_bought = True
                else:
                    if hold_days >= min_hold_days:
                        fill = _fill_sell(close_today)
                        cash = position * fill; position = 0.0
                        signal = "SELL"; buy_price = None
                    else:
                        signal = "HOLD"

            elif position == 0.0 and buy_condition:
                fill = _fill_buy(close_today)
                position = cash / fill; cash = 0.0
                signal = "BUY"; buy_price = fill
                hold_days = 0; just_bought = True

            elif can_sell:
                fill = _fill_sell(close_today)
                cash = position * fill; position = 0.0
                signal = "SELL"; buy_price = None

        elif sb == "2":
            if buy_condition and sell_condition:
                if position == 0.0:
                    fill = _fill_buy(close_today)
                    position = cash / fill; cash = 0.0
                    signal = "BUY"; buy_price = fill
                    hold_days = 0; just_bought = True
                else:
                    signal = "HOLD"
            elif position == 0.0 and buy_condition:
                fill = _fill_buy(close_today)
                position = cash / fill; cash = 0.0
                signal = "BUY"; buy_price = fill
                hold_days = 0; just_bought = True
            elif can_sell:
                fill = _fill_sell(close_today)
                cash = position * fill; position = 0.0
                signal = "SELL"; buy_price = None

        else:  # '3'
            if buy_condition and sell_condition:
                if position == 0.0:
                    signal = "HOLD"
                else:
                    if hold_days >= min_hold_days:
                        fill = _fill_sell(close_today)
                        cash = position * fill; position = 0.0
                        signal = "SELL"; buy_price = None
                    else:
                        signal = "HOLD"
            elif buy_condition and position == 0.0:
                fill = _fill_buy(close_today)
                position = cash / fill; cash = 0.0
                signal = "BUY"; buy_price = fill
                hold_days = 0; just_bought = True
            elif can_sell:
                fill = _fill_sell(close_today)
                cash = position * fill; position = 0.0
                signal = "SELL"; buy_price = None

        # ✅ 체결 후 카운터 업데이트 (이중 증가 방지)
        if position > 0.0:
            if not just_bought:
                hold_days += 1
        else:
            hold_days = 0

        total = cash + (position * close_today if position > 0.0 else 0.0)
        asset_curve.append(total)

        logs.append({
            "날짜": pd.to_datetime(base["Date"].iloc[i]).strftime("%Y-%m-%d"),
            "종가": round(close_today, 2),
            "신호": signal,
            "자산": round(total),
            "매수시그널": buy_condition,
            "매도시그널": sell_condition,
            "손절발동": bool(stop_hit),
            "익절발동": bool(take_hit),
            "추세만족": bool(trend_ok),
            "매수가격비교": round(cl_b - ma_b, 6),   # (+면 종가>MA)
            "매도가격비교": round(cl_s - ma_s, 6),   # (-면 종가<MA)
            "매수이유": (f"종가({cl_b:.2f}) > MA_BUY({ma_b:.2f})" + (" + 추세필터 통과" if trend_ok else " + 추세필터 불통과")) if buy_condition else "",
            "매도이유": (f"종가({cl_s:.2f}) < MA_SELL({ma_s:.2f})") if sell_condition else "",
            "양시그널": buy_condition and sell_condition,
            "보유일": hold_days
        })

    if not asset_curve:
        return {}

    df = pd.DataFrame({"Date": base["Date"].iloc[-len(asset_curve):].values, "Asset": asset_curve})
    mdd_series = pd.Series(asset_curve)
    peak = mdd_series.cummax()
    drawdown = mdd_series / peak - 1.0
    mdd = float(drawdown.min() * 100)

    mdd_pos = int(np.argmin(drawdown.values))
    mdd_date = pd.to_datetime(df["Date"].iloc[mdd_pos])

    recovery_date = None
    for j in range(mdd_pos, len(df)):
        if df["Asset"].iloc[j] >= peak.iloc[mdd_pos]:
            recovery_date = pd.to_datetime(df["Date"].iloc[j])
            break

    # 승률
    trade_pairs, cache_buy = [], None
    for log in logs:
        if log["신호"] == "BUY":
            cache_buy = log
        elif log["신호"] == "SELL" and cache_buy:
            trade_pairs.append((cache_buy, log))
            cache_buy = None
    wins = sum(1 for b, s in trade_pairs if s["종가"] > b["종가"])
    total_trades = len(trade_pairs)
    win_rate = round((wins / total_trades) * 100, 2) if total_trades else 0.0

    initial_cash_val = float(initial_cash)
    final_asset = float(asset_curve[-1])

    return {
        "최종 자산": round(final_asset),
        "수익률 (%)": round((final_asset - initial_cash_val) / initial_cash_val * 100, 2),
        "승률 (%)": win_rate,
        "MDD (%)": round(mdd, 2),
        "MDD 발생일": mdd_date.strftime("%Y-%m-%d"),
        "MDD 회복일": recovery_date.strftime("%Y-%m-%d") if recovery_date is not None else "미회복",
        "회복 기간 (일)": (recovery_date - mdd_date).days if recovery_date is not None else None,
        "총 매매 횟수": total_trades,
        "매매 로그": logs
    }


# ===== Fast Random Sims =====
def run_random_simulations_fast(n_simulations, base, x_sig, x_trd, ma_dict_sig):
    results = []
    for _ in range(n_simulations):
        ma_buy = random.choice([5, 10, 15, 25, 50])
        offset_ma_buy = random.choice([1, 5, 15, 25])
        offset_cl_buy = random.choice([1, 5, 15, 25])
        buy_operator = random.choice(["<",">"])

        ma_sell = random.choice([5, 10, 15, 25])
        offset_ma_sell = random.choice([1, 5, 15, 25])
        offset_cl_sell = random.choice([1, 5, 15, 25])

        # ✅ 0을 섞어서 None 활성화
        mcs = random.choice([0, 1, 5, 15, 25])
        ma_compare_short = None if mcs == 0 else mcs
        ma_compare_long  = random.choice([1, 5, 15, 25])
        offset_compare_short = random.choice([1, 15, 25])
        offset_compare_long  = random.choice([1, 15, 25])

        stop_loss_pct = random.choice([0])
        take_profit_pct = random.choice([0])

        # 필요한 MA가 dict에 없으면 즉석 계산해서 추가(재사용)
        for w in [ma_buy, ma_sell, ma_compare_short, ma_compare_long]:
            if w and w not in ma_dict_sig:
                ma_dict_sig[w] = _fast_ma(x_sig, w)

        r = backtest_fast(
            base, x_sig, x_trd, ma_dict_sig,
            ma_buy, offset_ma_buy, ma_sell, offset_ma_sell,
            offset_cl_buy, offset_cl_sell,
            ma_compare_short, ma_compare_long,
            offset_compare_short, offset_compare_long,
            stop_loss_pct=stop_loss_pct, take_profit_pct=take_profit_pct
        )
        if not r:
            continue

        result_clean = {k: v for k, v in r.items() if k != "매매 로그"}
        results.append({
            **result_clean,
            "ma_buy": ma_buy, "offset_ma_buy": offset_ma_buy, "buy_operator": buy_operator,
            "ma_sell": ma_sell, "offset_ma_sell": offset_ma_sell,
            "offset_cl_buy": offset_cl_buy, "offset_cl_sell": offset_cl_sell,
            "ma_compare_short": ma_compare_short, "ma_compare_long": ma_compare_long,
            "offset_compare_short": offset_compare_short, "offset_compare_long": offset_compare_long,
            "stop_loss": stop_loss_pct, "take_profit": take_profit_pct,
            "승률": r["승률 (%)"], "수익률": r["수익률 (%)"]
        })
    return pd.DataFrame(results)


# ✅ UI 버튼 및 시각화
if st.button("✅ 백테스트 실행"):
    # 1) 이번 실행에 필요한 MA 윈도우 풀 구성
    ma_pool = [ma_buy, ma_sell]
    if (ma_compare_short or 0) > 0: ma_pool.append(ma_compare_short)
    if (ma_compare_long  or 0) > 0: ma_pool.append(ma_compare_long)

    # 2) 기준 DF + MA 사전계산
    base, x_sig, x_trd, ma_dict_sig = prepare_base(
        signal_ticker, trade_ticker, start_date, end_date, ma_pool
    )

    # 3) 백테스트 실행
    result = backtest_fast(
        base, x_sig, x_trd, ma_dict_sig,
        ma_buy, offset_ma_buy, ma_sell, offset_ma_sell,
        offset_cl_buy, offset_cl_sell,
        ma_compare_short if (ma_compare_short or 0) > 0 else None,
        ma_compare_long  if (ma_compare_long  or 0) > 0 else None,
        offset_compare_short, offset_compare_long,
        initial_cash=initial_cash_ui,
        stop_loss_pct=stop_loss_pct, take_profit_pct=take_profit_pct,
        min_hold_days=min_hold_days,
        strategy_behavior=strategy_behavior,
        fee_bps=fee_bps, slip_bps=slip_bps
    )

    if result:
        st.subheader("📊 백테스트 결과 요약")
        summary = {k: v for k, v in result.items() if k != "매매 로그"}
        st.json(summary)

        df_log = pd.DataFrame(result["매매 로그"])
        df_log["날짜"] = pd.to_datetime(df_log["날짜"])
        df_log.set_index("날짜", inplace=True)

        # ===== 성과지표 보강 (연율화/샤프/벤치마크)
        eq = df_log["자산"].pct_change().dropna()
        if not eq.empty:
            ann_ret = (1 + eq.mean()) ** 252 - 1
            ann_vol = eq.std() * (252 ** 0.5)
            sharpe = (ann_ret / ann_vol) if ann_vol > 0 else 0.0
        else:
            ann_ret = ann_vol = sharpe = 0.0

        st.write({
            "연율화 수익률 CAGR(%)": round(ann_ret * 100, 2),
            "연율화 변동성(%)": round(ann_vol * 100, 2),
            "샤프": round(sharpe, 2),
        })

        # ===== 그래프 그리기 =====
        fig = go.Figure()

        # 벤치마크 (Buy&Hold)
        bench = initial_cash_ui * (df_log["종가"] / df_log["종가"].iloc[0])
        bh_ret = round((bench.iloc[-1] - initial_cash_ui) / initial_cash_ui * 100, 2)

        fig.add_trace(go.Scatter(
            x=df_log.index,
            y=bench,
            mode="lines",
            name="Benchmark",
            yaxis="y1",
            line=dict(dash="dot")
        ))

        # 자산 곡선 (왼쪽 y축)
        fig.add_trace(go.Scatter(
            x=df_log.index,
            y=df_log["자산"],
            mode="lines",
            name="Asset",
            yaxis="y1"
        ))

        # 보유 구간 배경 음영
        pos_step = df_log["신호"].map({"BUY": 1, "SELL": -1}).fillna(0).cumsum()
        in_pos = pos_step > 0
        pos_asset = df_log["자산"].where(in_pos)
        fig.add_trace(go.Scatter(
            x=df_log.index,
            y=pos_asset,
            mode="lines",
            name="In-Position",
            yaxis="y1",
            line=dict(width=0),
            fill="tozeroy",
            fillcolor="rgba(0,150,0,0.08)",
            hoverinfo="skip",
            showlegend=False
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

        # 손절/익절 마커 (자산 축)
        sl = df_log[df_log["손절발동"] == True]
        tp = df_log[df_log["익절발동"] == True]
        if not sl.empty:
            fig.add_trace(go.Scatter(
                x=sl.index, y=sl["자산"], mode="markers", name="손절",
                yaxis="y1", marker=dict(symbol="x", size=9)
            ))
        if not tp.empty:
            fig.add_trace(go.Scatter(
                x=tp.index, y=tp["자산"], mode="markers", name="익절",
                yaxis="y1", marker=dict(symbol="star", size=10)
            ))

        # 레이아웃 설정
        fig.update_layout(
            title=f"📈 자산 & 종가 흐름 (BUY/SELL 시점 포함) — 벤치마크 수익률 {bh_ret}%",
            yaxis=dict(title="Asset"),
            yaxis2=dict(title="Price", overlaying="y", side="right"),
            hovermode="x unified",
            height=800
        )

        st.plotly_chart(fig, use_container_width=True)

        # ===== 트레이드 페어 요약 =====
        pairs, buy_cache = [], None
        for _, r in df_log.reset_index().iterrows():
            if r["신호"] == "BUY":
                buy_cache = r
            elif r["신호"] == "SELL" and buy_cache is not None:
                pnl = (r["종가"] - buy_cache["종가"]) / buy_cache["종가"] * 100
                pairs.append({
                    "진입일": buy_cache["날짜"],
                    "청산일": r["날짜"],
                    "진입가": buy_cache["종가"],
                    "청산가": r["종가"],
                    "보유일": r["보유일"],
                    "수익률(%)": round(pnl, 2),
                    "청산이유": "손절" if r["손절발동"] else ("익절" if r["익절발동"] else "규칙매도")
                })
                buy_cache = None

        if pairs:
            st.subheader("🧾 트레이드 요약")
            st.dataframe(pd.DataFrame(pairs))

        # 다운로드 버튼 (로그)
        with st.expander("🧾 매매 로그"):
            st.dataframe(df_log)
        csv = df_log.reset_index().to_csv(index=False).encode("utf-8-sig")
        st.download_button("⬇️ 백테스트 결과 다운로드 (CSV)", data=csv, file_name="backtest_result.csv", mime="text/csv")


if st.button("🧪 랜덤 전략 시뮬레이션 (30회 실행)"):
    # 랜덤 가능성 있는 MA 윈도우 풀
    ma_pool = [5, 10, 15, 25, 50]
    base, x_sig, x_trd, ma_dict_sig = prepare_base(
        signal_ticker, trade_ticker, start_date, end_date, ma_pool
    )
    if seed:
        random.seed(int(seed))
    df_sim = run_random_simulations_fast(30, base, x_sig, x_trd, ma_dict_sig)
    st.subheader("📈 랜덤 전략 시뮬레이션 결과")
    st.dataframe(df_sim.sort_values(by="수익률", ascending=False).reset_index(drop=True))






