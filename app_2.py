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

import random
import re

def _parse_choices(text, cast="int"):
    """
    콤마/공백 구분 입력 문자열을 리스트로 파싱.
    cast: "int" | "float" | "str" | "bool"
    예) "1, 5, 10" -> [1,5,10]
        ">, <" -> [">", "<"]
        "True, False" -> [True, False]
    """
    if text is None:
        return []

    # 쉼표/공백 기준으로 토큰화
    tokens = [t for t in re.split(r"[,\s]+", str(text).strip()) if t != ""]
    if not tokens:
        return []

    def _to_bool(s):
        s = s.strip().lower()
        return s in ("1", "true", "t", "y", "yes")

    out = []
    for t in tokens:
        if cast == "int":
            out.append(int(t))
        elif cast == "float":
            out.append(float(t))
        elif cast == "bool":
            out.append(_to_bool(t))
        else:
            out.append(str(t))
    # 중복 제거 + 원래 순서 유지
    seen = set()
    dedup = []
    for v in out:
        if (v if cast != "str" else (v,)) in seen:
            continue
        seen.add(v if cast != "str" else (v,))
        dedup.append(v)
    return dedup

def _pick_one(choices, fallback):
    """choices가 비었으면 fallback 사용, 아니면 random.choice"""
    return random.choice(choices) if choices else fallback



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
    """KRX(숫자티커)용: OHLC 로딩 (손절/익절 장중체크용)"""
    df = stock.get_etf_ohlcv_by_date(
        start_date.strftime("%Y%m%d"),
        end_date.strftime("%Y%m%d"),
        ticker
    )
    df = df.reset_index().rename(columns={
        "날짜": "Date", "시가": "Open", "고가": "High", "저가": "Low", "종가": "Close"
    })
    df = df[["Date", "Open", "High", "Low", "Close"]].dropna()
    return df

@st.cache_data(show_spinner=False, ttl=3600)
def get_yf_data_cached(ticker: str, start_date, end_date):
    """야후파이낸스용: OHLC 로딩 (손절/익절 장중체크용)"""
    df = yf.download(ticker, start=start_date, end=end_date)
    if isinstance(df.columns, pd.MultiIndex):
        tu = ticker.upper()
        try:
            o = df[("Open",  tu)]
            h = df[("High",  tu)]
            l = df[("Low",   tu)]
            c = df[("Close", tu)]
            df = pd.concat([o, h, l, c], axis=1)
            df.columns = ["Open", "High", "Low", "Close"]
        except Exception:
            # 멀티컬럼 구조가 다를 때의 안전장치
            df = df.droplevel(1, axis=1)[["Open", "High", "Low", "Close"]]
    else:
        df = df[["Open", "High", "Low", "Close"]]
    df = df.reset_index().rename(columns={"Date": "Date"})
    df = df[["Date", "Open", "High", "Low", "Close"]].dropna()
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
    """신호용 Close + 트레이드용 OHLC 병합 및 신속 MA 계산"""
    sig = get_data(signal_ticker, start_date, end_date).sort_values("Date")
    trd = get_data(trade_ticker,  start_date, end_date).sort_values("Date")

    # 신호는 Close만 쓰도록 명시
    sig = sig.rename(columns={"Close": "Close_sig"})[["Date", "Close_sig"]]
    # 트레이드는 OHLC 전부 사용 (장중 체결용)
    trd = trd.rename(columns={
        "Open": "Open_trd", "High": "High_trd",
        "Low": "Low_trd",   "Close": "Close_trd"
    })[["Date", "Open_trd", "High_trd", "Low_trd", "Close_trd"]]

    base = pd.merge(sig, trd, on="Date", how="inner").dropna().reset_index(drop=True)

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

### 오늘의 시그널 ####
def check_signal_today(
    df,
    ma_buy, offset_ma_buy, ma_sell, offset_ma_sell,
    offset_cl_buy, offset_cl_sell,
    ma_compare_short=None, ma_compare_long=None,
    offset_compare_short=1, offset_compare_long=1,
    buy_operator=">", sell_operator="<",
    use_trend_in_buy=True, use_trend_in_sell=False
):

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
    if use_trend_in_buy or use_trend_in_sell:
        try:
            ma_short = float(df["MA_SHORT"].iloc[i - offset_compare_short])
            ma_long = float(df["MA_LONG"].iloc[i - offset_compare_long])
            trend_ok = ma_short >= ma_long
            trend_msg = f"{ma_short:.2f} vs {ma_long:.2f} → {'매수추세' if trend_ok else '매도추세'}"
        except:
            trend_msg = "❗데이터 부족"
            #trend_ok = False

    st.write(f"📈 추세 조건: {trend_msg}")

# ▶ 부호/추세 반영한 판정
    buy_base  = (cl_b > ma_b) if (buy_operator == ">") else (cl_b < ma_b)
    sell_base = (cl_s < ma_s) if (sell_operator == "<") else (cl_s > ma_s)

    buy_ok  = (buy_base  and trend_ok)      if use_trend_in_buy  else buy_base
    sell_ok = (sell_base and (not trend_ok)) if use_trend_in_sell else sell_base
    
    st.write(f"💡 매수판단: 종가({cl_b:.2f}) "
             f"{'>' if buy_operator=='>' else '<'} MA({ma_b:.2f})"
             f"{' + 추세필터' if use_trend_in_buy else ''} → "
             f"{'매수조건 ✅' if buy_ok else '조건부족 ❌'}")

    st.write(f"💡 매도판단: 종가({cl_s:.2f}) "
             f"{'<' if sell_operator=='<' else '>'} MA({ma_s:.2f})"
             f"{' + 역추세필터' if use_trend_in_sell else ''} → "
             f"{'매도조건 ✅' if sell_ok else '조건부족 ❌'}")
    
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
                ms_long  = df["MA_LONG"].iloc[j - offset_compare_long]
                trend_pass = (ms_short >= ms_long)

            _buy_base  = (cb > mb) if (buy_operator == ">") else (cb < mb)
            _sell_base = (cs < ms) if (sell_operator == "<") else (cs > ms)

            _buy_ok  = (_buy_base  and trend_pass)      if use_trend_in_buy  else _buy_base
            _sell_ok = (_sell_base and (not trend_pass)) if use_trend_in_sell else _sell_base

            if last_buy_date is None and _buy_ok:
                last_buy_date = df["Date"].iloc[j]
            if last_sell_date is None and _sell_ok:
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
    "SOXL 전략1": {
        "signal_ticker": "SOXL", "trade_ticker": "SOXL",
        "ma_buy": 15, "offset_ma_buy": 15, "offset_cl_buy": 5,
        "ma_sell": 25, "offset_ma_sell": 1, "offset_cl_sell": 5,
        "ma_compare_short": 5, "ma_compare_long": 5,
        "offset_compare_short": 25, "offset_compare_long": 1,
        "buy_operator": "<", "sell_operator": "<",
        "use_trend_in_buy": True, "use_trend_in_sell": True,
        "stop_loss_pct": 0.0, "take_profit_pct": 30.0
    },

    "SOXL 전략2": {
        "signal_ticker": "SOXL", "trade_ticker": "SOXL",
        "offset_cl_buy": 1, "buy_operator": "<", "offset_ma_buy": 5, "ma_buy": 25,
        "offset_cl_sell": 1, "sell_operator": "<", "offset_ma_sell": 1, "ma_sell": 25, 
        "use_trend_in_buy": True, "use_trend_in_sell": True,
        "offset_compare_short": 15, "ma_compare_short": 10,
        "offset_compare_long": 1, "ma_compare_long": 10,         
        "stop_loss_pct": 0.0, "take_profit_pct": 50.0
    },

    "390390 전략": {
        "signal_ticker": "390390", "trade_ticker": "390390",
        "offset_cl_buy": 15, "buy_operator": ">", "offset_ma_buy": 1, "ma_buy": 20,
        "offset_cl_sell": 5, "sell_operator": "<", "offset_ma_sell": 25, "ma_sell": 10, 
        "use_trend_in_buy": True, "use_trend_in_sell": True,
        "offset_compare_short": 25, "ma_compare_short": 15,
        "offset_compare_long": 1, "ma_compare_long": 15,         
        "stop_loss_pct": 0.0, "take_profit_pct": 0.0
    },

    "SOXS 전략": {
        "signal_ticker": "SOXS", "trade_ticker": "SOXS",
        "offset_cl_buy": 5, "buy_operator": ">", "offset_ma_buy": 15, "ma_buy": 10,
        "offset_cl_sell": 1, "sell_operator": "<", "offset_ma_sell": 5, "ma_sell": 5, 
        "use_trend_in_buy": True, "use_trend_in_sell": False,
        "offset_compare_short": 15, "ma_compare_short": 5,
        "offset_compare_long": 1, "ma_compare_long": 5,         
        "stop_loss_pct": 30.0, "take_profit_pct": 50.0
    },

    "SOXS 전략2": {
        "signal_ticker": "SOXS", "trade_ticker": "SOXS",
        "offset_cl_buy": 25, "buy_operator": "<", "offset_ma_buy": 1, "ma_buy": 5,
        "offset_cl_sell": 15, "sell_operator": ">", "offset_ma_sell": 25, "ma_sell": 5, 
        "use_trend_in_buy": True, "use_trend_in_sell": True,
        "offset_compare_short": 25, "ma_compare_short": 20,
        "offset_compare_long": 1, "ma_compare_long": 20,         
        "stop_loss_pct": 0.0, "take_profit_pct": 50.0
    },
    
    "TSLL 전략": {
        "signal_ticker": "TSLL", "trade_ticker": "TSLL",
        "offset_cl_buy": 5, "buy_operator": ">", "offset_ma_buy": 15, "ma_buy": 20,
        "offset_cl_sell": 1, "sell_operator": ">", "offset_ma_sell": 25, "ma_sell": 20, 
        "use_trend_in_buy": True, "use_trend_in_sell": True,
        "offset_compare_short": 25, "ma_compare_short": 15,
        "offset_compare_long": 1, "ma_compare_long": 15,         
        "stop_loss_pct": 0.0, "take_profit_pct": 0.0
    },
      
    "SOXS를 SOXL 보고 사기 전략": {
        "signal_ticker": "SOXL", "trade_ticker": "SOXS",
        "offset_cl_buy": 1, "buy_operator": "<", "offset_ma_buy": 15, "ma_buy": 10,
        "offset_cl_sell": 1, "sell_operator": "<", "offset_ma_sell": 1, "ma_sell": 5, 
        "use_trend_in_buy": True, "use_trend_in_sell": True,
        "offset_compare_short": 1, "ma_compare_short": 10,
        "offset_compare_long": 15, "ma_compare_long": 10,         
        "stop_loss_pct": 0.0, "take_profit_pct": 25.0
    },

    "465580 전략": {
        "signal_ticker": "465580", "trade_ticker": "465580",
        "offset_cl_buy": 5, "buy_operator": "<", "offset_ma_buy": 1, "ma_buy": 10,
        "offset_cl_sell": 1, "sell_operator": "<", "offset_ma_sell": 25, "ma_sell": 5, 
        "use_trend_in_buy": True, "use_trend_in_sell": True,
        "offset_compare_short": 5, "ma_compare_short": 10,
        "offset_compare_long": 1, "ma_compare_long": 10,         
        "stop_loss_pct": 0.0, "take_profit_pct": 25.0
    }

}

# ✅ UI 구성
st.set_page_config(page_title="전략 백테스트", layout="wide")
st.title("📊 전략 백테스트 웹앱")

st.markdown("모든 매매는 종가 매매이나, 손절,익절은 장중 시가. n일전 데이터 기반으로 금일 종가 매매를 한다.")
st.markdown("KODEX미국반도체 390390, KDOEX인버스 114800, KODEX미국나스닥100 379810, ACEKRX금현물 411060, ACE미국30년국채액티브(H) 453850, ACE미국빅테크TOP7Plus 465580")

    # 📌 프리셋 선택 UI
selected_preset = st.selectbox("🎯 전략 프리셋 선택", ["직접 설정"] + list(PRESETS.keys()))
preset_values = {} if selected_preset == "직접 설정" else PRESETS[selected_preset]


col1, col2 = st.columns(2)
with col1:
    signal_ticker = st.text_input(
        "시그널 판단용 티커",
        value=preset_values.get("signal_ticker", "SOXL"),
        key="signal_ticker_input"      # ✅ 고유 key
    )
with col2:
    trade_ticker = st.text_input(
        "실제 매매 티커",
        value=preset_values.get("trade_ticker", "SOXL"),
        key="trade_ticker_input"       # ✅ 고유 key
    )

col3, col4 = st.columns(2)
with col3:
    start_date = st.date_input("시작일", value=datetime.date(2010, 1, 1),
                               min_value=datetime.date(1990, 1, 1),
                               max_value=datetime.date.today()
                              )
with col4:
    end_date = st.date_input("종료일", value=datetime.date.today(),
                             min_value=start_date,
                             max_value=datetime.date.today()
                            )

with st.expander("📈 전략 조건 설정"):

    ops = [">", "<"]

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("**📥 매수 조건**")
        offset_cl_buy = st.number_input("□일 전 종가", key="offset_cl_buy", value=preset_values.get("offset_cl_buy", 25))
        buy_operator = st.selectbox("매수 조건 부호", ops, index=ops.index(preset_values.get("buy_operator", ">")))
        offset_ma_buy = st.number_input("□일 전", key="offset_ma_buy", value=preset_values.get("offset_ma_buy", 1))
        ma_buy = st.number_input("□일 이동평균선", key="ma_buy", value=preset_values.get("ma_buy", 25))
        st.markdown("---")
        use_trend_in_buy = st.checkbox("매수에 추세필터 적용", value=preset_values.get("use_trend_in_buy", True))
        offset_compare_short = st.number_input("□일 전", key="offset_compare_short", value=preset_values.get("offset_compare_short", 25))
        ma_compare_short = st.number_input("□일 이동평균선이 (short)", key="ma_compare_short", value=preset_values.get("ma_compare_short", 25))
        offset_compare_long = st.number_input("□일 전", key="offset_compare_long", value=preset_values.get("offset_compare_long", 1))
        ma_compare_long = st.number_input("□일 이동평균선 (long)보다 커야 **매수**", key="ma_compare_long", value=preset_values.get("ma_compare_long", 25))

    with col_right:
        st.markdown("**📤 매도 조건**")
        offset_cl_sell = st.number_input("□일 전 종가", key="offset_cl_sell", value=preset_values.get("offset_cl_sell", 1))
        sell_operator = st.selectbox("매도 조건 부호", ops, index=ops.index(preset_values.get("sell_operator", "<")))
        offset_ma_sell = st.number_input("□일 전", key="offset_ma_sell", value=preset_values.get("offset_ma_sell", 1))
        ma_sell = st.number_input("□일 이동평균선", key="ma_sell", value=preset_values.get("ma_sell", 25))
        stop_loss_pct = st.number_input("손절 기준 (%)", key="stop_loss_pct", value=preset_values.get("stop_loss_pct", 0.0), step=0.5)
        take_profit_pct = st.number_input("익절 기준 (%)", key="take_profit_pct", value=preset_values.get("take_profit_pct", 0.0), step=0.5)
        min_hold_days = st.number_input("매수 후 최소 보유일", key="min_hold_days", value=0, min_value=0, step=1)
        st.markdown("---")
        use_trend_in_sell = st.checkbox("매도는 역추세만(추세 불통과일 때만)", value=preset_values.get("use_trend_in_sell", False))
        

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
        check_signal_today(
            df_today,
            ma_buy=ma_buy, offset_ma_buy=offset_ma_buy,
            ma_sell=ma_sell, offset_ma_sell=offset_ma_sell,
            offset_cl_buy=offset_cl_buy, offset_cl_sell=offset_cl_sell,
            ma_compare_short=ma_compare_short if ma_compare_short > 0 else None,
            ma_compare_long=ma_compare_long if ma_compare_long > 0 else None,
            offset_compare_short=offset_compare_short,
            offset_compare_long=offset_compare_long,
            buy_operator=buy_operator,
            sell_operator=sell_operator,
            use_trend_in_buy=use_trend_in_buy,
            use_trend_in_sell=use_trend_in_sell
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
    use_trend_in_buy=True,
    use_trend_in_sell=False,
    buy_operator=">", sell_operator="<",
    execution_lag_days=1,              # ✅ 추가: 신호 발생 후 몇 거래일 뒤에 체결할지 (기본 1일)
    execution_price_mode="next_open"   # ✅ 추가: "next_open" | "next_close"
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

    # === 트레이드 OHLC 배열 (intraday 체크용) ===
    xO = base["Open_trd"].to_numpy(dtype=float)
    xH = base["High_trd"].to_numpy(dtype=float)
    xL = base["Low_trd"].to_numpy(dtype=float)

    # ===== 변수 =====
    cash = float(initial_cash)
    position = 0.0
    buy_price = None
    asset_curve, logs = [], []
    sb = strategy_behavior[:1]
    hold_days = 0
    # === 예약 주문(신호 발생일로부터 N일 뒤 체결) 상태 ===
    pending_action = None      # "BUY" 또는 "SELL" 예약
    pending_due_idx = None     # 언제 체결할지 (인덱스)

    def _fill_buy(px: float) -> float:
        return px * (1 + (slip_bps + fee_bps) / 10000.0)

    def _fill_sell(px: float) -> float:
        return px * (1 - (slip_bps + fee_bps) / 10000.0)

    def _check_intraday_exit(buy_px, o, h, l):
        """장중 손절/익절 체결 체크.
        - 갭: 시가가 기준선을 넘어선 경우 시가 체결
        - 장중: 해당선 터치 시 선 가격 체결
        - 동시 터치: '손절 우선'
        """
        if buy_px is None:
            return False, False, None

        stop_trigger = False
        take_trigger = False
        fill_px = None

        if stop_loss_pct > 0:
            stop_line = buy_px * (1 - stop_loss_pct / 100.0)
            if o <= stop_line:
                stop_trigger = True
                fill_px = o
            elif l <= stop_line:
                stop_trigger = True
                fill_px = stop_line

        if take_profit_pct > 0:
            take_line = buy_px * (1 + take_profit_pct / 100.0)
            if o >= take_line:
                # 갭상승 -> 시가 체결 (익절)
                if not stop_trigger:
                    take_trigger = True
                    fill_px = o
                # stop_trigger도 True이면 아래 동시터치 처리에서 정리
            elif h >= take_line:
                if not stop_trigger:  # 이미 stop이 잡혔으면 우선권은 stop
                    take_trigger = True
                    fill_px = take_line

        # 같은 날 둘 다 터치된 경우 우선순위: 손절 우선
        if stop_trigger and take_trigger:
            stop_line = buy_px * (1 - stop_loss_pct / 100.0)
            take_line = buy_px * (1 + take_profit_pct / 100.0)
            # 시가가 둘 중 하나를 이미 넘은 경우는 위에서 처리됨.
            # 그 외에는 보수적으로 stop 우선
            fill_px = stop_line
            take_trigger = False

        return stop_trigger, take_trigger, fill_px
     
    
    for i in range(idx0, n):

        just_bought = False
        exec_price = None  # 이번 턴 체결가(있으면 기록)

        # -------------------------------------------------
        # (A) 예약 주문 체결 처리: i가 도래하면 먼저 체결
        # -------------------------------------------------
        # 체결가 기준: 다음날 시가/종가 선택
        def _exec_pending(action):
            nonlocal cash, position, buy_price, hold_days
            px_base = xO[i] if execution_price_mode == "next_open" else x_trd[i]
            if action == "BUY" and position == 0.0:
                fill = _fill_buy(px_base)
                position = cash / fill; cash = 0.0
                buy_price = fill         # 반드시 기록
                return "BUY", fill, True  # (signal, exec_price, just_bought)
            elif action == "SELL" and position > 0.0:
                fill = _fill_sell(px_base)
                cash = position * fill; position = 0.0
                return "SELL", fill, False
            return "HOLD", None, False

        just_bought = False
        exec_price = None
        signal = "HOLD"

        if (pending_action is not None) and (pending_due_idx == i):
            signal, exec_price, just_bought = _exec_pending(pending_action)
            if signal == "SELL":
                buy_price = None
            pending_action, pending_due_idx = None, None

        # -------------------------------------------------
        # -------------------------------------------------

        # 값 가져오기
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

        open_today  = xO[i]
        high_today  = xH[i]
        low_today   = xL[i]
        close_today = x_trd[i]
        profit_pct  = ((close_today - buy_price) / buy_price * 100) if buy_price else 0.0

        # ===== 조건 계산 =====
        signal = "HOLD"

        # 매수/매도 기본 시그널
        buy_base  = (cl_b > ma_b) if (buy_operator == ">") else (cl_b < ma_b)
        sell_base = (cl_s < ma_s) if (sell_operator == "<") else (cl_s > ma_s)

        buy_condition  = (buy_base and trend_ok) if use_trend_in_buy  else buy_base
        sell_condition = (sell_base and (not trend_ok)) if use_trend_in_sell else sell_base


        
        # ===== Intraday 손절/익절 체크 (보유 시 즉시 체결; 예약보다 우선) =====
        stop_hit, take_hit, intraday_px = (False, False, None)
        if position > 0.0 and (stop_loss_pct > 0 or take_profit_pct > 0):
            stop_hit, take_hit, intraday_px = _check_intraday_exit(buy_price, open_today, high_today, low_today)

        if position > 0.0 and (stop_hit or take_hit):
            # 최소보유일 무시 + 오늘 바로 체결
            px = intraday_px if intraday_px is not None else close_today
            fill = _fill_sell(px)
            cash = position * fill; position = 0.0
            signal = "SELL"; exec_price = fill; buy_price = None
            # 이 날에는 더 이상 예약/추가 체결 잡지 않음
            pending_action, pending_due_idx = None, None

        base_sell = sell_condition
        can_sell  = (position > 0.0) and base_sell and (hold_days >= min_hold_days)
        def _schedule(action):
            nonlocal pending_action, pending_due_idx
            pending_action = action
            pending_due_idx = i + int(execution_lag_days)
            
   
        # ===== 체결 =====
        # ===== 조건 계산 =====
        # (이전 코드의 buy_condition / sell_condition 계산은 그대로 사용)
        # ...
        base_sell = (sell_condition)  # stop/take는 위에서 이미 처리했으므로 여기선 순수 규칙만
        can_sell  = (position > 0.0) and base_sell and (hold_days >= min_hold_days)

        # ===== 체결 대신 "예약"만 생성 =====
        # sb: "1","2","3" 행동 규칙은 그대로 적용하여 '오늘 예약할 액션'을 결정

        if sb == "1":
            if buy_condition and sell_condition:
                if position == 0.0:
                    _schedule("BUY")
                else:
                    if can_sell:
                        _schedule("SELL")
            elif position == 0.0 and buy_condition:
                _schedule("BUY")
            elif can_sell:
                _schedule("SELL")

        elif sb == "2":
            if buy_condition and sell_condition:
                if position == 0.0:
                    _schedule("BUY")
                # 보유 중이면 HOLD (예약 안 걸음)
            elif position == 0.0 and buy_condition:
                _schedule("BUY")
            elif can_sell:
                _schedule("SELL")

        else:  # '3'
            if buy_condition and sell_condition:
                if position > 0.0 and can_sell:
                    _schedule("SELL")
                # 포지션 없으면 HOLD
            elif (position == 0.0) and buy_condition:
                _schedule("BUY")
            elif can_sell:
                _schedule("SELL")

        

        # 보유일 카운터
        if position > 0.0:
            if not just_bought:
                hold_days += 1
        else:
            hold_days = 0

        total = cash + (position * close_today if position > 0.0 else 0.0)
        asset_curve.append(total)

        # 예약 상태 텍스트
        pending_text = None
        if pending_action is not None:
            # 데이터 범위 넘어가면 체결 못 하므로 표시만
            due_date = base["Date"].iloc[pending_due_idx] if pending_due_idx is not None and pending_due_idx < n else None
            pending_text = f"{pending_action} 예약 (체결일: {due_date.strftime('%Y-%m-%d') if due_date is not None else '범위밖'})"


        logs.append({
            "날짜": pd.to_datetime(base["Date"].iloc[i]).strftime("%Y-%m-%d"),
            "종가": round(close_today, 2),       # 차트 표시는 종가 기준
            "체결가": round(exec_price, 4) if exec_price is not None else None,  # 실제 체결가 기록
            "신호": signal,
            "자산": round(total),
            "매수시그널": buy_condition,
            "매도시그널": sell_condition,
            "손절발동": bool(stop_hit),
            "익절발동": bool(take_hit),
            "추세만족": bool(trend_ok),
            "매수가격비교": round(cl_b - ma_b, 6),
            "매도가격비교": round(cl_s - ma_s, 6),
            "매수이유": (f"종가({cl_b:.2f}) {'>' if buy_operator=='>' else '<'} MA_BUY({ma_b:.2f})" + (" + 추세필터 통과" if trend_ok else " + 추세필터 불통과")) if buy_condition else "",
            "매도이유": (f"종가({cl_s:.2f}) {'<' if sell_operator=='<' else '>'} MA_SELL({ma_s:.2f})") if sell_condition else "",
            "예약상태": pending_text,       # ✅ 추가: 예약 상황 가시화
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

    # 승률/Profit Factor (이제 체결가 기준으로 재계산)
    trade_pairs, cache_buy = [], None
    for log in logs:
        if log["신호"] == "BUY":
            cache_buy = log
        elif log["신호"] == "SELL" and cache_buy:
            trade_pairs.append((cache_buy, log))
            cache_buy = None
    wins = 0
    trade_returns = []
    gross_profit = 0.0
    gross_loss = 0.0

    for b, s in trade_pairs:
        pb = b["체결가"] if b["체결가"] else b["종가"]
        ps = s["체결가"] if s["체결가"] else s["종가"]
        r = (ps - pb) / pb
        trade_returns.append(r)
        if r >= 0:
            wins += 1
            gross_profit += r
        else:
            gross_loss += (-r)

    total_trades = len(trade_pairs)
    win_rate = round((wins / total_trades) * 100, 2) if total_trades else 0.0
    avg_trade_return_pct = round((np.mean(trade_returns) * 100), 2) if trade_returns else 0.0
    median_trade_return_pct = round((np.median(trade_returns) * 100), 2) if trade_returns else 0.0
    profit_factor = round((gross_profit / gross_loss), 2) if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)

    initial_cash_val = float(initial_cash)
    final_asset = float(asset_curve[-1])

    return {
        "평균 거래당 수익률 (%)": avg_trade_return_pct,
        "수익률 (%)": round((final_asset - initial_cash_val) / initial_cash_val * 100, 2),
        "승률 (%)": win_rate,
        "MDD (%)": round(mdd, 2),
        "중앙값 거래당 수익률 (%)": median_trade_return_pct,
        "Profit Factor": profit_factor,
        "총 매매 횟수": total_trades,
        "MDD 발생일": mdd_date.strftime("%Y-%m-%d"),
        "MDD 회복일": recovery_date.strftime("%Y-%m-%d") if recovery_date is not None else "미회복",
        "회복 기간 (일)": (recovery_date - mdd_date).days if recovery_date is not None else None,
        "매매 로그": logs,
        "최종 자산": round(final_asset)
    }

# ===== Fast Random Sims =====
def run_random_simulations_fast(
    n_simulations, base, x_sig, x_trd, ma_dict_sig,
    initial_cash=5_000_000, fee_bps=0, slip_bps=0,
    randomize_sell_operator=False,  # 필요 시 True로
    choices_dict=None               # ✅ 추가: 사용자 입력 후보값 딕셔너리
):
    results = []
    if choices_dict is None:
        choices_dict = {}

    def _pick_one(choices, fallback):
        return random.choice(choices) if choices else fallback

    results = []
    for _ in range(n_simulations):
        ma_buy             = _pick_one(choices_dict.get("ma_buy", []),               random.choice([1, 5, 10, 15, 25]))
        offset_ma_buy      = _pick_one(choices_dict.get("offset_ma_buy", []),        random.choice([1, 5, 15, 25]))
        offset_cl_buy      = _pick_one(choices_dict.get("offset_cl_buy", []),        random.choice([1, 5, 15, 25]))
        buy_operator       = _pick_one(choices_dict.get("buy_operator", []),         random.choice([">", "<"]))

        ma_sell            = _pick_one(choices_dict.get("ma_sell", []),              random.choice([1, 5, 10, 15, 25]))
        offset_ma_sell     = _pick_one(choices_dict.get("offset_ma_sell", []),       random.choice([1, 5, 15, 25]))
        offset_cl_sell     = _pick_one(choices_dict.get("offset_cl_sell", []),       random.choice([1, 5, 15, 25]))
        sell_operator      = _pick_one(choices_dict.get("sell_operator", []),        random.choice(["<", ">"]))

        use_trend_in_buy   = _pick_one(choices_dict.get("use_trend_in_buy", []),     random.choice([True, False]))
        use_trend_in_sell  = _pick_one(choices_dict.get("use_trend_in_sell", []),    random.choice([True, False]))

        ma_compare_short   = _pick_one(choices_dict.get("ma_compare_short", []),     random.choice([1, 5, 15, 25]))
        ma_compare_long_choice = _pick_one(choices_dict.get("ma_compare_long", []), random.choice([5,15,25]))
        if ma_compare_long_choice == "same":
            ma_compare_long = ma_compare_short
        else:
            ma_compare_long = ma_compare_long_choice
        offset_compare_short = _pick_one(choices_dict.get("offset_compare_short", []), random.choice([1, 15, 25]))
        offset_compare_long  = _pick_one(choices_dict.get("offset_compare_long", []),  1)
        stop_loss_pct      = _pick_one(choices_dict.get("stop_loss_pct", []),        0.0)
        take_profit_pct    = _pick_one(choices_dict.get("take_profit_pct", []),      random.choice([0.0, 25.0, 50.0]))

        
        # 필요한 MA 즉석 보충
        for w in [ma_buy, ma_sell, ma_compare_short, ma_compare_long]:
            if w and w not in ma_dict_sig:
                ma_dict_sig[w] = _fast_ma(x_sig, w)

        r = backtest_fast(
            base, x_sig, x_trd, ma_dict_sig,
            ma_buy, offset_ma_buy, ma_sell, offset_ma_sell,
            offset_cl_buy, offset_cl_sell,
            ma_compare_short, ma_compare_long,
            offset_compare_short, offset_compare_long,
            initial_cash=initial_cash,
            stop_loss_pct=stop_loss_pct, take_profit_pct=take_profit_pct,
            strategy_behavior="1. 포지션 없으면 매수 / 보유 중이면 매도",
            min_hold_days=0,
            fee_bps=fee_bps, slip_bps=slip_bps,
            use_trend_in_buy=use_trend_in_buy,
            use_trend_in_sell=use_trend_in_sell,
            buy_operator=buy_operator, sell_operator=sell_operator,
            execution_lag_days=1,
            execution_price_mode="next_open"
        )
        
        if not r:
            continue

        # "매매 로그" 제외한 요약만
        result_clean = {k: v for k, v in r.items() if k != "매매 로그"}

        results.append({
            **result_clean,
            "매수종가일": offset_cl_buy, "매수비교": buy_operator, "매수이평일": offset_ma_buy, "매수이평": ma_buy, 
            "매도종가일": offset_cl_sell, "매도비교": sell_operator, "매도이평일": offset_ma_sell, "매도이평": ma_sell,
            "매수추세": use_trend_in_buy, "매도추세": use_trend_in_sell,
            "과거이평일": offset_compare_short, "과거이평": ma_compare_short, "최근이평일": offset_compare_long, "최근이평": ma_compare_long,
            "손절": stop_loss_pct, "익절": take_profit_pct,
            # ⛔ 중복 제거: 여기서는 별도의 "수익률" / "승률" 컬럼 추가하지 않음
            # (이미 result_clean에 "수익률 (%)", "승률 (%)"가 있음)
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
        fee_bps=fee_bps, slip_bps=slip_bps,
        use_trend_in_buy=use_trend_in_buy,
        use_trend_in_sell=use_trend_in_sell,
        buy_operator=buy_operator,
        sell_operator=sell_operator,
        execution_lag_days=1,                # ✅ 다음 거래일 체결
        execution_price_mode="next_open"     # ✅ 다음날 시가로 체결 (원하면 "next_close")
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
            "평균 거래당 수익률(%)": result.get("평균 거래당 수익률 (%)", 0.0),
            "ProfitFactor": result.get("Profit Factor", 0.0),
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
                pb = buy_cache["체결가"] if pd.notna(buy_cache.get("체결가")) else buy_cache["종가"]
                ps = r["체결가"] if pd.notna(r.get("체결가")) else r["종가"]
                pnl = (ps - pb) / pb * 100
                pairs.append({
                    "진입일": buy_cache["날짜"],
                    "청산일": r["날짜"],
                    "진입가(체결가)": round(pb, 4),
                    "청산가(체결가)": round(ps, 4),
                    "보유일": r["보유일"],
                    "수익률(%)": round(pnl, 2),
                    "청산이유": "손절" if r["손절발동"] else ("익절" if r["익절발동"] else "규칙매도")
                })
                buy_cache = None

        if pairs:
            st.subheader("🧾 트레이드 요약 (체결가 기준)")
            st.dataframe(pd.DataFrame(pairs))

        # 다운로드 버튼 (로그)
        with st.expander("🧾 매매 로그"):
            st.dataframe(df_log)
        csv = df_log.reset_index().to_csv(index=False).encode("utf-8-sig")
        st.download_button("⬇️ 백테스트 결과 다운로드 (CSV)", data=csv, file_name="backtest_result.csv", mime="text/csv")

# --- 랜덤 시뮬 후보 입력 (간단 파서 포함) ---
import re

def _parse_list(text, typ="int"):
    if text is None:
        return []
    toks = [t for t in re.split(r"[,\s]+", str(text).strip()) if t]
    def to_bool(s):
        s = s.strip().lower()
        return s in ("1","true","t","y","yes","on")
    out = []
    for t in toks:
        if typ == "int":
            if str(t).lower() == "same":   # ✅ same 키워드 허용
                out.append("same")
            else:
                out.append(int(t))
        elif typ == "float":
            out.append(float(t))
        elif typ == "bool":
            out.append(to_bool(t))
        else:
            out.append(str(t))
    # 중복 제거(입력 순서 유지)
    seen, dedup = set(), []
    for v in out:
        k = (typ, v)
        if k in seen: 
            continue
        seen.add(k); dedup.append(v)
    return dedup

with st.expander("🎲 랜덤 시뮬 변수 후보 입력", expanded=False):
    colL, colR = st.columns(2)
    with colL:
        txt_offset_cl_buy     = st.text_input("offset_cl_buy 후보",     "1,5,15,25")
        txt_buy_op            = st.text_input("buy_operator 후보",      ">,<")
        txt_offset_ma_buy     = st.text_input("offset_ma_buy 후보",     "1,5,15,25")
        txt_ma_buy            = st.text_input("ma_buy 후보",            "5,10,15,20,25")

        txt_offset_cl_sell    = st.text_input("offset_cl_sell 후보",    "1,5,15,25")
        txt_sell_op           = st.text_input("sell_operator 후보",     "<,>")
        txt_offset_ma_sell    = st.text_input("offset_ma_sell 후보",    "1,5,15,25")
        txt_ma_sell           = st.text_input("ma_sell 후보",           "5,10,15,20,25")

    with colR:
        txt_off_cmp_s         = st.text_input("offset_compare_short 후보", "1,5,15,25")
        txt_ma_cmp_s          = st.text_input("ma_compare_short 후보",  "5,10,15,20,25")
        txt_off_cmp_l         = st.text_input("offset_compare_long 후보",  "1")
        txt_ma_cmp_l          = st.text_input("ma_compare_long 후보",   "same")

        txt_use_trend_buy     = st.text_input("use_trend_in_buy 후보(True/False)",  "True,False")
        txt_use_trend_sell    = st.text_input("use_trend_in_sell 후보(True/False)", "True,False")
        txt_stop_loss         = st.text_input("stop_loss_pct 후보(%)",  "0,30")
        txt_take_profit       = st.text_input("take_profit_pct 후보(%)","0,30,50")

    n_simulations = st.number_input("시뮬레이션 횟수", value=100, min_value=1, step=10)

choices_dict = {
    "ma_buy":               _parse_list(txt_ma_buy, "int"),
    "offset_ma_buy":        _parse_list(txt_offset_ma_buy, "int"),
    "offset_cl_buy":        _parse_list(txt_offset_cl_buy, "int"),
    "buy_operator":         _parse_list(txt_buy_op, "str"),

    "ma_sell":              _parse_list(txt_ma_sell, "int"),
    "offset_ma_sell":       _parse_list(txt_offset_ma_sell, "int"),
    "offset_cl_sell":       _parse_list(txt_offset_cl_sell, "int"),
    "sell_operator":        _parse_list(txt_sell_op, "str"),

    "use_trend_in_buy":     _parse_list(txt_use_trend_buy, "bool"),
    "use_trend_in_sell":    _parse_list(txt_use_trend_sell, "bool"),

    "ma_compare_short":     _parse_list(txt_ma_cmp_s, "int"),
    "ma_compare_long":      _parse_list(txt_ma_cmp_l, "int"),
    "offset_compare_short": _parse_list(txt_off_cmp_s, "int"),
    "offset_compare_long":  _parse_list(txt_off_cmp_l, "int"),

    "stop_loss_pct":        _parse_list(txt_stop_loss, "float"),
    "take_profit_pct":      _parse_list(txt_take_profit, "float"),
}


if st.button("🧪 랜덤 전략 시뮬레이션 실행"):
    ma_pool = [5, 10, 15, 25, 50]
    base, x_sig, x_trd, ma_dict_sig = prepare_base(
        signal_ticker, trade_ticker, start_date, end_date, ma_pool
    )
    if seed:
        random.seed(int(seed))
    df_sim = run_random_simulations_fast(
        int(n_simulations), base, x_sig, x_trd, ma_dict_sig,
        initial_cash=initial_cash_ui, fee_bps=fee_bps, slip_bps=slip_bps,
        choices_dict=choices_dict
    )
    st.subheader(f"📈 랜덤 전략 시뮬레이션 결과 (총 {n_simulations}회)")
    st.dataframe(df_sim.sort_values(by="수익률 (%)", ascending=False).reset_index(drop=True))








