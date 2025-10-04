
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

# ===============================
# 0) 페이지/헤더
# ===============================
st.set_page_config(page_title="전략 백테스트", layout="wide")
st.title("📊 전략 백테스트 웹앱")

st.markdown("모든 매매는 종가 매매이나, 손절·익절은 장중 시가. n일전 데이터 기반으로 금일 종가 매매를 한다.")
st.markdown(
    "예: KODEX미국반도체 390390, KODEX200 069500, KODEX인버스 114800, "
    "KODEX미국나스닥100 379810, ACE KRX 금현물 411060, ACE 미국30년국채액티브(H) 453850, "
    "ACE 미국빅테크TOP7Plus 465580"
)

# =====================================================================================
# 1) PRESETS (네 실제 PRESETS를 아래 딕셔너리에 덮어써서 사용하세요)
# =====================================================================================
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
        "offset_cl_sell": 1, "sell_operator": "<", "offset_ma_sell": 15, "ma_sell": 25,
        "use_trend_in_buy": True, "use_trend_in_sell": True,
        "offset_compare_short": 25, "ma_compare_short": 10,
        "offset_compare_long": 1, "ma_compare_long": 10,
        "stop_loss_pct": 0.0, "take_profit_pct": 0.0
    },

    "SOXL 전략3": {
        "signal_ticker": "SOXL", "trade_ticker": "SOXL",
        "offset_cl_buy": 1, "buy_operator": "<", "offset_ma_buy": 25, "ma_buy": 1,
        "offset_cl_sell": 1, "sell_operator": "<", "offset_ma_sell": 50, "ma_sell": 10,
        "use_trend_in_buy": True, "use_trend_in_sell": True,
        "offset_compare_short": 15, "ma_compare_short": 5,
        "offset_compare_long": 5, "ma_compare_long": 10,
        "stop_loss_pct": 30.0, "take_profit_pct": 10.0
    },
  
    "TSLL 전략": {
        "signal_ticker": "TSLL", "trade_ticker": "TSLL",
        "offset_cl_buy": 5, "buy_operator": ">", "offset_ma_buy": 15, "ma_buy": 20,
        "offset_cl_sell": 1, "sell_operator": "<", "offset_ma_sell": 5, "ma_sell": 10,
        "use_trend_in_buy": True, "use_trend_in_sell": True,
        "offset_compare_short": 25, "ma_compare_short": 15,
        "offset_compare_long": 1, "ma_compare_long": 15,         
        "stop_loss_pct": 0.0, "take_profit_pct": 0.0
    },

    "TSLL2 전략": {
        "signal_ticker": "TSLL", "trade_ticker": "TSLL",
        "offset_cl_buy": 5, "buy_operator": ">", "offset_ma_buy": 15, "ma_buy": 20,
        "offset_cl_sell": 1, "sell_operator": "<", "offset_ma_sell": 5, "ma_sell": 10,
        "use_trend_in_buy": False, "use_trend_in_sell": True,
        "offset_compare_short": 25, "ma_compare_short": 25,
        "offset_compare_long": 15, "ma_compare_long": 25,         
        "stop_loss_pct": 0.0, "take_profit_pct": 0.0
    },

    "GGLL 전략": {
        "signal_ticker": "GGLL", "trade_ticker": "GGLL",
        "offset_cl_buy": 15, "buy_operator": ">", "offset_ma_buy": 15, "ma_buy": 5,
        "offset_cl_sell": 1, "sell_operator": "<", "offset_ma_sell": 5, "ma_sell": 25,
        "use_trend_in_buy": True, "use_trend_in_sell": True,
        "offset_compare_short": 15, "ma_compare_short": 25,
        "offset_compare_long": 25, "ma_compare_long": 25,         
        "stop_loss_pct": 0.0, "take_profit_pct": 15.0
    },

    "BITX 전략": {
        "signal_ticker": "BITX", "trade_ticker": "BITX",
        "offset_cl_buy": 15, "buy_operator": ">", "offset_ma_buy": 25, "ma_buy": 5,
        "offset_cl_sell": 25, "sell_operator": ">", "offset_ma_sell": 1, "ma_sell": 15,
        "use_trend_in_buy": True, "use_trend_in_sell": True,
        "offset_compare_short": 25, "ma_compare_short": 15,
        "offset_compare_long": 5, "ma_compare_long": 15,         
        "stop_loss_pct": 0.0, "take_profit_pct": 0.0
    },

    "ETHU 전략": {
        "signal_ticker": "ETHU", "trade_ticker": "ETHU",
        "offset_cl_buy": 15, "buy_operator": "<", "offset_ma_buy": 5, "ma_buy": 25,
        "offset_cl_sell": 1, "sell_operator": ">", "offset_ma_sell": 1, "ma_sell": 10,
        "use_trend_in_buy": True, "use_trend_in_sell": False,
        "offset_compare_short": 1, "ma_compare_short": 20,
        "offset_compare_long": 15, "ma_compare_long": 15,         
        "stop_loss_pct": 0.0, "take_profit_pct": 10.0
    },

    "SOXS 전략": {
        "signal_ticker": "SOXS", "trade_ticker": "SOXS",
        "offset_cl_buy": 1, "buy_operator": ">", "offset_ma_buy": 20, "ma_buy": 1,
        "offset_cl_sell": 1, "sell_operator": "<", "offset_ma_sell": 1, "ma_sell": 20, 
        "use_trend_in_buy": True, "use_trend_in_sell": True,
        "offset_compare_short": 5, "ma_compare_short": 20,
        "offset_compare_long": 5, "ma_compare_long": 1,
        "stop_loss_pct": 0.0, "take_profit_pct": 10.0
    },

    "SLV 전략": {
        "signal_ticker": "SLV", "trade_ticker": "SLV",
        "offset_cl_buy": 5, "buy_operator": ">", "offset_ma_buy": 5, "ma_buy": 5,
        "offset_cl_sell": 5, "sell_operator": "<", "offset_ma_sell": 1, "ma_sell": 5, 
        "use_trend_in_buy": True, "use_trend_in_sell": True,
        "offset_compare_short": 15, "ma_compare_short": 10,
        "offset_compare_long": 1, "ma_compare_long": 15,         
        "stop_loss_pct": 0.0, "take_profit_pct": 10.0
    }, 

    "453850 ACE 미국30년국채 전략": {
        "signal_ticker": "453850", "trade_ticker": "453850",
        "offset_cl_buy": 15, "buy_operator": "<", "offset_ma_buy": 25, "ma_buy": 15,
        "offset_cl_sell": 25, "sell_operator": ">", "offset_ma_sell": 1, "ma_sell": 20,
        "use_trend_in_buy": True, "use_trend_in_sell": False,
        "offset_compare_short": 1, "ma_compare_short": 15,
        "offset_compare_long": 25, "ma_compare_long": 15,         
        "stop_loss_pct": 0.0, "take_profit_pct": 10.0
    },
          
    "465580 ACE미국빅테크TOP7PLUS": {
        "signal_ticker": "465580", "trade_ticker": "465580",
        "offset_cl_buy": 1, "buy_operator": ">", "offset_ma_buy": 1, "ma_buy": 5,
        "offset_cl_sell": 1, "sell_operator": "<", "offset_ma_sell": 1, "ma_sell": 25, 
        "use_trend_in_buy": False, "use_trend_in_sell": True,
        "offset_compare_short": 5, "ma_compare_short": 10,
        "offset_compare_long": 1, "ma_compare_long": 10,         
        "stop_loss_pct": 0.0, "take_profit_pct": 10.0
    },

    "390390 KODEX미국반도체": {
        "signal_ticker": "390390", "trade_ticker": "390390",
        "offset_cl_buy": 5, "buy_operator": "<", "offset_ma_buy": 1, "ma_buy": 5,
        "offset_cl_sell": 25, "sell_operator": ">", "offset_ma_sell": 1, "ma_sell": 20, 
        "use_trend_in_buy": False, "use_trend_in_sell": True,
        "offset_compare_short": 5, "ma_compare_short": 25,
        "offset_compare_long": 1, "ma_compare_long": 25,
        "stop_loss_pct": 0.0, "take_profit_pct": 10.0
    },

    "371460 TIGER차이나전기차SOLACTIVE": {
        "signal_ticker": "371460", "trade_ticker": "371460",
        "offset_cl_buy": 1, "buy_operator": ">", "offset_ma_buy": 5, "ma_buy": 10,
        "offset_cl_sell": 15, "sell_operator": ">", "offset_ma_sell": 1, "ma_sell": 5, 
        "use_trend_in_buy": True, "use_trend_in_sell": True,
        "offset_compare_short": 5, "ma_compare_short": 15,
        "offset_compare_long": 15, "ma_compare_long": 10,         
        "stop_loss_pct": 0.0, "take_profit_pct": 10.0
    },

    "483280 AITOP10커브드콜": {
        "signal_ticker": "483280", "trade_ticker": "483280",
        "offset_cl_buy": 25, "buy_operator": ">", "offset_ma_buy": 25, "ma_buy": 20,
        "offset_cl_sell": 25, "sell_operator": ">", "offset_ma_sell": 5, "ma_sell": 20, 
        "use_trend_in_buy": True, "use_trend_in_sell": True,
        "offset_compare_short": 1, "ma_compare_short": 20,
        "offset_compare_long": 15, "ma_compare_long": 5,         
        "stop_loss_pct": 0.0, "take_profit_pct": 0.0
    },
}

# =====================================================================================
# 2) 유틸/데이터 로딩 — (기존 로직이 있으면 그대로 써도 OK. 없을 때를 위한 기본 구현)
# =====================================================================================
def _normalize_krx_ticker(t: str) -> str:
    if not isinstance(t, str):
        t = str(t or "")
    t = t.strip().upper()
    t = re.sub(r"\.(KS|KQ)$", "", t)
    m = re.search(r"(\d{6})", t)
    return m.group(1) if m else ""

@st.cache_data(ttl=1800, show_spinner=False)
def get_krx_data(ticker: str, start_date, end_date) -> pd.DataFrame:
    from pykrx import stock
    code = _normalize_krx_ticker(ticker)
    if not code:
        return pd.DataFrame(columns=["Date","Open","High","Low","Close"])
    s = pd.to_datetime(start_date).strftime("%Y%m%d")
    e = pd.to_datetime(end_date).strftime("%Y%m%d")
    try:
        df = stock.get_etf_ohlcv_by_date(s, e, code)
        if df is None or df.empty:
            df = stock.get_market_ohlcv_by_date(s, e, code)
        if df is None or df.empty:
            return pd.DataFrame(columns=["Date","Open","High","Low","Close"])
        df = (df.reset_index()
                .rename(columns={"날짜":"Date","시가":"Open","고가":"High","저가":"Low","종가":"Close"})
                .loc[:, ["Date","Open","High","Low","Close"]]
                .dropna())
        return df
    except Exception:
        return pd.DataFrame(columns=["Date","Open","High","Low","Close"])

@st.cache_data(ttl=1800, show_spinner=False)
def get_yf_data(ticker: str, start_date, end_date) -> pd.DataFrame:
    import yfinance as yf
    try:
        df = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=False)
    except Exception:
        return pd.DataFrame(columns=["Date","Open","High","Low","Close"])
    if df is None or df.empty:
        return pd.DataFrame(columns=["Date","Open","High","Low","Close"])
    if isinstance(df.columns, pd.MultiIndex):
        try:
            tu = str(ticker).upper()
            o = df[("Open", tu)]; h = df[("High", tu)]; l = df[("Low", tu)]; c = df[("Close", tu)]
            df = pd.concat([o,h,l,c], axis=1)
            df.columns = ["Open","High","Low","Close"]
        except Exception:
            df = df.droplevel(1, axis=1)
            if not {"Open","High","Low","Close"}.issubset(df.columns):
                return pd.DataFrame(columns=["Date","Open","High","Low","Close"])
            df = df[["Open","High","Low","Close"]]
    else:
        if not {"Open","High","Low","Close"}.issubset(df.columns):
            return pd.DataFrame(columns=["Date","Open","High","Low","Close"])
        df = df[["Open","High","Low","Close"]]
    df = df.reset_index()
    if "Date" not in df.columns and "Datetime" in df.columns:
        df.rename(columns={"Datetime":"Date"}, inplace=True)
    return df[["Date","Open","High","Low","Close"]].dropna()

def get_data(ticker: str, start_date, end_date) -> pd.DataFrame:
    t = (ticker or "").strip()
    is_krx_like = t.isdigit() or t.lower().endswith((".ks", ".kq"))
    return (get_krx_data if is_krx_like else get_yf_data)(t, start_date, end_date)

def _rolling_ma(close: pd.Series, w: int) -> pd.Series:
    if not w or w <= 1:
        return close.astype(float)
    return close.rolling(int(w)).mean()

def check_signal_today_result(
    df: pd.DataFrame,
    *,
    ma_buy, offset_ma_buy, ma_sell, offset_ma_sell,
    offset_cl_buy, offset_cl_sell,
    ma_compare_short=None, ma_compare_long=None,
    offset_compare_short=1, offset_compare_long=1,
    buy_operator=">", sell_operator="<",
    use_trend_in_buy=True, use_trend_in_sell=False,
):
    """
    간단한 BUY/SELL/HOLD 판정. (너의 기존 로직이 있으면 그걸로 대체해도 OK)
    """
    out = {"label":"데이터없음","buy_ok":False,"sell_ok":False,
           "last_buy":None,"last_sell":None,"last_hold":None}
    if df is None or df.empty:
        return out
    d = df.copy().sort_values("Date").drop_duplicates("Date").reset_index(drop=True)
    d["Close"] = pd.to_numeric(d["Close"], errors="coerce")
    d = d.dropna(subset=["Close"])
    if d.empty:
        return out

    d["MA_BUY"]  = _rolling_ma(d["Close"], int(ma_buy))
    d["MA_SELL"] = _rolling_ma(d["Close"], int(ma_sell))
    if ma_compare_short and ma_compare_long:
        d["MA_SHORT"] = _rolling_ma(d["Close"], int(ma_compare_short))
        d["MA_LONG"]  = _rolling_ma(d["Close"], int(ma_compare_long))

    try:
        cl_b = float(d["Close"].iloc[-int(offset_cl_buy)])
        ma_b = float(d["MA_BUY"].iloc[-int(offset_ma_buy)])
        cl_s = float(d["Close"].iloc[-int(offset_cl_sell)])
        ma_s = float(d["MA_SELL"].iloc[-int(offset_ma_sell)])
    except Exception:
        return out

    trend_ok = True
    if ma_compare_short and ma_compare_long and "MA_SHORT" in d.columns and "MA_LONG" in d.columns:
        try:
            ms = float(d["MA_SHORT"].iloc[-int(offset_compare_short)])
            ml = float(d["MA_LONG"].iloc[-int(offset_compare_long)])
            trend_ok = (ms >= ml)
        except Exception:
            trend_ok = True

    buy_base  = (cl_b > ma_b) if (buy_operator == ">") else (cl_b < ma_b)
    sell_base = (cl_s < ma_s) if (sell_operator == "<") else (cl_s > ma_s)
    buy_ok  = (buy_base and trend_ok) if use_trend_in_buy  else buy_base
    sell_ok = (sell_base and (not trend_ok)) if use_trend_in_sell else sell_base

    label = "HOLD"
    if buy_ok and sell_ok:   label = "BUY & SELL"
    elif buy_ok:             label = "BUY"
    elif sell_ok:            label = "SELL"

    # 최근 BUY/SELL/HOLD 날짜 탐색
    last_buy = last_sell = last_hold = None
    safe_start = max(int(offset_cl_buy), int(offset_ma_buy), int(offset_cl_sell), int(offset_ma_sell),
                     int(offset_compare_short or 0), int(offset_compare_long or 0))
    for j in range(len(d)-1, safe_start-1, -1):
        try:
            cb = d["Close"].iloc[j - int(offset_cl_buy)]
            mb = d["MA_BUY"].iloc[j - int(offset_ma_buy)]
            cs = d["Close"].iloc[j - int(offset_cl_sell)]
            ms = d["MA_SELL"].iloc[j - int(offset_ma_sell)]

            trend_pass = True
            if (ma_compare_short and ma_compare_long and "MA_SHORT" in d.columns and "MA_LONG" in d.columns):
                ms_s = d["MA_SHORT"].iloc[j - int(offset_compare_short)]
                ms_l = d["MA_LONG"].iloc[j - int(offset_compare_long)]
                trend_pass = (ms_s >= ms_l)

            _buy_base  = (cb > mb) if (buy_operator == ">") else (cb < mb)
            _sell_base = (cs < ms) if (sell_operator == "<") else (cs > ms)
            _buy_ok  = (_buy_base and trend_pass)       if use_trend_in_buy  else _buy_base
            _sell_ok = (_sell_base and (not trend_pass)) if use_trend_in_sell else _sell_base

            if last_sell is None and _sell_ok: last_sell = d["Date"].iloc[j]
            if last_buy  is None and _buy_ok:  last_buy  = d["Date"].iloc[j]
            if last_hold is None and (not _buy_ok and not _sell_ok): last_hold = d["Date"].iloc[j]
            if last_buy and last_sell and last_hold: break
        except Exception:
            continue

    fmt = lambda x: (pd.to_datetime(x).strftime("%Y-%m-%d") if x is not None else None)
    return {
        "label": label,
        "buy_ok": bool(buy_ok),
        "sell_ok": bool(sell_ok),
        "last_buy": fmt(last_buy),
        "last_sell": fmt(last_sell),
        "last_hold": fmt(last_hold),
    }

# =====================================================================================
# 3) 프리셋 ↔ 위젯 동기화 (핵심)
# =====================================================================================
DEFAULTS = {
    "signal_ticker":"SOXL", "trade_ticker":"SOXL",
    "offset_cl_buy":25, "buy_operator":">", "offset_ma_buy":1, "ma_buy":25,
    "use_trend_in_buy":True, "offset_compare_short":25, "ma_compare_short":25,
    "offset_compare_long":1, "ma_compare_long":25,
    "offset_cl_sell":1, "sell_operator":"<", "offset_ma_sell":1, "ma_sell":25,
    "stop_loss_pct":0.0, "take_profit_pct":0.0, "min_hold_days":0,
    "use_trend_in_sell":False,
}

MAP_TO_WIDGET = {
    "signal_ticker": "signal_ticker_input",
    "trade_ticker": "trade_ticker_input",
    "offset_cl_buy": "offset_cl_buy",
    "buy_operator": "buy_operator",
    "offset_ma_buy": "offset_ma_buy",
    "ma_buy": "ma_buy",
    "use_trend_in_buy": "use_trend_in_buy",
    "offset_compare_short": "offset_compare_short",
    "ma_compare_short": "ma_compare_short",
    "offset_compare_long": "offset_compare_long",
    "ma_compare_long": "ma_compare_long",
    "offset_cl_sell": "offset_cl_sell",
    "sell_operator": "sell_operator",
    "offset_ma_sell": "offset_ma_sell",
    "ma_sell": "ma_sell",
    "stop_loss_pct": "stop_loss_pct",
    "take_profit_pct": "take_profit_pct",
    "min_hold_days": "min_hold_days",
    "use_trend_in_sell": "use_trend_in_sell",
}

def _coerce_number(v, as_int=False):
    if v is None: return 0 if as_int else 0.0
    try:
        return int(v) if as_int else float(v)
    except Exception:
        return int(float(v)) if as_int else float(v)

def _apply_preset_to_state(preset_name: str):
    p = {} if preset_name == "직접 설정" else {**DEFAULTS, **PRESETS.get(preset_name, {})}
    for k_preset, k_widget in MAP_TO_WIDGET.items():
        v = p.get(k_preset, DEFAULTS.get(k_preset))
        if k_widget in {"offset_cl_buy","offset_ma_buy","ma_buy",
                        "offset_compare_short","ma_compare_short","offset_compare_long","ma_compare_long",
                        "offset_cl_sell","offset_ma_sell","ma_sell","min_hold_days"}:
            v = _coerce_number(v, as_int=True)
        elif k_widget in {"stop_loss_pct","take_profit_pct"}:
            v = _coerce_number(v, as_int=False)
        elif k_widget in {"use_trend_in_buy","use_trend_in_sell"}:
            v = bool(v)
        st.session_state[k_widget] = v

# 최초 1회 기본값 세팅
if "init_done" not in st.session_state:
    _apply_preset_to_state("직접 설정")
    st.session_state["init_done"] = True

def _on_change_preset():
    _apply_preset_to_state(st.session_state["selected_preset"])
    st.rerun()

# =====================================================================================
# 4) UI — 프리셋 선택 & 기본 입력
# =====================================================================================
selected_preset = st.selectbox("🎯 전략 프리셋 선택", ["직접 설정"] + list(PRESETS.keys())), key="selected_preset",
    on_change=_on_change_preset
)

col1, col2 = st.columns(2)
with col1:
    st.text_input("시그널 판단용 티커", key="signal_ticker_input")
with col2:
    st.text_input("실제 매매 티커", key="trade_ticker_input")

col3, col4 = st.columns(2)
with col3:
    start_date = st.date_input(
        "시작일",
        value=datetime.date(2010, 1, 1),
        min_value=datetime.date(1990, 1, 1),
        max_value=datetime.date.today(),
        key="start_date"
    )
with col4:
    end_date = st.date_input(
        "종료일",
        value=datetime.date.today(),
        min_value=st.session_state["start_date"],
        max_value=datetime.date.today(),
        key="end_date"
    )

# =====================================================================================
# 5) 전략 조건 설정
# =====================================================================================
with st.expander("📈 전략 조건 설정", expanded=False):
    ops = [">", "<"]

    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown("**📥 매수 조건**")
        st.number_input("□일 전 종가", key="offset_cl_buy", step=1)
        st.selectbox("매수 조건 부호", ops, key="buy_operator",
                     index=ops.index(st.session_state.get("buy_operator", ">")))
        st.number_input("□일 전", key="offset_ma_buy", step=1)
        st.number_input("□일 이동평균선", key="ma_buy", step=1)

        st.markdown("---")
        st.checkbox("매수에 추세필터 적용", key="use_trend_in_buy")
        st.number_input("□일 전", key="offset_compare_short", step=1)
        st.number_input("□일 이동평균선이 (short)", key="ma_compare_short", step=1)
        st.number_input("□일 전", key="offset_compare_long", step=1)
        st.number_input("□일 이동평균선 (long)보다 커야 **매수**", key="ma_compare_long", step=1)

    with col_right:
        st.markdown("**📤 매도 조건**")
        st.number_input("□일 전 종가", key="offset_cl_sell", step=1)
        st.selectbox("매도 조건 부호", ops, key="sell_operator",
                     index=ops.index(st.session_state.get("sell_operator", "<")))
        st.number_input("□일 전", key="offset_ma_sell", step=1)
        st.number_input("□일 이동평균선", key="ma_sell", step=1)

        st.number_input("손절 기준 (%)", key="stop_loss_pct", step=0.5)
        st.number_input("익절 기준 (%)", key="take_profit_pct", step=0.5)
        st.number_input("매수 후 최소 보유일", key="min_hold_days", min_value=0, step=1)

        st.markdown("---")
        st.checkbox("매도는 역추세만(추세 불통과일 때만)", key="use_trend_in_sell")

strategy_behavior = st.selectbox(
    "⚙️ 매수/매도 조건 동시 발생 시 행동",
    options=[
        "1. 포지션 없으면 매수 / 보유 중이면 매도",
        "2. 포지션 없으면 매수 / 보유 중이면 HOLD",
        "3. 포지션 없으면 HOLD / 보유 중이면 매도",
    ],
    index=0,
    key="strategy_behavior"
)

# =====================================================================================
# 6) 실행 섹션 — 오늘 시그널 + 차트 (20MA 포함)
# =====================================================================================
def _to_date_str(x):
    try:
        return pd.to_datetime(x).strftime("%Y-%m-%d")
    except Exception:
        return None

if st.button("🔎 오늘 시그널 체크", key="btn_check_today"):
    sig = st.session_state["signal_ticker_input"]
    s = _to_date_str(st.session_state["start_date"])
    e = _to_date_str(st.session_state["end_date"])
    df = get_data(sig, s, e)

    if df is None or df.empty or "Close" not in df.columns:
        st.warning("❗ 데이터가 비어있어 시그널을 계산할 수 없습니다.")
        st.stop()

    df = df.sort_values("Date").drop_duplicates("Date").reset_index(drop=True)
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df.dropna(subset=["Close"])
    if df.empty:
        st.warning("❗ 유효한 종가가 없습니다.")
        st.stop()

    # 현재 파라미터 수집
    p = {k: st.session_state[k] for k in [
        "offset_cl_buy","buy_operator","offset_ma_buy","ma_buy",
        "use_trend_in_buy","offset_compare_short","ma_compare_short",
        "offset_compare_long","ma_compare_long",
        "offset_cl_sell","sell_operator","offset_ma_sell","ma_sell",
        "stop_loss_pct","take_profit_pct","min_hold_days","use_trend_in_sell",
    ]}
    res = check_signal_today_result(
        df,
        ma_buy=int(p["ma_buy"]), offset_ma_buy=int(p["offset_ma_buy"]),
        ma_sell=int(p["ma_sell"]), offset_ma_sell=int(p["offset_ma_sell"]),
        offset_cl_buy=int(p["offset_cl_buy"]), offset_cl_sell=int(p["offset_cl_sell"]),
        ma_compare_short=int(p["ma_compare_short"]) if p["ma_compare_short"] else None,
        ma_compare_long=int(p["ma_compare_long"]) if p["ma_compare_long"] else None,
        offset_compare_short=int(p["offset_compare_short"] or 1),
        offset_compare_long=int(p["offset_compare_long"] or 1),
        buy_operator=st.session_state["buy_operator"],
        sell_operator=st.session_state["sell_operator"],
        use_trend_in_buy=bool(p["use_trend_in_buy"]),
        use_trend_in_sell=bool(p["use_trend_in_sell"]),
    )

    st.success(f"📌 오늘 시그널: **{res['label']}**")
    st.write(
        f"- 최근 BUY:  {res.get('last_buy') or '-'}  "
        f"- 최근 SELL: {res.get('last_sell') or '-'}  "
        f"- 최근 HOLD: {res.get('last_hold') or '-'}"
    )

    # 20MA 라인 추가 차트
    df["MA20"] = df["Close"].rolling(20).mean()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Date"], y=df["Close"], name="Close"))
    fig.add_trace(go.Scatter(x=df["Date"], y=df["MA20"],  name="MA20", line=dict(width=2, dash="dot")))
    fig.update_layout(height=480, margin=dict(l=10,r=10,t=30,b=10))
    st.plotly_chart(fig, use_container_width=True)

# (필요 시) PRESETS 일괄 체크/백테스트 섹션은 이후에 추가 확장하면 됩니다.

#######################
# 체결/비용 & 기타
with st.expander("⚙️ 체결/비용 & 기타 설정", expanded=False):
    initial_cash_ui = st.number_input("초기 자본", value=5_000_000, step=100_000)
    fee_bps = st.number_input("거래수수료 (bps)", value=25, step=1)
    slip_bps = st.number_input("슬리피지 (bps)", value=0, step=1)
    seed = st.number_input("랜덤 시뮬 Seed (재현성)", value=0, step=1)
    if seed:
        random.seed(int(seed))

# ================== 탭 ==================
tab1, tab2, tab3, tab4 = st.tabs(["🎯 오늘 시그널", "📚 PRESETS 일괄", "🧪 백테스트", "🎲 랜덤"])
# ---------------- 공통 헬퍼 (탭들 위쪽 어딘가 1회만 정의해두면 좋음) ----------------
def _safe_int(x, default=0):
    try:
        return int(x)
    except Exception:
        try:
            return int(float(x))
        except Exception:
            return default

def _safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default

def _is_krx_ticker(tic: str) -> bool:
    if not isinstance(tic, str):
        tic = str(tic or "")
    t = tic.strip().lower()
    return t.isdigit() or t.endswith(".ks") or t.endswith(".kq")

def _today_offsets(p: dict) -> dict:
    """실시간 요약에서 오늘만 보도록 모든 오프셋 0으로 강제"""
    q = dict(p)
    q.update({
        "offset_cl_buy": 0, "offset_ma_buy": 0,
        "offset_cl_sell": 0, "offset_ma_sell": 0,
        "offset_compare_short": 0, "offset_compare_long": 0,
    })
    return q

def _current_params_from_state():
    """세션 상태에서 현재 전략 파라미터를 안전하게 꺼내 dict로 반환"""
    return {
        "ma_buy":               _safe_int(st.session_state.get("ma_buy", 25)),
        "offset_ma_buy":        _safe_int(st.session_state.get("offset_ma_buy", 1)),
        "ma_sell":              _safe_int(st.session_state.get("ma_sell", 25)),
        "offset_ma_sell":       _safe_int(st.session_state.get("offset_ma_sell", 1)),
        "offset_cl_buy":        _safe_int(st.session_state.get("offset_cl_buy", 1)),
        "offset_cl_sell":       _safe_int(st.session_state.get("offset_cl_sell", 1)),
        "ma_compare_short":     (_safe_int(st.session_state.get("ma_compare_short", 0)) or None),
        "ma_compare_long":      (_safe_int(st.session_state.get("ma_compare_long", 0)) or None),
        "offset_compare_short": _safe_int(st.session_state.get("offset_compare_short", 1)),
        "offset_compare_long":  _safe_int(st.session_state.get("offset_compare_long", 1)),
        "buy_operator":         st.session_state.get("buy_operator", ">"),
        "sell_operator":        st.session_state.get("sell_operator", "<"),
        "use_trend_in_buy":     bool(st.session_state.get("use_trend_in_buy", True)),
        "use_trend_in_sell":    bool(st.session_state.get("use_trend_in_sell", False)),
        "stop_loss_pct":        _safe_float(st.session_state.get("stop_loss_pct", 0.0)),
        "take_profit_pct":      _safe_float(st.session_state.get("take_profit_pct", 0.0)),
        "min_hold_days":        _safe_int(st.session_state.get("min_hold_days", 0)),
        "strategy_behavior":    st.session_state.get("strategy_behavior", "1. 포지션 없으면 매수 / 보유 중이면 매도"),
    }

# ───────────────────────────────────────
# TAB1: 오늘 시그널
# ───────────────────────────────────────
with tab1:
    # 최신 위젯/프리셋 값 읽기 (세션 기준)
    signal_ticker = st.session_state.get("signal_ticker_input", "SOXL")
    trade_ticker  = st.session_state.get("trade_ticker_input", "SOXL")
    start_date    = st.session_state.get("start_date")
    end_date      = st.session_state.get("end_date")
    params        = _current_params_from_state()

    c1, c2 = st.columns(2)
    with c1:
        if st.button("📌 오늘 시그널 체크", use_container_width=True, key="btn_check_eod"):
            df_today = get_data(signal_ticker, start_date, end_date)
            if df_today is None or df_today.empty:
                st.warning("데이터가 없습니다.")
            else:
                check_signal_today(
                    df_today,
                    ma_buy=params["ma_buy"], offset_ma_buy=params["offset_ma_buy"],
                    ma_sell=params["ma_sell"], offset_ma_sell=params["offset_ma_sell"],
                    offset_cl_buy=params["offset_cl_buy"], offset_cl_sell=params["offset_cl_sell"],
                    ma_compare_short=params["ma_compare_short"],
                    ma_compare_long=params["ma_compare_long"],
                    offset_compare_short=params["offset_compare_short"],
                    offset_compare_long=params["offset_compare_long"],
                    buy_operator=params["buy_operator"],
                    sell_operator=params["sell_operator"],
                    use_trend_in_buy=params["use_trend_in_buy"],
                    use_trend_in_sell=params["use_trend_in_sell"],
                    # strategy_behavior=params["strategy_behavior"],
                    # min_hold_days=int(params["min_hold_days"]),
                )
    with c2:
        if st.button("⚡ 오늘 시그널 체크 (실시간)", use_container_width=True, key="btn_check_realtime"):
            df_today = get_data(signal_ticker, start_date, end_date)
            if df_today is None or df_today.empty:
                st.error("기본 데이터 로딩 실패")
            else:
                if _is_krx_ticker(signal_ticker):
                    st.warning("국내 티커는 일봉 데이터로 판정합니다.")
                    check_signal_today(
                        df_today,
                        ma_buy=params["ma_buy"], offset_ma_buy=params["offset_ma_buy"],
                        ma_sell=params["ma_sell"], offset_ma_sell=params["offset_ma_sell"],
                        offset_cl_buy=params["offset_cl_buy"], offset_cl_sell=params["offset_cl_sell"],
                        ma_compare_short=params["ma_compare_short"],
                        ma_compare_long=params["ma_compare_long"],
                        offset_compare_short=params["offset_compare_short"],
                        offset_compare_long=params["offset_compare_long"],
                        buy_operator=params["buy_operator"],
                        sell_operator=params["sell_operator"],
                        use_trend_in_buy=params["use_trend_in_buy"],
                        use_trend_in_sell=params["use_trend_in_sell"],
                        # strategy_behavior=params["strategy_behavior"],
                        # min_hold_days=int(params["min_hold_days"]),
                    )
                else:
                    # 💡 check_signal_today_realtime에 **_extra가 받아지도록 함수 정의에 추가해두세요.
                    check_signal_today_realtime(
                        df_today, signal_ticker,
                        tz="America/New_York", session_start="09:30", session_end="16:00",
                        ma_buy=params["ma_buy"], offset_ma_buy=params["offset_ma_buy"],
                        ma_sell=params["ma_sell"], offset_ma_sell=params["offset_ma_sell"],
                        offset_cl_buy=params["offset_cl_buy"], offset_cl_sell=params["offset_cl_sell"],
                        ma_compare_short=params["ma_compare_short"], ma_compare_long=params["ma_compare_long"],
                        offset_compare_short=params["offset_compare_short"], offset_compare_long=params["offset_compare_long"],
                        buy_operator=params["buy_operator"], sell_operator=params["sell_operator"],
                        use_trend_in_buy=params["use_trend_in_buy"], use_trend_in_sell=params["use_trend_in_sell"],
                        # strategy_behavior=params["strategy_behavior"],
                        # min_hold_days=int(params["min_hold_days"]),
                    )

# ───────────────────────────────────────
# TAB2: PRESETS 일괄
# ───────────────────────────────────────
with tab2:
    st.markdown("#### 🧭 PRESETS 오늘 시그널 요약")

    start_date = st.session_state.get("start_date")
    end_date   = st.session_state.get("end_date")

    # EOD 기준
    if st.button("📚 PRESETS 전체 오늘 시그널 보기", use_container_width=True, key="btn_presets_eod"):
        rows = []
        for name, p in PRESETS.items():
            sig_tic = p.get("signal_ticker", p.get("trade_ticker"))
            df = get_data(sig_tic, start_date, end_date)
            if df is None or df.empty:
                res = {"label": "데이터없음", "last_buy": None, "last_sell": None, "last_hold": None}
            else:
                res = summarize_signal_today(df, p)
            rows.append({
                "전략명": name,
                "티커": sig_tic,
                "시그널": res.get("label", "-"),
                "최근 BUY":  res.get("last_buy")  or "-",
                "최근 SELL": res.get("last_sell") or "-",
                "최근 HOLD": res.get("last_hold") or "-",
                "예약(무포지션)": res.get("reserved_flat") or "-",
                "예약(보유중)":   res.get("reserved_hold") or "-",
            })
        df_view = pd.DataFrame(rows)
        st.dataframe(df_view, use_container_width=True, key="tbl_presets_eod")
        st.download_button(
            "⬇️ CSV 다운로드 (EOD)",
            data=df_view.to_csv(index=False).encode("utf-8-sig"),
            file_name="presets_signal_eod.csv",
            mime="text/csv",
            key="dl_presets_eod"
        )

    # 실시간(US 1분봉 집계 반영)
    if st.button("📚 PRESETS 전체 오늘 시그널 (실시간)", use_container_width=True, key="btn_presets_rt"):
        rows = []
        tz = "America/New_York"
        session_start, session_end = "09:30", "16:00"

        for name, p in PRESETS.items():
            sig_tic = p.get("signal_ticker", p.get("trade_ticker"))

            df0 = get_data(sig_tic, start_date, end_date)
            src = "EOD"
            df_rt = df0.copy()

            if df0 is not None and not df0.empty and not _is_krx_ticker(sig_tic):
                daily_close_1m, last_px, last_ts = get_yf_1m_grouped_close(
                    sig_tic, tz=tz, session_start=session_start, session_end=session_end
                )
                if daily_close_1m is not None and not daily_close_1m.empty and last_ts is not None:
                    df_rt = df_rt.sort_values("Date").drop_duplicates(subset=["Date"]).reset_index(drop=True)
                    df_rt["Date"] = pd.to_datetime(df_rt["Date"])
                    df_rt["Date_only"] = df_rt["Date"].dt.date

                    ts = pd.Timestamp(last_ts)
                    # tz 정규화
                    if ts.tzinfo is None:
                        ts = ts.tz_localize("UTC").tz_convert(tz)
                    else:
                        ts = ts.tz_convert(tz)
                    today_sess_date = ts.date()

                    today_close = daily_close_1m.get(today_sess_date, np.nan)
                    if pd.notna(today_close):
                        if (df_rt["Date_only"] == today_sess_date).any():
                            df_rt.loc[df_rt["Date_only"] == today_sess_date, "Close"] = float(today_close)
                        else:
                            df_rt = pd.concat([df_rt, pd.DataFrame([{
                                "Date": pd.Timestamp(today_sess_date),
                                "Date_only": today_sess_date,
                                "Close": float(today_close)
                            }])], ignore_index=True)

                    df_rt = (df_rt.sort_values("Date")
                                   .drop_duplicates(subset=["Date"])
                                   .drop(columns=["Date_only"], errors="ignore")
                                   .reset_index(drop=True))
                    src = "yfinance_1m_grouped"

            if df_rt is not None and not df_rt.empty:
                p_rt = _today_offsets(p)
                res = summarize_signal_today(df_rt, p_rt)
            else:
                res = {"label": "데이터없음", "last_buy": None, "last_sell": None, "last_hold": None}

            rows.append({
                "전략명": name,
                "티커": sig_tic,
                "시그널": res.get("label","-"),
                "최근 BUY":  res.get("last_buy")  or "-",
                "최근 SELL": res.get("last_sell") or "-",
                "최근 HOLD": res.get("last_hold") or "-",
                "가격소스": src,
            })

        df_rt_view = pd.DataFrame(rows)
        st.dataframe(df_rt_view, use_container_width=True, key="tbl_presets_rt")
        st.download_button(
            "⬇️ CSV 다운로드 (실시간 요약)",
            data=df_rt_view.to_csv(index=False).encode("utf-8-sig"),
            file_name="presets_signal_realtime.csv",
            mime="text/csv",
            key="dl_presets_rt"
        )

# ───────────────────────────────────────
# TAB3: 백테스트
# ───────────────────────────────────────
with tab3:
    # 실행 버튼
    if st.button("✅ 백테스트 실행", use_container_width=True):
        # 1) MA 풀 구성
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
            execution_lag_days=1,
            execution_price_mode="next_close"
        )

        if result:
            st.subheader("📊 백테스트 결과 요약")
            summary = {k: v for k, v in result.items() if k != "매매 로그"}
            st.json(summary)

            colA, colB, colC, colD = st.columns(4)
            colA.metric("총 수익률", f"{summary.get('수익률 (%)', 0)}%")
            colB.metric("승률", f"{summary.get('승률 (%)', 0)}%")
            colC.metric("총 매매 횟수", summary.get("총 매매 횟수", 0))
            colD.metric("MDD", f"{summary.get('MDD (%)', 0)}%")

            df_log = pd.DataFrame(result["매매 로그"])
            df_log["날짜"] = pd.to_datetime(df_log["날짜"])
            df_log.set_index("날짜", inplace=True)

            # 성과지표 보강
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

  ############그래프##############
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

        # ✅ 20일 이동평균선 추가
        df_log["MA20"] = df_log["종가"].rolling(20).mean()
        fig.add_trace(go.Scatter(
            x=df_log.index, y=df_log["MA20"], mode="lines", name="20일 MA",
            yaxis="y2", line=dict(color="orange", dash="dash")
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


########
    with st.expander("🎲 랜덤 시뮬 변수 후보 입력", expanded=False):
        def _parse_list(text, typ="int"):
            if text is None: return []
            toks = [t for t in re.split(r"[,\s]+", str(text).strip()) if t]
            def to_bool(s): return str(s).strip().lower() in ("1","true","t","y","yes","on")
            out = []
            for t in toks:
                if typ == "int":
                    out.append("same" if str(t).lower()=="same" else int(t))
                elif typ == "float":
                    out.append(float(t))
                elif typ == "bool":
                    out.append(to_bool(t))
                else:
                    out.append(str(t))
            seen, dedup = set(), []
            for v in out:
                k = (typ, v)
                if k in seen: continue
                seen.add(k); dedup.append(v)
            return dedup

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
            txt_ma_cmp_s          = st.text_input("ma_compare_short 후보",     "5,10,15,20,25")
            txt_off_cmp_l         = st.text_input("offset_compare_long 후보",  "1,5,15,25")
            txt_ma_cmp_l          = st.text_input("ma_compare_long 후보",      "same")

            txt_use_trend_buy     = st.text_input("use_trend_in_buy 후보(True/False)",  "True,False")
            txt_use_trend_sell    = st.text_input("use_trend_in_sell 후보(True/False)", "True")
            txt_stop_loss         = st.text_input("stop_loss_pct 후보(%)",  "0")
            txt_take_profit       = st.text_input("take_profit_pct 후보(%)","0,10,30")

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

        if st.button("🧪 랜덤 전략 시뮬레이션 실행", use_container_width=True):
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

    with st.expander("🔎 자동 최적 전략 탐색 (Train/Test)", expanded=False):
        st.markdown("""
- 아래 후보군(위 랜덤 시뮬 입력과 동일 포맷)을 토대로 **랜덤 탐색**을 수행해요.  
- 기간을 **Train/Test로 분할**해서 **일반화 성능**을 같이 보여줍니다.  
- 제약조건(최소 매매 횟수, 최소 승률, 최대 MDD)을 걸어 후보를 거르며,  
  정렬은 선택한 **목표 지표의 Test 성과**를 기준으로 합니다.
""")
        colA, colB = st.columns(2)
        with colA:
            split_ratio = st.slider("Train 비중 (나머지 Test)", min_value=0.5, max_value=0.9, value=0.7, step=0.05)
            objective_metric = st.selectbox("목표 지표", ["수익률 (%)", "승률", "샤프", "Profit Factor", "MDD (%)"], index=0)
            objective_mode = "min" if objective_metric == "MDD (%)" else "max"
            n_trials = st.number_input("탐색 시도 횟수 (랜덤)", value=200, min_value=20, step=20)
            topn_show = st.number_input("상위 N개만 표시", value=10, min_value=5, step=5)
        with colB:
            min_trades = st.number_input("제약: 최소 매매 횟수", value=5, min_value=0, step=1)
            min_winrate = st.number_input("제약: 최소 승률(%)", value=55.0, step=1.0)
            max_mdd = st.number_input("제약: 최대 MDD(%) (0=미적용)", value=0.0, step=1.0)
            max_mdd = None if max_mdd == 0.0 else float(max_mdd)

        if st.button("🚀 자동 탐색 실행 (Train/Test)", use_container_width=True):
            # MA 풀: 후보군에서 자동 추출
            ma_pool = set([1, 5, 10, 15, 25])
            for key in ("ma_buy","ma_sell","ma_compare_short","ma_compare_long"):
                for v in choices_dict.get(key, []):
                    if v == "same":
                        continue
                    if isinstance(v, int) and v > 0:
                        ma_pool.add(v)

            if seed:
                random.seed(int(seed))

            df_auto = auto_search_train_test(
                signal_ticker=signal_ticker, trade_ticker=trade_ticker,
                start_date=start_date, end_date=end_date,
                split_ratio=float(split_ratio),
                choices_dict=choices_dict,
                n_trials=int(n_trials),
                objective_metric=objective_metric,
                objective_mode=objective_mode,
                initial_cash=initial_cash_ui,
                fee_bps=fee_bps, slip_bps=slip_bps,
                strategy_behavior=strategy_behavior,
                min_hold_days=min_hold_days,
                execution_lag_days=1,
                execution_price_mode="next_close",
                constraints={"min_trades": int(min_trades), "min_winrate": float(min_winrate), "max_mdd": max_mdd}
            )

            if df_auto.empty:
                st.warning("조건을 만족하는 결과가 없거나 데이터가 부족합니다. 후보군/제약/기간을 조정해 보세요.")
            else:
                st.subheader(f"🏆 자동 탐색 결과 (상위 {topn_show}개, Test {objective_metric} 기준 정렬)")
                st.dataframe(df_auto.head(int(topn_show)))
                csv_auto = df_auto.to_csv(index=False).encode("utf-8-sig")
                st.download_button("⬇️ 자동 탐색 결과 다운로드 (CSV)", data=csv_auto, file_name="auto_search_train_test.csv", mime="text/csv")

                with st.expander("🔥 베스트 파라미터 1개 즉시 적용(선택)", expanded=False):
                    best = df_auto.iloc[0].to_dict()
                    st.write({k: best[k] for k in [
                        "ma_buy","offset_ma_buy","offset_cl_buy","buy_operator",
                        "ma_sell","offset_ma_sell","offset_cl_sell","sell_operator",
                        "use_trend_in_buy","use_trend_in_sell",
                        "ma_compare_short","ma_compare_long",
                        "offset_compare_short","offset_compare_long",
                        "stop_loss_pct","take_profit_pct","min_hold_days"
                    ]})




