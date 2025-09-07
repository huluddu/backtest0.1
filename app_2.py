
# ============================================================
# app_2_improved_full.py  —  FULL PLUS (Integrated Upgrades)
# ------------------------------------------------------------
# 주요 개선/추가
# 1) 최소 수수료(fee_min) + 기존 fee_bps/slip_bps
# 2) SELL 사유 라벨링(규칙매도/signal, 손절, 익절, 트레일링)
# 3) 리스크 지표 보강(Sharpe, Sortino, Calmar, MDD 기간)
# 4) 트레이드 페어/로그 CSV + 설정 스냅샷(JSON) 다운로드
# 5) 유동성 필터(거래량/거래대금/미국 Volume) on/off
# 6) 부분 청산(익절 1단계) + 트레일링 스탑(고점대비 % 하락)
# 7) 체결 지연일(execution_lag_days), 체결가격 모드(next_open/next_close)
# 8) 기존 랜덤 시뮬레이터 유지(핵심 파라미터 기반)
# ------------------------------------------------------------
# 원본 구조와 최대한 호환되도록 작성했습니다.
# ============================================================

import streamlit as st
import pandas as pd
import yfinance as yf
import datetime
import plotly.graph_objects as go
import random
from pykrx import stock
import numpy as np
import re
import json
import math

# ------------------------ 공용 helpers ------------------------
def _fast_ma(x: np.ndarray, w: int) -> np.ndarray:
    if w is None or w <= 1:
        return x.astype(float)
    kernel = np.ones(w, dtype=float) / w
    y = np.full(x.shape, np.nan, dtype=float)
    if len(x) >= w:
        conv = np.convolve(x, kernel, mode="valid")
        y[w-1:] = conv
    return y

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
            if str(t).lower() == "same":   # same 키워드 허용
                out.append("same")
            else:
                out.append(int(t))
        elif typ == "float":
            out.append(float(t))
        elif typ == "bool":
            out.append(to_bool(t))
        else:
            out.append(str(t))
    seen, dedup = set(), []
    for v in out:
        k = (typ, v)
        if k in seen: 
            continue
        seen.add(k); dedup.append(v)
    return dedup

# ------------------------ 데이터 로딩 ------------------------
@st.cache_data(show_spinner=False, ttl=3600)
def get_krx_data_cached(ticker: str, start_date, end_date):
    # ETF OHLCV + 거래대금
    df = stock.get_etf_ohlcv_by_date(
        start_date.strftime("%Y%m%d"),
        end_date.strftime("%Y%m%d"),
        ticker
    )
    df = df.reset_index().rename(columns={
        "날짜": "Date", "시가": "Open", "고가": "High", "저가": "Low", "종가": "Close",
        "거래량": "Volume", "거래대금": "Amount"
    })
    df = df[["Date", "Open", "High", "Low", "Close", "Volume", "Amount"]].dropna()
    return df

@st.cache_data(show_spinner=False, ttl=3600)
def get_yf_data_cached(ticker: str, start_date, end_date):
    df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False)
    if isinstance(df.columns, pd.MultiIndex):
        tu = ticker.upper()
        try:
            o = df[("Open",  tu)]
            h = df[("High",  tu)]
            l = df[("Low",   tu)]
            c = df[("Close", tu)]
            v = df[("Volume", tu)]
            df = pd.concat([o, h, l, c, v], axis=1)
            df.columns = ["Open", "High", "Low", "Close", "Volume"]
        except Exception:
            d2 = df.droplevel(1, axis=1)
            df = d2[["Open", "High", "Low", "Close", "Volume"]]
    else:
        if "Volume" not in df.columns:
            df["Volume"] = np.nan
        df = df[["Open", "High", "Low", "Close", "Volume"]]
    df = df.reset_index().rename(columns={"Date": "Date"})
    df = df[["Date", "Open", "High", "Low", "Close", "Volume"]].dropna()
    return df

def get_data(ticker: str, start_date, end_date) -> pd.DataFrame:
    try:
        if ticker.lower().endswith(".ks") or ticker.isdigit():
            return get_krx_data_cached(ticker, start_date, end_date)
        return get_yf_data_cached(ticker, start_date, end_date)
    except Exception as e:
        st.error(f"❌ 데이터 로딩 실패: {e}")
        return pd.DataFrame()

# ------------------------ 베이스 준비 ------------------------
@st.cache_data(show_spinner=False, ttl=1800)
def prepare_base(signal_ticker, trade_ticker, start_date, end_date, ma_pool):
    sig = get_data(signal_ticker, start_date, end_date).sort_values("Date")
    trd = get_data(trade_ticker,  start_date, end_date).sort_values("Date")

    sig = sig.rename(columns={"Close": "Close_sig"})[["Date", "Close_sig"]]
    # 트레이드는 OHLC + Volume/Amount 사용
    keep = ["Date", "Open", "High", "Low", "Close", "Volume"]
    if "Amount" in trd.columns:
        keep.append("Amount")
    trd = trd[keep].copy().rename(columns={
        "Open": "Open_trd", "High": "High_trd",
        "Low": "Low_trd",   "Close": "Close_trd",
        "Volume": "Volume_trd", "Amount": "Amount_trd"
    })

    base = pd.merge(sig, trd, on="Date", how="inner").dropna(subset=["Close_sig","Open_trd","High_trd","Low_trd","Close_trd"]).reset_index(drop=True)

    x_sig = base["Close_sig"].to_numpy(dtype=float)
    x_trd = base["Close_trd"].to_numpy(dtype=float)

    ma_dict_sig = {}
    for w in sorted(set([w for w in ma_pool if w and w > 0])):
        ma_dict_sig[w] = _fast_ma(x_sig, w)

    return base, x_sig, x_trd, ma_dict_sig

# ------------------------ 리스크 지표 ------------------------
def _drawdown_curve(values: np.ndarray):
    peak = -np.inf
    dd = np.zeros_like(values, dtype=float)
    start_idx, end_idx = 0, 0
    worst = 0.0
    cur_start = 0
    for i, v in enumerate(values):
        if v > peak:
            peak = v
            cur_start = i
        dd[i] = (v - peak) / peak if peak > 0 else 0.0
        if dd[i] < worst:
            worst = dd[i]
            start_idx = cur_start
            end_idx = i
    return dd, worst, start_idx, end_idx

def compute_risk_metrics(equity: pd.Series, periods_per_year: int = 252):
    equity = equity.dropna().astype(float)
    if equity.empty:
        return {}
    rets = equity.pct_change().fillna(0.0).to_numpy()
    avg = np.mean(rets)
    vol = np.std(rets, ddof=1)
    ann_ret = (1.0 + avg) ** periods_per_year - 1.0
    ann_vol = vol * math.sqrt(periods_per_year)
    downside = rets[rets < 0]
    dvol = np.std(downside, ddof=1) if len(downside) > 0 else 0.0
    sortino = (ann_ret) / (dvol * math.sqrt(periods_per_year)) if dvol > 0 else np.nan
    sharpe = (ann_ret) / ann_vol if ann_vol > 0 else np.nan
    _, worst_dd, dd_s, dd_e = _drawdown_curve(equity.to_numpy())
    mdd_pct = abs(worst_dd) * 100.0
    calmar = (ann_ret) / abs(worst_dd) if worst_dd < 0 else np.nan
    return {
        "연율화 수익률(%)": round(ann_ret * 100, 2),
        "연율화 변동성(%)": round(ann_vol * 100, 2),
        "샤프": round(sharpe, 3) if not np.isnan(sharpe) else None,
        "소르티노": round(sortino, 3) if not np.isnan(sortino) else None,
        "MDD(%)": round(mdd_pct, 2),
        "Calmar": round(calmar, 3) if not np.isnan(calmar) else None,
        "MDD_시작_인덱스": int(dd_s),
        "MDD_끝_인덱스": int(dd_e),
    }

# ------------------------ UI: 기본 ------------------------
st.set_page_config(page_title="전략 백테스트 (Full Plus)", layout="wide")
st.title("📊 전략 백테스트 웹앱 — Full Plus")

st.markdown("모든 매매는 **종가 매매**, 손절·익절·트레일링은 **장중(시가/터치)** 기준. "
            "N일 전 데이터 기반의 신호로 금일 종가/다음 시가 체결을 시뮬레이션합니다.")

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
    "TSLL 전략": {
        "signal_ticker": "TSLL", "trade_ticker": "TSLL",
        "offset_cl_buy": 5, "buy_operator": ">", "offset_ma_buy": 15, "ma_buy": 20,
        "offset_cl_sell": 1, "sell_operator": ">", "offset_ma_sell": 25, "ma_sell": 20, 
        "use_trend_in_buy": True, "use_trend_in_sell": True,
        "offset_compare_short": 25, "ma_compare_short": 15,
        "offset_compare_long": 1, "ma_compare_long": 15,         
        "stop_loss_pct": 0.0, "take_profit_pct": 0.0
    },
    "390390 전략": {
        "signal_ticker": "390390", "trade_ticker": "390390",
        "offset_cl_buy": 15, "buy_operator": "<", "offset_ma_buy": 25, "ma_buy": 5,
        "offset_cl_sell": 5, "sell_operator": "<", "offset_ma_sell": 25, "ma_sell": 15, 
        "use_trend_in_buy": True, "use_trend_in_sell": True,
        "offset_compare_short": 25, "ma_compare_short": 15,
        "offset_compare_long": 1, "ma_compare_long": 15,         
        "stop_loss_pct": 0.0, "take_profit_pct": 0.0
    },
}

selected_preset = st.selectbox("🎯 전략 프리셋 선택", ["직접 설정"] + list(PRESETS.keys()))
preset_values = {} if selected_preset == "직접 설정" else PRESETS[selected_preset]

col1, col2 = st.columns(2)
with col1:
    signal_ticker = st.text_input("시그널 판단용 티커", value=preset_values.get("signal_ticker", "SOXL"), key="signal_ticker_input")
with col2:
    trade_ticker = st.text_input("실제 매매 티커", value=preset_values.get("trade_ticker", "SOXL"), key="trade_ticker_input")

col3, col4 = st.columns(2)
with col3:
    start_date = st.date_input("시작일", value=datetime.date(2010, 1, 1),
                               min_value=datetime.date(1990, 1, 1),
                               max_value=datetime.date.today())
with col4:
    end_date = st.date_input("종료일", value=datetime.date.today(),
                             min_value=start_date,
                             max_value=datetime.date.today())

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
    fee_min = st.number_input("최소 수수료(원/달러)", value=0.0, step=1.0)
    slip_bps = st.number_input("슬리피지 (bps)", value=0, step=1)
    execution_price_mode = st.selectbox("체결가격 모드", ["next_close", "next_open"], index=0)
    execution_lag_days = st.number_input("체결 지연일 (신호 후 N일)", value=1, min_value=0, step=1)
    seed = st.number_input("랜덤 시뮬 Seed (재현성)", value=0, step=1)
    if seed:
        random.seed(int(seed))

with st.expander("🧱 유동성 필터(선택)", expanded=False):
    use_liq_filter = st.checkbox("유동성 필터 사용", value=False)
    min_volume = st.number_input("최소 거래량", value=0, min_value=0, step=1000)
    min_amount = st.number_input("최소 거래대금(원)", value=0, min_value=0, step=1000000)

with st.expander("🪓 부분 청산 & 트레일링 스탑(선택)", expanded=False):
    use_scale_out = st.checkbox("부분 청산(익절 1단계) 사용", value=False)
    tp1_pct = st.number_input("익절 1단계 수익률(%)", value=10.0, step=0.5)
    tp1_ratio = st.number_input("익절 1단계 청산 비율(0~1)", value=0.5, min_value=0.0, max_value=1.0, step=0.1)
    use_trailing = st.checkbox("트레일링 스탑 사용", value=False)
    trail_pct = st.number_input("트레일링 폭(%)", value=8.0, step=0.5)

# ---- Safety defaults to avoid NameError on reruns ----
try:
    use_trend_in_buy
except NameError:
    use_trend_in_buy = True
try:
    use_trend_in_sell
except NameError:
    use_trend_in_sell = False


# ------------------------ 오늘의 시그널 ------------------------
def check_signal_today(df,
                       ma_buy, offset_ma_buy, ma_sell, offset_ma_sell,
                       offset_cl_buy, offset_cl_sell,
                       ma_compare_short=None, ma_compare_long=None,
                       offset_compare_short=1, offset_compare_long=1,
                       buy_operator=">", sell_operator="<",
                       use_trend_in_buy=True, use_trend_in_sell=False):
    df = df.copy().sort_values("Date").reset_index(drop=True)
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df["MA_BUY"] = df["Close"].rolling(ma_buy).mean()
    df["MA_SELL"] = df["Close"].rolling(ma_sell).mean()
    if ma_compare_short and ma_compare_long:
        df["MA_SHORT"] = df["Close"].rolling(ma_compare_short).mean()
        df["MA_LONG"]  = df["Close"].rolling(ma_compare_long).mean()
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
            ma_long  = float(df["MA_LONG"].iloc[i - offset_compare_long])
            trend_ok = ma_short >= ma_long
            trend_msg = f"{ma_short:.2f} vs {ma_long:.2f} → {'매수추세' if trend_ok else '매도추세'}"
        except:
            trend_msg = "❗데이터 부족"
    st.write(f"📈 추세 조건: {trend_msg}")
    buy_base  = (cl_b > ma_b) if (buy_operator == ">") else (cl_b < ma_b)
    sell_base = (cl_s < ma_s) if (sell_operator == "<") else (cl_s > ma_s)
    buy_ok  = (buy_base and trend_ok) if use_trend_in_buy  else buy_base
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

# ------------------------ 백테스트 ------------------------
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
    fee_bps=0, slip_bps=0, fee_min=0.0,
    use_trend_in_buy=True,
    use_trend_in_sell=False,
    buy_operator=">", sell_operator="<",
    execution_lag_days=1, execution_price_mode="next_close",
    # 옵션: 유동성/부분청산/트레일링
    use_liq_filter=False, min_volume=0, min_amount=0,
    use_scale_out=False, tp1_pct=10.0, tp1_ratio=0.5,
    use_trailing=False, trail_pct=8.0
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

    xO = base["Open_trd"].to_numpy(dtype=float)
    xH = base["High_trd"].to_numpy(dtype=float)
    xL = base["Low_trd"].to_numpy(dtype=float)

    cash = float(initial_cash)
    position = 0.0
    buy_price = None
    peak_since_entry = None  # 트레일링 기준가
    scaled_once = False      # 부분 청산 1회 여부
    asset_curve, logs = [], []
    sb = strategy_behavior[:1]
    hold_days = 0
    pending_action = None
    pending_due_idx = None

    def _price_next(i):
        return xO[i] if execution_price_mode == "next_open" else x_trd[i]

    def _apply_min_fee_on_buy(fill_px, shares):
        nonlocal position
        notional = fill_px * shares
        proportional_fee = notional * (fee_bps / 10000.0)
        extra = max(fee_min - proportional_fee, 0.0)
        if extra > 0:
            reduce_shares = extra / fill_px
            return max(shares - reduce_shares, 0.0)
        return shares

    def _fill_buy(px: float):
        nonlocal cash, position, buy_price, peak_since_entry, scaled_once
        fill = px * (1 + (slip_bps + fee_bps) / 10000.0)
        if cash <= 0: 
            return None
        shares = cash / fill
        shares = _apply_min_fee_on_buy(fill, shares)
        cost = shares * fill
        cash = 0.0
        position += shares
        buy_price = fill
        peak_since_entry = fill
        scaled_once = False
        return fill

    def _fill_sell(px: float, ratio: float = 1.0):
        nonlocal cash, position, buy_price, peak_since_entry
        if position <= 0 or ratio <= 0:
            return None
        fill = px * (1 - (slip_bps + fee_bps) / 10000.0)
        qty = position * min(ratio, 1.0)
        notional = qty * fill
        cash += notional
        proportional_fee = (qty * px) * (fee_bps / 10000.0)
        extra = max(fee_min - proportional_fee, 0.0)
        if extra > 0:
            cash -= extra
        position -= qty
        if position <= 0:
            position = 0.0
            buy_price = None
            peak_since_entry = None
        return fill

    def _check_intraday_exit(buy_px, o, h, l):
        if buy_px is None:
            return False, False, None
        stop_trigger = False
        take_trigger = False
        fill_px = None
        if stop_loss_pct > 0:
            stop_line = buy_px * (1 - stop_loss_pct / 100.0)
            if o <= stop_line:
                stop_trigger = True; fill_px = o
            elif l <= stop_line:
                stop_trigger = True; fill_px = stop_line
        if take_profit_pct > 0:
            take_line = buy_px * (1 + take_profit_pct / 100.0)
            if o >= take_line:
                if not stop_trigger:
                    take_trigger = True; fill_px = o
            elif h >= take_line:
                if not stop_trigger:
                    take_trigger = True; fill_px = take_line
        if stop_trigger and take_trigger:
            # 동시 터치 시 손절 우선
            fill_px = buy_px * (1 - stop_loss_pct / 100.0)
            take_trigger = False
        return stop_trigger, take_trigger, fill_px

    def _liquidity_ok(i):
        if not use_liq_filter:
            return True
        vol_ok = True
        amt_ok = True
        if "Volume_trd" in base.columns and min_volume > 0:
            v = base["Volume_trd"].iloc[i]
            vol_ok = (pd.notna(v) and v >= min_volume)
        if "Amount_trd" in base.columns and min_amount > 0:
            a = base["Amount_trd"].iloc[i]
            amt_ok = (pd.notna(a) and a >= min_amount)
        return vol_ok and amt_ok

    for i in range(idx0, n):
        just_bought = False
        exec_price = None
        signal = "HOLD"
        sell_reason = ""

        # 예약 체결
        if (pending_action is not None) and (pending_due_idx == i):
            px_base = _price_next(i)
            if pending_action == "BUY" and position == 0.0 and _liquidity_ok(i):
                exec_price = _fill_buy(px_base); just_bought = exec_price is not None
                signal = "BUY" if exec_price is not None else "HOLD"
            elif pending_action == "SELL" and position > 0.0:
                exec_price = _fill_sell(px_base, ratio=1.0)
                signal = "SELL" if exec_price is not None else "HOLD"
                sell_reason = "signal"
            pending_action, pending_due_idx = None, None

        try:
            cl_b = float(x_sig[i - offset_cl_buy])
            ma_b = float(ma_buy_arr[i - offset_ma_buy])
            cl_s = float(x_sig[i - offset_cl_sell])
            ma_s = float(ma_sell_arr[i - offset_ma_sell])
        except Exception:
            total = cash + (position * x_trd[i] if position > 0.0 else 0.0)
            asset_curve.append(total)
            continue

        trend_ok = True
        if (ma_s_arr is not None) and (ma_l_arr is not None):
            ms = ma_s_arr[i - offset_compare_short] if i - offset_compare_short >= 0 else np.nan
            ml = ma_l_arr[i - offset_compare_long]  if i - offset_compare_long  >= 0 else np.nan
            trend_ok = (np.isfinite(ms) and np.isfinite(ml) and ms >= ml)

        open_today  = base["Open_trd"].iloc[i]
        high_today  = base["High_trd"].iloc[i]
        low_today   = base["Low_trd"].iloc[i]
        close_today = x_trd[i]

        # 트레일링 기준 업데이트
        if position > 0.0 and use_trailing:
            if peak_since_entry is None:
                peak_since_entry = close_today
            else:
                peak_since_entry = max(peak_since_entry, high_today)

        buy_base  = (cl_b > ma_b) if (buy_operator == ">") else (cl_b < ma_b)
        sell_base = (cl_s < ma_s) if (sell_operator == "<") else (cl_s > ma_s)
        buy_condition  = (buy_base and trend_ok) if use_trend_in_buy  else buy_base
        sell_condition = (sell_base and (not trend_ok)) if use_trend_in_sell else sell_base

        # 장중 손절/익절 우선
        stop_hit, take_hit, intraday_px = (False, False, None)
        if position > 0.0 and (stop_loss_pct > 0 or take_profit_pct > 0):
            stop_hit, take_hit, intraday_px = _check_intraday_exit(buy_price, open_today, high_today, low_today)

        # 트레일링 스탑(고점대비 절대폭)
        trail_hit = False
        if position > 0.0 and use_trailing and peak_since_entry is not None and trail_pct > 0:
            trail_line = peak_since_entry * (1 - trail_pct / 100.0)
            if open_today <= trail_line:
                trail_hit = True; intraday_px = open_today
            elif low_today <= trail_line:
                trail_hit = True; intraday_px = trail_line

        if position > 0.0 and (stop_hit or take_hit or trail_hit):
            px = intraday_px if intraday_px is not None else close_today
            exec = _fill_sell(px, ratio=1.0)
            if exec is not None:
                signal = "SELL"; exec_price = exec
                sell_reason = "stop_loss" if stop_hit else ("take_profit" if take_hit else "trailing_stop")
                pending_action, pending_due_idx = None, None

        # 부분 청산(익절 1단계) — 장중 TP 라인 터치 시 부분 매도
        if position > 0.0 and use_scale_out and not scaled_once and tp1_pct > 0:
            tp1_line = (buy_price or close_today) * (1 + tp1_pct / 100.0)
            if open_today >= tp1_line:
                ex = _fill_sell(open_today, ratio=tp1_ratio)
                if ex is not None:
                    sell_reason = sell_reason or "scale_out"
                    scaled_once = True
            elif high_today >= tp1_line:
                ex = _fill_sell(tp1_line, ratio=tp1_ratio)
                if ex is not None:
                    sell_reason = sell_reason or "scale_out"
                    scaled_once = True

        def _schedule(action):
            nonlocal pending_action, pending_due_idx
            pending_action = action
            pending_due_idx = i + int(execution_lag_days)

        can_sell  = (position > 0.0) and sell_condition and (hold_days >= min_hold_days)

        if signal == "HOLD":  # 아직 체결되지 않았다면 전략 규칙으로 예약 판단
            if sb == "1":
                if buy_condition and sell_condition:
                    if position == 0.0:
                        if _liquidity_ok(i): _schedule("BUY")
                    else:
                        if can_sell: _schedule("SELL")
                elif position == 0.0 and buy_condition:
                    if _liquidity_ok(i): _schedule("BUY")
                elif can_sell:
                    _schedule("SELL")
            elif sb == "2":
                if buy_condition and sell_condition:
                    if position == 0.0 and _liquidity_ok(i): _schedule("BUY")
                elif position == 0.0 and buy_condition:
                    if _liquidity_ok(i): _schedule("BUY")
                elif can_sell:
                    _schedule("SELL")
            else:  # '3'
                if buy_condition and sell_condition:
                    if position > 0.0 and can_sell: _schedule("SELL")
                elif (position == 0.0) and buy_condition:
                    if _liquidity_ok(i): _schedule("BUY")
                elif can_sell:
                    _schedule("SELL")

        # 보유일 카운트 & 트레일링 peak 갱신
        if position > 0.0:
            if not just_bought:
                hold_days += 1
        else:
            hold_days = 0

        total = cash + (position * close_today if position > 0.0 else 0.0)
        asset_curve.append(total)

        pending_text = None
        if pending_action is not None:
            due_date = base["Date"].iloc[pending_due_idx] if pending_due_idx is not None and pending_due_idx < n else None
            pending_text = f"{pending_action} 예약 (체결일: {due_date.strftime('%Y-%m-%d') if due_date is not None else '범위밖'})"

        logs.append({
            "날짜": pd.to_datetime(base["Date"].iloc[i]).strftime("%Y-%m-%d"),
            "종가": round(close_today, 2),
            "체결가": round(exec_price, 6) if exec_price is not None else None,
            "신호": signal,
            "자산": round(total),
            "매수시그널": bool(buy_condition),
            "매도시그널": bool(sell_condition),
            "손절발동": bool(stop_hit),
            "익절발동": bool(take_hit),
            "트레일발동": bool(trail_hit),
            "추세만족": bool(trend_ok),
            "예약상태": pending_text,
            "보유일": hold_days,
            "사유": sell_reason
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

    # 트레이드 페어 요약 (체결가 우선)
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
    pair_rows = []

    for b, s in trade_pairs:
        pb = b.get("체결가") if b.get("체결가") is not None else b.get("종가")
        ps = s.get("체결가") if s.get("체결가") is not None else s.get("종가")
        if (pb is None) or (ps is None):
            continue
        r = (ps - pb) / pb
        trade_returns.append(r)
        if r >= 0: wins += 1; gross_profit += r
        else: gross_loss += (-r)
        pair_rows.append({
            "진입일": b["날짜"],
            "청산일": s["날짜"],
            "진입가(체결가)": round(pb, 6),
            "청산가(체결가)": round(ps, 6),
            "보유일": s.get("보유일"),
            "수익률(%)": round(r * 100, 3),
            "청산사유": s.get("사유") or ("손절" if s.get("손절발동") else ("익절" if s.get("익절발동") else ("trailing" if s.get("트레일발동") else "signal")))
        })

    total_trades = len(trade_returns)
    win_rate = round((wins / total_trades) * 100, 2) if total_trades else 0.0
    avg_trade_return_pct = round((np.mean(trade_returns) * 100), 2) if trade_returns else 0.0
    median_trade_return_pct = round((np.median(trade_returns) * 100), 2) if trade_returns else 0.0
    profit_factor = round((gross_profit / gross_loss), 3) if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)
    final_asset = float(asset_curve[-1])

    result = {
        "평균 거래당 수익률 (%)": avg_trade_return_pct,
        "수익률 (%)": round((final_asset - float(initial_cash)) / float(initial_cash) * 100, 2),
        "승률 (%)": win_rate,
        "MDD (%)": round(mdd, 2),
        "중앙값 거래당 수익률 (%)": median_trade_return_pct,
        "Profit Factor": profit_factor,
        "총 매매 횟수": total_trades,
        "MDD 발생일": mdd_date.strftime("%Y-%m-%d"),
        "MDD 회복일": recovery_date.strftime("%Y-%m-%d") if recovery_date is not None else "미회복",
        "회복 기간 (일)": (recovery_date - mdd_date).days if recovery_date is not None else None,
        "매매 로그": logs,
        "트레이드 페어": pair_rows,
        "최종 자산": round(final_asset)
    }
    return result

# ------------------------ 실행 & 시각화 ------------------------
if st.button("✅ 백테스트 실행"):
    ma_pool = [ma_buy, ma_sell]
    if (ma_compare_short or 0) > 0: ma_pool.append(ma_compare_short)
    if (ma_compare_long  or 0) > 0: ma_pool.append(ma_compare_long)

    base, x_sig, x_trd, ma_dict_sig = prepare_base(
        signal_ticker, trade_ticker, start_date, end_date, ma_pool
    )

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
        fee_bps=fee_bps, slip_bps=slip_bps, fee_min=fee_min,
        use_trend_in_buy=use_trend_in_buy,
        use_trend_in_sell=use_trend_in_sell,
        buy_operator=buy_operator, sell_operator=sell_operator,
        execution_lag_days=int(execution_lag_days),
        execution_price_mode=execution_price_mode,
        use_liq_filter=use_liq_filter, min_volume=int(min_volume), min_amount=int(min_amount),
        use_scale_out=use_scale_out, tp1_pct=float(tp1_pct), tp1_ratio=float(tp1_ratio),
        use_trailing=use_trailing, trail_pct=float(trail_pct)
    )

    if result:
        st.subheader("📊 백테스트 결과 요약")
        summary = {k: v for k, v in result.items() if k not in ("매매 로그", "트레이드 페어")}
        colA, colB, colC, colD = st.columns(4)
        colA.metric("총 수익률", f"{summary.get('수익률 (%)', 0)}%")
        colB.metric("승률", f"{summary.get('승률 (%)', 0)}%")
        colC.metric("총 매매 횟수", summary.get("총 매매 횟수", 0))
        colD.metric("MDD", f"{summary.get('MDD (%)', 0)}%")
        st.json(summary)

        # 리스크 지표
        df_log = pd.DataFrame(result["매매 로그"])
        df_log["날짜"] = pd.to_datetime(df_log["날짜"])
        df_log.set_index("날짜", inplace=True)
        risk = compute_risk_metrics(df_log["자산"])
        st.write({"Sharpe": risk.get("샤프"), "Sortino": risk.get("소르티노"), "Calmar": risk.get("Calmar")})

        # 차트
        fig = go.Figure()
        bench = initial_cash_ui * (df_log["종가"] / df_log["종가"].iloc[0])
        bh_ret = round((bench.iloc[-1] - initial_cash_ui) / initial_cash_ui * 100, 2)
        fig.add_trace(go.Scatter(x=df_log.index, y=bench, mode="lines", name="Benchmark", yaxis="y1", line=dict(dash="dot")))
        fig.add_trace(go.Scatter(x=df_log.index, y=df_log["자산"], mode="lines", name="Asset", yaxis="y1"))

        pos_step = df_log["신호"].map({"BUY": 1, "SELL": -1}).fillna(0).cumsum()
        in_pos = pos_step > 0
        pos_asset = df_log["자산"].where(in_pos)
        fig.add_trace(go.Scatter(x=df_log.index, y=pos_asset, mode="lines", name="In-Position",
                                 yaxis="y1", line=dict(width=0), fill="tozeroy",
                                 fillcolor="rgba(0,150,0,0.08)", hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=df_log.index, y=df_log["종가"], mode="lines", name="Price", yaxis="y2"))
        buy_points = df_log[df_log["신호"] == "BUY"]
        sell_points = df_log[df_log["신호"] == "SELL"]
        fig.add_trace(go.Scatter(x=buy_points.index, y=buy_points["종가"], mode="markers", name="BUY", yaxis="y2",
                                 marker=dict(color="green", size=6, symbol="triangle-up")))
        fig.add_trace(go.Scatter(x=sell_points.index, y=sell_points["종가"], mode="markers", name="SELL", yaxis="y2",
                                 marker=dict(color="red", size=6, symbol="triangle-down")))
        sl = df_log[df_log["손절발동"] == True]
        tp = df_log[df_log["익절발동"] == True]
        tr = df_log[df_log["트레일발동"] == True]
        if not sl.empty:
            fig.add_trace(go.Scatter(x=sl.index, y=sl["자산"], mode="markers", name="손절", yaxis="y1", marker=dict(symbol="x", size=9)))
        if not tp.empty:
            fig.add_trace(go.Scatter(x=tp.index, y=tp["자산"], mode="markers", name="익절", yaxis="y1", marker=dict(symbol="star", size=10)))
        if not tr.empty:
            fig.add_trace(go.Scatter(x=tr.index, y=tr["자산"], mode="markers", name="트레일", yaxis="y1", marker=dict(symbol="circle-open", size=9)))

        fig.update_layout(title=f"📈 자산 & 종가 흐름 (BUY/SELL 시점 포함) — 벤치마크 수익률 {bh_ret}%",
                          yaxis=dict(title="Asset"), yaxis2=dict(title="Price", overlaying="y", side="right"),
                          hovermode="x unified", height=800)
        st.plotly_chart(fig, use_container_width=True)

        # 트레이드 페어
        if result.get("트레이드 페어"):
            st.subheader("🧾 트레이드 요약 (체결가 기준)")
            df_pairs = pd.DataFrame(result["트레이드 페어"])
            st.dataframe(df_pairs)
            st.download_button("⬇️ 트레이드 페어 다운로드 (CSV)",
                               data=df_pairs.to_csv(index=False).encode("utf-8-sig"),
                               file_name="trade_pairs.csv", mime="text/csv")

        # 로그 다운로드
        with st.expander("🧾 매매 로그"):
            st.dataframe(df_log)
        st.download_button("⬇️ 백테스트 로그 다운로드 (CSV)",
                           data=df_log.reset_index().to_csv(index=False).encode("utf-8-sig"),
                           file_name="backtest_log.csv", mime="text/csv")

        # 설정 스냅샷(JSON)
        cfg = {
            "signal_ticker": signal_ticker, "trade_ticker": trade_ticker,
            "start_date": str(start_date), "end_date": str(end_date),
            "params": {
                "ma_buy": ma_buy, "offset_ma_buy": offset_ma_buy, "offset_cl_buy": offset_cl_buy,
                "ma_sell": ma_sell, "offset_ma_sell": offset_ma_sell, "offset_cl_sell": offset_cl_sell,
                "ma_compare_short": ma_compare_short, "ma_compare_long": ma_compare_long,
                "offset_compare_short": offset_compare_short, "offset_compare_long": offset_compare_long,
                "use_trend_in_buy": use_trend_in_buy, "use_trend_in_sell": use_trend_in_sell,
                "buy_operator": buy_operator, "sell_operator": sell_operator,
                "stop_loss_pct": stop_loss_pct, "take_profit_pct": take_profit_pct,
                "min_hold_days": min_hold_days
            },
            "costs": {"fee_bps": fee_bps, "fee_min": fee_min, "slip_bps": slip_bps},
            "exec": {"execution_price_mode": execution_price_mode, "execution_lag_days": int(execution_lag_days)},
            "filters": {"use_liq_filter": use_liq_filter, "min_volume": int(min_volume), "min_amount": int(min_amount)},
            "scale_trail": {"use_scale_out": use_scale_out, "tp1_pct": float(tp1_pct), "tp1_ratio": float(tp1_ratio),
                            "use_trailing": use_trailing, "trail_pct": float(trail_pct)}
        }
        st.download_button("⬇️ 설정 스냅샷 (JSON)",
                           data=json.dumps(cfg, ensure_ascii=False, indent=2),
                           file_name="run_config_snapshot.json", mime="application/json")

# ------------------------ 랜덤 시뮬 ------------------------
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
    "ma_buy":               _parse_list(locals().get("txt_ma_buy",""), "int"),
    "offset_ma_buy":        _parse_list(locals().get("txt_offset_ma_buy",""), "int"),
    "offset_cl_buy":        _parse_list(locals().get("txt_offset_cl_buy",""), "int"),
    "buy_operator":         _parse_list(locals().get("txt_buy_op",""), "str"),
    "ma_sell":              _parse_list(locals().get("txt_ma_sell",""), "int"),
    "offset_ma_sell":       _parse_list(locals().get("txt_offset_ma_sell",""), "int"),
    "offset_cl_sell":       _parse_list(locals().get("txt_offset_cl_sell",""), "int"),
    "sell_operator":        _parse_list(locals().get("txt_sell_op",""), "str"),
    "use_trend_in_buy":     _parse_list(locals().get("txt_use_trend_buy",""), "bool"),
    "use_trend_in_sell":    _parse_list(locals().get("txt_use_trend_sell",""), "bool"),
    "ma_compare_short":     _parse_list(locals().get("txt_ma_cmp_s",""), "int"),
    "ma_compare_long":      _parse_list(locals().get("txt_ma_cmp_l",""), "int"),
    "offset_compare_short": _parse_list(locals().get("txt_off_cmp_s",""), "int"),
    "offset_compare_long":  _parse_list(locals().get("txt_off_cmp_l",""), "int"),
    "stop_loss_pct":        _parse_list(locals().get("txt_stop_loss",""), "float"),
    "take_profit_pct":      _parse_list(locals().get("txt_take_profit",""), "float"),
}

def run_random_simulations_fast(
    n_simulations, base, x_sig, x_trd, ma_dict_sig,
    initial_cash=5_000_000, fee_bps=0, slip_bps=0,
    choices_dict=None
):
    results = []
    if choices_dict is None:
        choices_dict = {}

    def _pick_one(choices, fallback):
        return random.choice(choices) if choices else fallback

    for _ in range(n_simulations):
        ma_buy_             = _pick_one(choices_dict.get("ma_buy", []),               random.choice([1, 5, 10, 15, 25]))
        offset_ma_buy_      = _pick_one(choices_dict.get("offset_ma_buy", []),        random.choice([1, 5, 15, 25]))
        offset_cl_buy_      = _pick_one(choices_dict.get("offset_cl_buy", []),        random.choice([1, 5, 15, 25]))
        buy_operator_       = _pick_one(choices_dict.get("buy_operator", []),         random.choice([">", "<"]))
        ma_sell_            = _pick_one(choices_dict.get("ma_sell", []),              random.choice([1, 5, 10, 15, 25]))
        offset_ma_sell_     = _pick_one(choices_dict.get("offset_ma_sell", []),       random.choice([1, 5, 15, 25]))
        offset_cl_sell_     = _pick_one(choices_dict.get("offset_cl_sell", []),       random.choice([1, 5, 15, 25]))
        sell_operator_      = _pick_one(choices_dict.get("sell_operator", []),        random.choice(["<", ">"]))
        use_trend_in_buy_   = _pick_one(choices_dict.get("use_trend_in_buy", []),     random.choice([True, False]))
        use_trend_in_sell_  = _pick_one(choices_dict.get("use_trend_in_sell", []),    random.choice([True, False]))
        ma_compare_short_   = _pick_one(choices_dict.get("ma_compare_short", []),     random.choice([1, 5, 15, 25]))
        ma_compare_long_choice = _pick_one(choices_dict.get("ma_compare_long", []),   random.choice([5,15,25]))
        if ma_compare_long_choice == "same":
            ma_compare_long_ = ma_compare_short_
        else:
            ma_compare_long_ = ma_compare_long_choice
        offset_compare_short_= _pick_one(choices_dict.get("offset_compare_short", []), random.choice([1, 15, 25]))
        offset_compare_long_ = _pick_one(choices_dict.get("offset_compare_long", []),  1)
        stop_loss_pct_      = _pick_one(choices_dict.get("stop_loss_pct", []),        0.0)
        take_profit_pct_    = _pick_one(choices_dict.get("take_profit_pct", []),      random.choice([0.0, 25.0, 50.0]))

        for w in [ma_buy_, ma_sell_, ma_compare_short_, ma_compare_long_]:
            if w and w not in ma_dict_sig:
                ma_dict_sig[w] = _fast_ma(x_sig, w)

        r = backtest_fast(
            base, x_sig, x_trd, ma_dict_sig,
            ma_buy_, offset_ma_buy_, ma_sell_, offset_ma_sell_,
            offset_cl_buy_, offset_cl_sell_,
            ma_compare_short_, ma_compare_long_,
            offset_compare_short_, offset_compare_long_,
            initial_cash=initial_cash,
            stop_loss_pct=stop_loss_pct_, take_profit_pct=take_profit_pct_,
            strategy_behavior="1. 포지션 없으면 매수 / 보유 중이면 매도",
            min_hold_days=0,
            fee_bps=fee_bps, slip_bps=slip_bps, fee_min=0.0,
            use_trend_in_buy=use_trend_in_buy_,
            use_trend_in_sell=use_trend_in_sell_,
            buy_operator=buy_operator_, sell_operator=sell_operator_,
            execution_lag_days=1, execution_price_mode="next_close",
            use_liq_filter=False, min_volume=0, min_amount=0,
            use_scale_out=False, tp1_pct=10.0, tp1_ratio=0.5,
            use_trailing=False, trail_pct=8.0
        )
        if not r:
            continue
        result_clean = {k: v for k, v in r.items() if k != "매매 로그" and k != "트레이드 페어"}
        results.append({
            **result_clean,
            "매수종가일": offset_cl_buy_, "매수비교": buy_operator_, "매수이평일": offset_ma_buy_, "매수이평": ma_buy_, 
            "매도종가일": offset_cl_sell_, "매도비교": sell_operator_, "매도이평일": offset_ma_sell_, "매도이평": ma_sell_,
            "매수추세": use_trend_in_buy_, "매도추세": use_trend_in_sell_,
            "과거이평일": offset_compare_short_, "과거이평": ma_compare_short_, "최근이평일": offset_compare_long_, "최근이평": ma_compare_long_,
            "손절": stop_loss_pct_, "익절": take_profit_pct_,
        })
    return pd.DataFrame(results)

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
