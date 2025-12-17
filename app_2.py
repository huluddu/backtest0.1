import streamlit as st
import pandas as pd
import yfinance as yf
import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import random
from pykrx import stock
import numpy as np
import re
import google.generativeai as genai
import json
import os

# ==========================================
# 1. 초기 설정 및 헬퍼 함수
# ==========================================
st.set_page_config(page_title="주식 백테스트 & 시그널 Ultimate", page_icon="📈", layout="wide")

STRATEGY_FILE = "my_strategies.json"

def load_saved_strategies():
    if not os.path.exists(STRATEGY_FILE): return {}
    try:
        with open(STRATEGY_FILE, "r", encoding="utf-8") as f: return json.load(f)
    except: return {}

def save_strategy_to_file(name, params):
    saved = load_saved_strategies()
    saved[name] = params
    with open(STRATEGY_FILE, "w", encoding="utf-8") as f: json.dump(saved, f, ensure_ascii=False, indent=4)

def delete_strategy_from_file(name):
    saved = load_saved_strategies()
    if name in saved:
        del saved[name]
        with open(STRATEGY_FILE, "w", encoding="utf-8") as f: json.dump(saved, f, ensure_ascii=False, indent=4)
        return True
    return False

def _init_default_state():
    defaults = {
        "signal_ticker_input": "SOXL", "trade_ticker_input": "SOXL",
        "buy_operator": ">", "sell_operator": "<",
        "strategy_behavior": "1. 포지션 없으면 매수 / 보유 중이면 매도",
        "offset_cl_buy": 0, "offset_cl_sell": 0,
        "offset_ma_buy": 0, "offset_ma_sell": 0,
        "ma_buy": 20, "ma_sell": 10,
        "use_trend_in_buy": True, "use_trend_in_sell": False,
        "ma_compare_short": 5, "ma_compare_long": 20,
        "offset_compare_short": 0, "offset_compare_long": 0,
        "stop_loss_pct": 0.0, "take_profit_pct": 0.0, "min_hold_days": 0,
        "fee_bps": 25, "slip_bps": 1,
        "preset_name": "직접 설정",
        "gemini_api_key": "",
        "auto_run_trigger": False,
        "use_rsi_filter": False, "rsi_period": 14, "rsi_min": 30, "rsi_max": 70
    }
    for k, v in defaults.items():
        if k not in st.session_state: st.session_state[k] = v

def _apply_preset_to_state(preset: dict | None):
    if not preset: return
    for k, v in preset.items():
        key_name = k if not k.endswith("_ticker") else k.replace("_ticker", "_ticker_input")
        st.session_state[key_name] = v

def _on_preset_change(PRESETS: dict):
    name = st.session_state.get("preset_name", "직접 설정")
    preset = {} if name == "직접 설정" else PRESETS.get(name, {})
    _apply_preset_to_state(preset)

def apply_opt_params(row):
    try:
        updates = {
            "ma_buy": int(row["ma_buy"]), "offset_ma_buy": int(row["offset_ma_buy"]),
            "offset_cl_buy": int(row["offset_cl_buy"]), "buy_operator": str(row["buy_operator"]),
            "ma_sell": int(row["ma_sell"]), "offset_ma_sell": int(row["offset_ma_sell"]),
            "offset_cl_sell": int(row["offset_cl_sell"]), "sell_operator": str(row["sell_operator"]),
            "use_trend_in_buy": bool(row["use_trend_in_buy"]), "use_trend_in_sell": bool(row["use_trend_in_sell"]),
            "ma_compare_short": int(row["ma_compare_short"]) if not pd.isna(row["ma_compare_short"]) else 20,
            "ma_compare_long": int(row["ma_compare_long"]) if not pd.isna(row["ma_compare_long"]) else 50,
            "offset_compare_short": int(row["offset_compare_short"]),
            "offset_compare_long": int(row["offset_compare_long"]),
            "stop_loss_pct": float(row["stop_loss_pct"]),
            "take_profit_pct": float(row["take_profit_pct"]),
            "auto_run_trigger": True
        }
        for k, v in updates.items(): st.session_state[k] = v
        st.session_state["preset_name"] = "직접 설정"
    except Exception as e: st.error(f"설정 적용 오류: {e}")

def _parse_choices(text, cast="int"):
    if text is None: return []
    tokens = [t for t in re.split(r"[,\s]+", str(text).strip()) if t != ""]
    if not tokens: return []
    def _to_bool(s): return s.strip().lower() in ("1", "true", "t", "y", "yes")
    out = []
    for t in tokens:
        if cast == "int": out.append("same" if str(t).lower()=="same" else int(t))
        elif cast == "float": out.append(float(t))
        elif cast == "bool": out.append(_to_bool(t))
        else: out.append(str(t))
    seen = set()
    dedup = []
    for v in out:
        if (v if cast != "str" else (v,)) in seen: continue
        seen.add(v if cast != "str" else (v,))
        dedup.append(v)
    return dedup

def _normalize_krx_ticker(t: str) -> str:
    t = str(t or "").strip().upper()
    m = re.search(r"(\d{6})", t)
    return m.group(1) if m else ""

def _fast_ma(x: np.ndarray, w: int) -> np.ndarray:
    if w is None or w <= 1: return x.astype(float)
    y = pd.Series(x).rolling(window=w).mean().to_numpy()
    return y

# ==========================================
# 2. 데이터 로딩
# ==========================================
@st.cache_data(show_spinner=False, ttl=3600)
def get_data(ticker: str, start_date, end_date) -> pd.DataFrame:
    try:
        t = (ticker or "").strip()
        is_krx = t.isdigit() or t.lower().endswith(".ks") or t.lower().endswith(".kq")
        if is_krx:
            code = _normalize_krx_ticker(t)
            s, e = start_date.strftime("%Y%m%d"), end_date.strftime("%Y%m%d")
            df = stock.get_market_ohlcv_by_date(s, e, code)
            if df.empty: df = stock.get_etf_ohlcv_by_date(s, e, code)
            df = df.reset_index().rename(columns={"날짜":"Date","시가":"Open","고가":"High","저가":"Low","종가":"Close"})
        else:
            df = yf.download(t, start=start_date, end=end_date, progress=False, auto_adjust=False)
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.droplevel(1)
            df = df.reset_index()
            if "Date" in df.columns: df["Date"] = df["Date"].dt.tz_localize(None)

        if df.empty: return pd.DataFrame()
        return df[["Date", "Open", "High", "Low", "Close"]].dropna()
    except: return pd.DataFrame()

@st.cache_data(show_spinner=False, ttl=1800)
def prepare_base(signal_ticker, trade_ticker, start_date, end_date, ma_pool):
    sig = get_data(signal_ticker, start_date, end_date).sort_values("Date")
    trd = get_data(trade_ticker, start_date, end_date).sort_values("Date")
    if sig.empty or trd.empty: return None, None, None, None
    
    sig = sig.rename(columns={"Close": "Close_sig"})[["Date", "Close_sig"]]
    trd = trd.rename(columns={"Open": "Open_trd", "High": "High_trd", "Low": "Low_trd", "Close": "Close_trd"})
    
    base = pd.merge(sig, trd, on="Date", how="inner").dropna().reset_index(drop=True)
    x_sig = base["Close_sig"].to_numpy(dtype=float)
    x_trd = base["Close_trd"].to_numpy(dtype=float)
    
    ma_dict_sig = {int(w): _fast_ma(x_sig, int(w)) for w in ma_pool if w and int(w) > 0}
    return base, x_sig, x_trd, ma_dict_sig

# ==========================================
# 3. 로직 함수
# ==========================================
def ask_gemini_analysis(summary, params, ticker, api_key, model_name):
    if not api_key: return "⚠️ API Key가 없습니다."
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name or "gemini-1.5-flash")
        prompt = f"전문 퀀트 관점에서 분석: {ticker} 전략 결과 {summary}. 파라미터: {params}. 개선점 제안."
        return model.generate_content(prompt).text
    except Exception as e: return f"❌ 오류: {e}"

def check_signal_today(df, ma_buy, offset_ma_buy, ma_sell, offset_ma_sell, offset_cl_buy, offset_cl_sell, ma_compare_short, ma_compare_long, offset_compare_short, offset_compare_long, buy_operator, sell_operator, use_trend_in_buy, use_trend_in_sell):
    if df.empty: st.warning("데이터 없음"); return
    df = df.copy().sort_values("Date").reset_index(drop=True)
    
    # 지표 계산
    df["MA_BUY"] = df["Close"].rolling(ma_buy).mean()
    df["MA_SELL"] = df["Close"].rolling(ma_sell).mean()
    df["MA_S"] = df["Close"].rolling(ma_compare_short).mean()
    df["MA_L"] = df["Close"].rolling(ma_compare_long).mean()
    
    i = len(df) - 1 # 가장 최근 데이터 인덱스
    
    # 백테스트와 동일한 판정 로직 (최근 데이터를 '어제'로 간주하여 시그널 추출)
    try:
        cl_b, ma_b = df["Close"].iloc[i - offset_cl_buy], df["MA_BUY"].iloc[i - offset_ma_buy]
        cl_s, ma_s = df["Close"].iloc[i - offset_cl_sell], df["MA_SELL"].iloc[i - offset_ma_sell]
        
        trend_ok = True
        if use_trend_in_buy or use_trend_in_sell:
            ms, ml = df["MA_S"].iloc[i - offset_compare_short], df["MA_L"].iloc[i - offset_compare_long]
            trend_ok = (ms >= ml)

        buy_cond = (cl_b > ma_b) if buy_operator == ">" else (cl_b < ma_b)
        sell_cond = (cl_s < ma_s) if sell_operator == "<" else (cl_s > ma_s)
        
        is_buy = (buy_cond and trend_ok) if use_trend_in_buy else buy_cond
        is_sell = (sell_cond and not trend_ok) if use_trend_in_sell else sell_cond

        st.subheader(f"📌 시그널 결과 (기준일: {df['Date'].iloc[i].date()})")
        c1, c2, c3 = st.columns(3)
        c1.metric("매수 조건", "만족" if is_buy else "미충족")
        c2.metric("매도 조건", "만족" if is_sell else "미충족")
        c3.metric("추세(정배열)", "YES" if trend_ok else "NO")
        
        if is_buy: st.success("🚀 [매수] 내일 시가 진입 추천")
        elif is_sell: st.error("붕괴 📉 [매도] 내일 시가 청산 추천")
        else: st.info("⏸ [관망] 현재 신호 없음")
    except: st.error("데이터 부족으로 계산 불가")

def backtest_fast(base, x_sig, x_trd, ma_dict_sig, ma_buy, offset_ma_buy, ma_sell, offset_ma_sell, offset_cl_buy, offset_cl_sell, ma_compare_short, ma_compare_long, offset_compare_short, offset_compare_long, initial_cash, stop_loss_pct, take_profit_pct, strategy_behavior, min_hold_days, fee_bps, slip_bps, use_trend_in_buy, use_trend_in_sell, buy_operator, sell_operator, **kwargs):
    n = len(base)
    idx0 = max(ma_buy, ma_sell, ma_compare_short, ma_compare_long, 50) + 2
    
    xO, xH, xL, xC = base["Open_trd"].values, base["High_trd"].values, base["Low_trd"].values, x_trd
    ma_b_arr, ma_s_arr = ma_dict_sig[ma_buy], ma_dict_sig[ma_sell]
    ma_trend_s, ma_trend_l = ma_dict_sig[ma_compare_short], ma_dict_sig[ma_compare_long]
    
    cash, pos, entry_price, hold_days = float(initial_cash), 0.0, 0.0, 0
    logs = []
    
    for i in range(idx0, n):
        # 1. 신호 판정 (중요: i-1 시점의 데이터로 오늘(i) 매매 결정)
        prev = i - 1
        cl_b, ma_b = x_sig[prev - offset_cl_buy], ma_b_arr[prev - offset_ma_buy]
        cl_s, ma_s = x_sig[prev - offset_cl_sell], ma_s_arr[prev - offset_ma_sell]
        
        trend_ok = (ma_trend_s[prev - offset_compare_short] >= ma_trend_l[prev - offset_compare_long])
        
        buy_sig = (cl_b > ma_b) if buy_operator == ">" else (cl_b < ma_b)
        if use_trend_in_buy: buy_sig = buy_sig and trend_ok
            
        sell_sig = (cl_s < ma_s) if sell_operator == "<" else (cl_s > ma_s)
        if use_trend_in_sell: sell_sig = sell_sig and not trend_ok

        # 2. 실행 (오늘 시가)
        today_open = xO[i]
        signal, reason, exec_px = "HOLD", "", 0.0
        
        # 보유 중 - 매도 체크
        if pos > 0:
            hold_days += 1
            # 익절/손절 체크 (고가/저가 기준)
            if stop_loss_pct > 0 and xL[i] <= entry_price * (1 - stop_loss_pct/100):
                signal, reason, exec_px = "SELL", "손절", entry_price * (1 - stop_loss_pct/100)
            elif take_profit_pct > 0 and xH[i] >= entry_price * (1 + take_profit_pct/100):
                signal, reason, exec_px = "SELL", "익절", entry_price * (1 + take_profit_pct/100)
            elif sell_sig and hold_days >= min_hold_days:
                signal, reason, exec_px = "SELL", "전략매도", today_open
            
            if signal == "SELL":
                cash = pos * exec_px * (1 - (fee_bps + slip_bps)/10000)
                pos, hold_days, entry_price = 0.0, 0, 0.0
        
        # 미보유 - 매수 체크
        elif pos == 0 and buy_sig:
            signal, reason, exec_px = "BUY", "전략매수", today_open
            pos = (cash / exec_px) * (1 - (fee_bps + slip_bps)/10000)
            cash, entry_price, hold_days = 0.0, exec_px, 0

        total = cash + (pos * xC[i])
        logs.append({"날짜": base["Date"].iloc[i], "신호": signal, "이유": reason, "체결가": exec_px, "종가": xC[i], "자산": total})

    if not logs: return {}
    df_log = pd.DataFrame(logs)
    ret = (df_log["자산"].iloc[-1] - initial_cash) / initial_cash * 100
    mdd = ((df_log["자산"] - df_log["자산"].cummax()) / df_log["자산"].cummax()).min() * 100
    
    return {"수익률 (%)": round(ret, 2), "MDD (%)": round(mdd, 2), "매매 로그": logs, "최종 자산": df_log["자산"].iloc[-1]}

# ==========================================
# 4. 메인 UI
# ==========================================
_init_default_state()
PRESETS = load_saved_strategies()

# Sidebar & Preset 로직은 기존과 유사하게 유지하되 위 함수들 활용
with st.sidebar:
    st.header("⚙️ 설정")
    api_key = st.text_input("Gemini API Key", type="password", key="gemini_key_input")
    selected_preset = st.selectbox("🎯 프리셋", ["직접 설정"] + list(PRESETS.keys()), key="preset_name", on_change=_on_preset_change, args=(PRESETS,))

# 입력 폼
col1, col2 = st.columns(2)
sig_tk = col1.text_input("시그널 티커", value=st.session_state.signal_ticker_input)
trd_tk = col2.text_input("매매 티커", value=st.session_state.trade_ticker_input)

# [상세 설정 UI - 생략 (기존과 동일)]
# ... (기존 설정 슬라이더 및 넘버인풋 코드들) ...

# 탭 구성
tab1, tab3 = st.tabs(["🎯 실시간 시그널", "🧪 백테스트"])

with tab1:
    if st.button("📌 현재 시그널 확인"):
        df = get_data(sig_tk, datetime.date.today() - datetime.timedelta(days=365), datetime.date.today())
        check_signal_today(df, st.session_state.ma_buy, st.session_state.offset_ma_buy, 
                           st.session_state.ma_sell, st.session_state.offset_ma_sell,
                           st.session_state.offset_cl_buy, st.session_state.offset_cl_sell,
                           st.session_state.ma_compare_short, st.session_state.ma_compare_long,
                           st.session_state.offset_compare_short, st.session_state.offset_compare_long,
                           st.session_state.buy_operator, st.session_state.sell_operator,
                           st.session_state.use_trend_in_buy, st.session_state.use_trend_in_sell)

with tab3:
    if st.button("✅ 백테스트 실행"):
        ma_pool = {st.session_state.ma_buy, st.session_state.ma_sell, st.session_state.ma_compare_short, st.session_state.ma_compare_long}
        base, x_sig, x_trd, ma_dict = prepare_base(sig_tk, trd_tk, datetime.date(2020,1,1), datetime.date.today(), ma_pool)
        
        if base is not None:
            res = backtest_fast(base, x_sig, x_trd, ma_dict, 
                                st.session_state.ma_buy, st.session_state.offset_ma_buy,
                                st.session_state.ma_sell, st.session_state.offset_ma_sell,
                                st.session_state.offset_cl_buy, st.session_state.offset_cl_sell,
                                st.session_state.ma_compare_short, st.session_state.ma_compare_long,
                                st.session_state.offset_compare_short, st.session_state.offset_compare_long,
                                5000000, st.session_state.stop_loss_pct, st.session_state.take_profit_pct,
                                st.session_state.strategy_behavior, st.session_state.min_hold_days,
                                st.session_state.fee_bps, st.session_state.slip_bps,
                                st.session_state.use_trend_in_buy, st.session_state.use_trend_in_sell,
                                st.session_state.buy_operator, st.session_state.sell_operator)
            
            if res:
                st.metric("수익률", f"{res['수익률 (%)']}%")
                st.metric("MDD", f"{res['MDD (%)']}%")
                df_res = pd.DataFrame(res['매매 로그'])
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df_res["날짜"], y=df_res["자산"], name="자산 곡선"))
                st.plotly_chart(fig)
                st.dataframe(df_res[df_res["신호"] != "HOLD"]) # 매매 발생 로그만 표시
