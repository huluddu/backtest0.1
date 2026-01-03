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
st.set_page_config(page_title="QuantLab: 리얼 타임 로직 적용", page_icon="⚡", layout="wide")

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
    st.toast(f"✅ 전략 '{name}' 저장 완료!")

def delete_strategy_from_file(name):
    saved = load_saved_strategies()
    if name in saved:
        del saved[name]
        with open(STRATEGY_FILE, "w", encoding="utf-8") as f: json.dump(saved, f, ensure_ascii=False, indent=4)
        return True
    return False

def _init_default_state():
    if "chat_history" not in st.session_state: st.session_state["chat_history"] = [] # 이 줄 추가
    defaults = {
        "signal_ticker_input": "SOXL", "trade_ticker_input": "SOXL",
        "market_ticker_input": "SPY", 
        "buy_operator": ">", "sell_operator": "<",
        "strategy_behavior": "1. 포지션 없으면 매수 / 보유 중이면 매도",
        "offset_cl_buy": 1, "offset_cl_sell": 1,
        "offset_ma_buy": 1, "offset_ma_sell": 1,
        "ma_buy": 50, "ma_sell": 10,
        "use_trend_in_buy": True, "use_trend_in_sell": False,
        "ma_compare_short": 20, "ma_compare_long": 50,
        "offset_compare_short": 1, "offset_compare_long": 1,
        "stop_loss_pct": 0.0, "take_profit_pct": 0.0, "min_hold_days": 0,
        "fee_bps": 25, "slip_bps": 1,
        "preset_name": "직접 설정",
        "gemini_api_key": "",
        "auto_run_trigger": False,
        "use_rsi_filter": False, "rsi_period": 14, "rsi_min": 30, "rsi_max": 70,
        "use_market_filter": False, "market_ma_period": 200,
        "use_bollinger": False, "bb_period": 20, "bb_std": 2.0,
        "bb_entry_type": "상단선 돌파 (추세)",
        "bb_exit_type": "중심선(MA) 이탈"
    }
    for k, v in defaults.items():
        if k not in st.session_state: st.session_state[k] = v

def _on_preset_change():
    name = st.session_state["preset_name_selector"]
    st.session_state["preset_name"] = name
    
    all_presets = st.session_state.get("ALL_PRESETS_DATA", {})
    preset = all_presets.get(name, {})
    
    if not preset: return

    for k, v in preset.items():
        key_name = k
        if k == "signal_ticker": key_name = "signal_ticker_input"
        elif k == "trade_ticker": key_name = "trade_ticker_input"
        elif k == "market_ticker": key_name = "market_ticker_input"
        
        if key_name in st.session_state:
            st.session_state[key_name] = v

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
            "auto_run_trigger": True,
            "preset_name_selector": "직접 설정"
        }
        for k, v in updates.items(): st.session_state[k] = v
        st.toast("✅ 설정이 적용되었습니다! 백테스트 탭을 확인하세요.")
    except Exception as e: st.error(f"설정 적용 오류: {e}")

def _parse_choices(text, cast="int"):
    if text is None: return []
    tokens = [t for t in re.split(r"[,\s]+", str(text).strip()) if t != ""]
    if not tokens: return []
    def _to_bool(s): return s.strip().lower() in ("1", "true", "t", "y", "yes")
    out = []
    for t in tokens:
        try:
            if cast == "int": out.append("same" if str(t).lower()=="same" else int(t))
            elif cast == "float": out.append(float(t))
            elif cast == "bool": out.append(_to_bool(t))
            else: out.append(str(t))
        except: continue
    seen = set()
    dedup = []
    for v in out:
        if (v if cast != "str" else (v,)) in seen: continue
        seen.add(v if cast != "str" else (v,))
        dedup.append(v)
    return dedup

def _normalize_krx_ticker(t: str) -> str:
    if not isinstance(t, str): t = str(t or "")
    t = t.strip().upper()
    t = re.sub(r"\.(KS|KQ)$", "", t)
    m = re.search(r"(\d{6})", t)
    return m.group(1) if m else ""

def _fast_ma(x: np.ndarray, w: int) -> np.ndarray:
    if w is None or w <= 1: return x.astype(float)
    kernel = np.ones(w, dtype=float) / w
    y = np.full(x.shape, np.nan, dtype=float)
    if len(x) >= w:
        conv = np.convolve(x, kernel, mode="valid")
        y[w-1:] = conv
    return y

def calculate_bollinger_bands(close_data, period, std_dev_mult):
    period = int(period)
    close_series = pd.Series(close_data)
    ma = close_series.rolling(window=period).mean()
    std = close_series.rolling(window=period).std()
    upper = ma + (std * std_dev_mult)
    lower = ma - (std * std_dev_mult)
    return ma.to_numpy(), upper.to_numpy(), lower.to_numpy()

# ==========================================
# 2. 데이터 로딩
# ==========================================
@st.cache_data(show_spinner=False, ttl=3600)
def get_data(ticker: str, start_date, end_date) -> pd.DataFrame:
    try:
        t = (ticker or "").strip()
        if not t: return pd.DataFrame()
        is_krx = t.isdigit() or t.lower().endswith(".ks") or t.lower().endswith(".kq")
        if is_krx:
            code = _normalize_krx_ticker(t)
            s, e = start_date.strftime("%Y%m%d"), end_date.strftime("%Y%m%d")
            df = stock.get_etf_ohlcv_by_date(s, e, code)
            if df is None or df.empty: df = stock.get_market_ohlcv_by_date(s, e, code)
            if not df.empty:
                df = df.reset_index().rename(columns={"날짜":"Date","시가":"Open","고가":"High","저가":"Low","종가":"Close"})
        else:
            df = yf.download(t, start=start_date, end=end_date, progress=False, auto_adjust=False)
            if df.empty:
                df = yf.download(t, period="max", progress=False, auto_adjust=False)
                if not df.empty:
                    df = df[df.index <= pd.Timestamp(end_date)]

            if isinstance(df.columns, pd.MultiIndex):
                try: 
                    if t in df.columns.levels[1]: df = df.xs(t, axis=1, level=1)
                    else: df = df.droplevel(1, axis=1)
                except: df = df.droplevel(1, axis=1)
            
            df = df.reset_index()
            if "Datetime" in df.columns: df.rename(columns={"Datetime": "Date"}, inplace=True)
            if "Date" in df.columns and pd.api.types.is_datetime64_any_dtype(df["Date"]):
                df["Date"] = df["Date"].dt.tz_localize(None)

        if df is None or df.empty: return pd.DataFrame(columns=["Date", "Open", "High", "Low", "Close"])
        cols = ["Open", "High", "Low", "Close"]
        for c in cols:
            if c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce')
        
        return df[["Date", "Open", "High", "Low", "Close"]].dropna()
    except Exception as e:
        return pd.DataFrame(columns=["Date", "Open", "High", "Low", "Close"])

@st.cache_data(show_spinner=False, ttl=1800)
def prepare_base(signal_ticker, trade_ticker, market_ticker, start_date, end_date, ma_pool, market_ma_period=200):
    sig = get_data(signal_ticker, start_date, end_date).sort_values("Date")
    trd = get_data(trade_ticker,  start_date, end_date).sort_values("Date")
    
    if sig.empty or trd.empty: return None, None, None, None, None, None
    
    sig = sig.rename(columns={"Close": "Close_sig", "Open":"Open_sig", "High":"High_sig", "Low":"Low_sig"})[["Date", "Close_sig", "Open_sig", "High_sig", "Low_sig"]]
    trd = trd.rename(columns={"Open": "Open_trd", "High": "High_trd", "Low": "Low_trd", "Close": "Close_trd"})
    
    base = pd.merge(sig, trd, on="Date", how="inner")
    
    x_mkt, ma_mkt_arr = None, None
    if market_ticker:
        mkt = get_data(market_ticker, start_date, end_date).sort_values("Date")
        if not mkt.empty:
            mkt = mkt.rename(columns={"Close": "Close_mkt"})[["Date", "Close_mkt"]]
            base = pd.merge(base, mkt, on="Date", how="inner")
            
    base = base.dropna().reset_index(drop=True)
    
    x_sig = base["Close_sig"].to_numpy(dtype=float)
    x_trd = base["Close_trd"].to_numpy(dtype=float)

    if "Close_mkt" in base.columns:
        x_mkt = base["Close_mkt"].to_numpy(dtype=float)
        ma_mkt_arr = _fast_ma(x_mkt, int(market_ma_period))

    ma_dict_sig = {}
    for w in sorted(set([int(w) for w in ma_pool if w and w > 0])):
        ma_dict_sig[w] = _fast_ma(x_sig, w)
        
    return base, x_sig, x_trd, ma_dict_sig, x_mkt, ma_mkt_arr

# ==========================================
# 3. 로직 함수
# ==========================================
def calculate_indicators(close_data, rsi_period):
    rsi_period = int(rsi_period)
    df = pd.DataFrame({'close': close_data})
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.to_numpy()

def ask_gemini_analysis(summary, params, ticker, api_key, model_name):
    if not api_key: return "⚠️ API Key를 입력해주세요."
    try:
        genai.configure(api_key=api_key)
        m_name = model_name if model_name else "gemini-1.5-flash"
        model = genai.GenerativeModel(m_name)
        
        prompt = f"""
        당신은 상위 1% 퀀트 트레이더입니다. 
        이 전략은 '종가 매매(Market On Close)'를 기준으로 백테스트 되었습니다.

        [투자 대상]: {ticker}
        [전략 설정]: {params}
        
        [백테스트 결과]
        - 수익률: {summary.get('수익률 (%)')}%
        - MDD: {summary.get('MDD (%)')}%
        - 승률: {summary.get('승률 (%)')}%
        - Profit Factor: {summary.get('Profit Factor')}
        - 총 매매 횟수: {summary.get('총 매매 횟수')}회

        [요청사항]
        1. 📊 **성과 진단**: 이 전략의 장점과 치명적인 단점은 무엇인가요?
        2. 🛠️ **튜닝 가이드**: 지표(이평선, 볼린저 등)의 기간을 어떻게 조절하면 좋을까요?
        3. 💡 **종합 평가**: 실전 투자에 적합한가요? (추천/보류/비추천)
        """
        with st.spinner("🤖 Gemini가 전략을 분석 중입니다..."):
            response = model.generate_content(prompt)
            return response.text
    except Exception as e: return f"❌ Gemini 분석 오류: {e}"

def ask_gemini_chat(question, res, params, ticker, api_key, model_name):
    if not api_key: return "⚠️ API Key를 입력해주세요."
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name if model_name else "gemini-1.5-flash")
        context = f"""
        당신은 월스트리트의 상위 1% 퀀트 전문가입니다. 다음 전략 데이터를 바탕으로 사용자의 질문에 답하세요.
        [데이터] 수익률: {res.get('수익률 (%)') or 0}%, MDD: {res.get('MDD (%)') or 0}%, 
        승률: {res.get('승률 (%)') or 0}%, PF: {res.get('Profit Factor') or 0}, 티커: {ticker}
        [설정] {params}
        사용자 질문: {question}
        냉철하고 논리적으로 트레이더의 관점에서 조언하세요.
        """
        response = model.generate_content(context)
        return response.text
    except Exception as e: return f"❌ 오류: {e}"

def check_signal_today(df, ma_buy, offset_ma_buy, ma_sell, offset_ma_sell, offset_cl_buy, offset_cl_sell, ma_compare_short, ma_compare_long, offset_compare_short, offset_compare_long, buy_operator, sell_operator, use_trend_in_buy, use_trend_in_sell,
                       use_market_filter=False, market_ticker="", market_ma_period=200, 
                       use_bollinger=False, bb_period=20, bb_std=2.0, bb_entry_type="상단선 돌파 (추세)", bb_exit_type="중심선(MA) 이탈"):
    if df.empty: st.warning("데이터 없음"); return
    
    has_market = "Close_mkt" in df.columns
    ma_buy, ma_sell = int(ma_buy), int(ma_sell)
    offset_ma_buy, offset_ma_sell = int(offset_ma_buy), int(offset_ma_sell)
    offset_cl_buy, offset_cl_sell = int(offset_cl_buy), int(offset_cl_sell)
    ma_compare_short = int(ma_compare_short) if ma_compare_short else 0
    ma_compare_long = int(ma_compare_long) if ma_compare_long else 0

    df = df.copy().sort_values("Date").reset_index(drop=True)
    df["Close"] = pd.to_numeric(df["Close_sig"], errors="coerce") 
    df["MA_BUY"], df["MA_SELL"] = df["Close"].rolling(ma_buy).mean(), df["Close"].rolling(ma_sell).mean()
    
    if has_market and use_market_filter:
        df["MA_MKT"] = df["Close_mkt"].rolling(int(market_ma_period)).mean()
    
    if use_bollinger:
        m, u, l = calculate_bollinger_bands(df["Close"], bb_period, bb_std)
        df["BB_UP"], df["BB_MID"], df["BB_LO"] = u, m, l

    if ma_compare_short and ma_compare_long:
        df["MA_SHORT"], df["MA_LONG"] = df["Close"].rolling(ma_compare_short).mean(), df["Close"].rolling(ma_compare_long).mean()
    
    i = len(df) - 1
    try:
        if i - max(offset_cl_buy, offset_ma_buy, offset_cl_sell, offset_ma_sell) < 0:
            st.error("데이터 부족"); return
        
        market_ok = True
        if has_market and use_market_filter:
            market_ok = df["Close_mkt"].iloc[i] > df["MA_MKT"].iloc[i]

        cl_b = float(df["Close"].iloc[i - offset_cl_buy])
        cl_s = float(df["Close"].iloc[i - offset_cl_sell])
        ref_date = df["Date"].iloc[-1].strftime('%Y-%m-%d')
        
        buy_ok, sell_ok = False, False
        cond_str, sell_cond_str = "", ""

        if use_bollinger:
            bb_u = float(df["BB_UP"].iloc[i])
            bb_m = float(df["BB_MID"].iloc[i])
            bb_l = float(df["BB_LO"].iloc[i])
            
            if "상단선" in bb_entry_type:
                buy_ok = cl_b > bb_u; cond_str = f"종가 > 상단 {bb_u:.2f}"
            elif "하단선" in bb_entry_type:
                buy_ok = cl_b < bb_l; cond_str = f"종가 < 하단 {bb_l:.2f}"
            else:
                buy_ok = cl_b > bb_m; cond_str = f"종가 > 중심 {bb_m:.2f}"

            if "상단선" in bb_exit_type:
                sell_ok = cl_s < bb_u; sell_cond_str = f"종가 < 상단 {bb_u:.2f}"
            elif "하단선" in bb_exit_type:
                sell_ok = cl_s < bb_l; sell_cond_str = f"종가 < 하단 {bb_l:.2f}"
            else:
                sell_ok = cl_s < bb_m; sell_cond_str = f"종가 < 중심 {bb_m:.2f}"
        else:
            ma_b = float(df["MA_BUY"].iloc[i - offset_ma_buy])
            ma_s = float(df["MA_SELL"].iloc[i - offset_ma_sell])
            trend_ok = True
            if (use_trend_in_buy or use_trend_in_sell) and "MA_SHORT" in df.columns:
                trend_ok = df["MA_SHORT"].iloc[i - offset_compare_short] >= df["MA_LONG"].iloc[i - offset_compare_long]

            buy_base = (cl_b > ma_b) if (buy_operator == ">") else (cl_b < ma_b)
            sell_base = (cl_s < ma_s) if (sell_operator == "<") else (cl_s > ma_s)
            
            buy_ok = (buy_base and trend_ok) if use_trend_in_buy else buy_base
            sell_ok = (sell_base and (not trend_ok)) if use_trend_in_sell else sell_base
            cond_str = f"종가 {cl_b:.2f} {buy_operator} 이평 {ma_b:.2f}"
            sell_cond_str = f"종가 {cl_s:.2f} {sell_operator} 이평 {ma_s:.2f}"

        final_buy = buy_ok and market_ok
        st.subheader(f"📌 시그널 ({ref_date})")
        st.write(f"💡 매수({bb_entry_type if use_bollinger else '이평'}): {cond_str} → {'✅' if buy_ok else '❌'}")
        if buy_ok and not market_ok: st.warning("⚠️ 시장 필터 미충족")
        st.write(f"💡 매도: {sell_cond_str} → {'✅' if sell_ok else '❌'}")
        
        if final_buy: st.success("🚀 매수 진입 (종가)")
        elif sell_ok: st.error("💧 매도 청산 (종가)")
        else: st.info("⏸ 관망")

    except Exception as e: st.error(f"오류: {e}")

def summarize_signal_today(df, p):
    if df is None or df.empty: return {"label": "N/A", "last_buy": "-", "last_sell": "-", "last_hold": "-"}
    return {"label": "확인필요", "last_buy": "-", "last_sell": "-", "last_hold": "-"}

def backtest_fast(base, x_sig, x_trd, ma_dict_sig, ma_buy, offset_ma_buy, ma_sell, offset_ma_sell, offset_cl_buy, offset_cl_sell, ma_compare_short, ma_compare_long, offset_compare_short, offset_compare_long, initial_cash, stop_loss_pct, take_profit_pct, strategy_behavior, min_hold_days, fee_bps, slip_bps, use_trend_in_buy, use_trend_in_sell, buy_operator, sell_operator, 
                  use_rsi_filter=False, rsi_period=14, rsi_min=30, rsi_max=70,
                  use_market_filter=False, x_mkt=None, ma_mkt_arr=None,
                  use_bollinger=False, bb_period=20, bb_std=2.0, 
                  bb_entry_type="상단선 돌파 (추세)", bb_exit_type="중심선(MA) 이탈"):
    n = len(base)
    if n == 0: return {}
    ma_buy_arr, ma_sell_arr = ma_dict_sig.get(int(ma_buy)), ma_dict_sig.get(int(ma_sell))
    ma_s_arr = ma_dict_sig.get(int(ma_compare_short)) if ma_compare_short else None
    ma_l_arr = ma_dict_sig.get(int(ma_compare_long)) if ma_compare_long else None
    rsi_arr = calculate_indicators(x_sig, int(rsi_period)) if use_rsi_filter else None
    
    bb_up, bb_mid, bb_lo = None, None, None
    if use_bollinger: bb_mid, bb_up, bb_lo = calculate_bollinger_bands(x_sig, bb_period, bb_std)

    idx0 = 50
    xC_trd = x_trd
    cash, position, hold_days, entry_price = float(initial_cash), 0.0, 0, 0.0
    logs, asset_curve = [], []

    def _fill(px, type): return px * (1 + (slip_bps + fee_bps)/10000.0) if type=='buy' else px * (1 - (slip_bps + fee_bps)/10000.0)

    for i in range(idx0, n):
        just_bought = False
        exec_price, signal, reason = None, "HOLD", None
        close_today = xC_trd[i]
        # [중요] 손절/익절 체크를 위한 고가/저가 데이터
        open_today, low_today, high_today = base["Open_trd"].iloc[i], base["Low_trd"].iloc[i], base["High_trd"].iloc[i]

        try:
            cl_b, ma_b = x_sig[i - offset_cl_buy], ma_buy_arr[i - offset_ma_buy]
            cl_s, ma_s = x_sig[i - offset_cl_sell], ma_sell_arr[i - offset_ma_sell]
        except: asset_curve.append(cash + position * close_today); continue

        buy_cond, sell_cond = False, False

        if use_bollinger:
            idx_b = i - offset_cl_buy
            idx_s = i - offset_cl_sell
            
            if "상단선" in str(bb_entry_type): buy_cond = cl_b > bb_up[idx_b]
            elif "하단선" in str(bb_entry_type): buy_cond = cl_b < bb_lo[idx_b]
            else: buy_cond = cl_b > bb_mid[idx_b]

            if "상단선" in str(bb_exit_type): sell_cond = cl_s < bb_up[idx_s]
            elif "하단선" in str(bb_exit_type): sell_cond = cl_s < bb_lo[idx_s]
            else: sell_cond = cl_s < bb_mid[idx_s]
        else:
            t_ok = True
            if ma_s_arr is not None: t_ok = ma_s_arr[i-offset_compare_short] >= ma_l_arr[i-offset_compare_long]
            buy_cond = ((cl_b > ma_b) if buy_operator == ">" else (cl_b < ma_b)) and (t_ok if use_trend_in_buy else True)
            sell_cond = ((cl_s < ma_s) if sell_operator == "<" else (cl_s > ma_s)) and ((not t_ok) if use_trend_in_sell else True)

        if buy_cond and use_rsi_filter and rsi_arr[i-1] > rsi_max: buy_cond = False
        if buy_cond and use_market_filter and x_mkt[i] < ma_mkt_arr[i]: buy_cond = False

        stop_hit, take_hit = False, False
        sold_today = False 

        if position > 0:
            # [중요] 손절: 오늘 저가가 손절가 이하로 내려갔는가?
            if stop_loss_pct > 0:
                sl_price = entry_price * (1 - stop_loss_pct / 100)
                if low_today <= sl_price: 
                    stop_hit = True
                    # 갭하락(시초가부터 손절가 밑)이면 시초가, 아니면 손절가에 체결
                    exec_price = open_today if open_today < sl_price else sl_price
            
            # [중요] 익절: 오늘 고가가 목표가 이상으로 올라갔는가?
            if take_profit_pct > 0 and not stop_hit:
                tp_price = entry_price * (1 + take_profit_pct / 100)
                if high_today >= tp_price: 
                    take_hit = True
                    # 갭상승(시초가부터 목표가 위)이면 시초가, 아니면 목표가에 체결
                    exec_price = open_today if open_today > tp_price else tp_price

            if stop_hit or take_hit:
                if not stop_hit and not take_hit: exec_price = close_today 
                cash = position * _fill(exec_price, 'sell')
                position, signal, reason, entry_price = 0.0, "SELL", "손절" if stop_hit else "익절", 0.0
                sold_today = True # 오늘 팔았음

        # [전략 매도] 손절/익절 안 나갔을 때만 종가 체크
        if position > 0 and signal == "HOLD":
            if sell_cond and hold_days >= int(min_hold_days):
                exec_price = close_today
                cash = position * _fill(exec_price, 'sell')
                position, signal, reason, entry_price = 0.0, "SELL", "전략매도", 0.0
                sold_today = True

        # [전략 매수] 오늘 안 팔았을 때만 진입
        elif position == 0 and not sold_today:
            if buy_cond:
                exec_price = close_today
                position = cash / _fill(exec_price, 'buy')
                cash, signal, reason, just_bought, entry_price = 0.0, "BUY", "전략매수", True, exec_price

        hold_days = hold_days + 1 if position > 0 and not just_bought else 0
        total = cash + (position * close_today)
        asset_curve.append(total)
        
        logs.append({
            "날짜": base["Date"].iloc[i], 
            "종가": close_today, 
            "신호": signal, 
            "체결가": exec_price, 
            "자산": total, 
            "이유": reason,
            "손절발동": stop_hit,
            "익절발동": take_hit
        })

    if not logs: return {}
    s = pd.Series(asset_curve)
    
    g_profit, g_loss, wins = 0, 0, 0
    last_buy_price = None
    for r in logs:
        if r['신호'] == 'BUY':
            last_buy_price = r['체결가']
        elif r['신호'] == 'SELL' and last_buy_price:
            pnl = (r['체결가'] - last_buy_price) / last_buy_price
            if pnl > 0:
                wins += 1
                g_profit += pnl
            else:
                g_loss += abs(pnl)
            last_buy_price = None
            
    total_sells = len([l for l in logs if l['신호']=='SELL'])
    pf = (g_profit / g_loss) if g_loss > 0 else 999.0
    win_rate = (wins / total_sells * 100) if total_sells > 0 else 0.0

    return {
        "수익률 (%)": round((asset_curve[-1] - initial_cash)/initial_cash*100, 2),
        "MDD (%)": round(((s - s.cummax()) / s.cummax()).min() * 100, 2),
        "승률 (%)": round(win_rate, 2),
        "Profit Factor": round(pf, 2),
        "총 매매 횟수": total_sells,
        "매매 로그": logs,
        "차트데이터": {"ma_buy_arr": ma_buy_arr[idx0:], "ma_sell_arr": ma_sell_arr[idx0:], "base": base.iloc[idx0:].reset_index(drop=True), "bb_up": bb_up[idx0:] if use_bollinger else None, "bb_lo": bb_lo[idx0:] if use_bollinger else None}
    }

def auto_search_train_test(signal_ticker, trade_ticker, start_date, end_date, split_ratio, choices_dict, n_trials=50, initial_cash=5000000, fee_bps=0, slip_bps=0, strategy_behavior="1", min_hold_days=0, constraints=None, **kwargs):
    ma_pool = set([5, 10, 20, 60, 120])
    for k in ["ma_buy", "ma_sell", "ma_compare_short", "ma_compare_long"]:
        for v in choices_dict.get(k, []):
            try:
                if int(v) > 0: ma_pool.add(int(v))
            except: pass
            
    base_full, x_sig_full, x_trd_full, ma_dict, _, _ = prepare_base(signal_ticker, trade_ticker, "", start_date, end_date, list(ma_pool))
    if base_full is None: return pd.DataFrame()
    
    split_idx = int(len(base_full) * split_ratio)
    base_tr, base_te = base_full.iloc[:split_idx].reset_index(drop=True), base_full.iloc[split_idx:].reset_index(drop=True)
    x_sig_tr, x_sig_te = x_sig_full[:split_idx], x_sig_full[split_idx:]
    x_trd_tr, x_trd_te = x_trd_full[:split_idx], x_trd_full[split_idx:]
    
    results = []
    defaults = {"ma_buy": 50, "ma_sell": 10, "offset_ma_buy": 0, "offset_ma_sell": 0, "offset_cl_buy":0, "offset_cl_sell":0, "buy_operator":">", "sell_operator":"<"}
    
    constraints = constraints or {}
    min_tr = constraints.get("min_trades", 0)
    min_wr = constraints.get("min_winrate", 0)
    limit_mdd = constraints.get("limit_mdd", 0)
    min_train_r = constraints.get("min_train_ret", -999.0)
    min_test_r = constraints.get("min_test_ret", -999.0)

    for _ in range(int(n_trials)):
        p = {}
        for k in choices_dict.keys():
            arr = choices_dict[k]
            p[k] = random.choice(arr) if arr else defaults.get(k)
        
        common_args = {
            "ma_dict_sig": ma_dict,
            "ma_buy": int(p.get('ma_buy', 50)), "offset_ma_buy": int(p.get('offset_ma_buy', 0)),
            "ma_sell": int(p.get('ma_sell', 10)), "offset_ma_sell": int(p.get('offset_ma_sell', 0)),
            "offset_cl_buy": int(p.get('offset_cl_buy', 0)), "offset_cl_sell": int(p.get('offset_cl_sell', 0)),
            "ma_compare_short": int(p.get('ma_compare_short')) if p.get('ma_compare_short') else 0,
            "ma_compare_long": int(p.get('ma_compare_long')) if p.get('ma_compare_long') else 0,
            "offset_compare_short": int(p.get('offset_compare_short', 0)), "offset_compare_long": int(p.get('offset_compare_long', 0)),
            "initial_cash": initial_cash, "stop_loss_pct": float(p.get('stop_loss_pct', 0)), "take_profit_pct": float(p.get('take_profit_pct', 0)),
            "strategy_behavior": strategy_behavior, "min_hold_days": min_hold_days, "fee_bps": fee_bps, "slip_bps": slip_bps,
            "use_trend_in_buy": p.get('use_trend_in_buy', True), "use_trend_in_sell": p.get('use_trend_in_sell', False),
            "buy_operator": p.get('buy_operator', '>'), "sell_operator": p.get('sell_operator', '<')
        }

        res_full = backtest_fast(base_full, x_sig_full, x_trd_full, **common_args)
        if not res_full: continue
        
        if res_full.get('총 매매 횟수', 0) < min_tr: continue
        if res_full.get('승률 (%)', 0) < min_wr: continue
        
        if limit_mdd > 0:
             if res_full.get('MDD (%)', 0) < -abs(limit_mdd): continue

        res_tr = backtest_fast(base_tr, x_sig_tr, x_trd_tr, **common_args)
        if res_tr.get('수익률 (%)', -999) < min_train_r: continue

        res_te = backtest_fast(base_te, x_sig_te, x_trd_te, **common_args)
        if res_te.get('수익률 (%)', -999) < min_test_r: continue

        row = {
            "Full_수익률(%)": res_full.get('수익률 (%)'), "Full_MDD(%)": res_full.get('MDD (%)'), "Full_승률(%)": res_full.get('승률 (%)'), "Full_총매매": res_full.get('총 매매 횟수'),
            "Test_수익률(%)": res_te.get('수익률 (%)'), "Test_MDD(%)": res_te.get('MDD (%)'),
            "Train_수익률(%)": res_tr.get('수익률 (%)'),
            "ma_buy": p.get('ma_buy'), "offset_ma_buy": p.get('offset_ma_buy'), "offset_cl_buy": p.get('offset_cl_buy'), "buy_operator": p.get('buy_operator'),
            "ma_sell": p.get('ma_sell'), "offset_ma_sell": p.get('offset_ma_sell'), "offset_cl_sell": p.get('offset_cl_sell'), "sell_operator": p.get('sell_operator'),
            "use_trend_in_buy": p.get('use_trend_in_buy'), "use_trend_in_sell": p.get('use_trend_in_sell'),
            "ma_compare_short": p.get('ma_compare_short'), "ma_compare_long": p.get('ma_compare_long'), "offset_compare_short": p.get('offset_compare_short'), "offset_compare_long": p.get('offset_compare_long'),
            "stop_loss_pct": p.get('stop_loss_pct'), "take_profit_pct": p.get('take_profit_pct')
        }
        results.append(row)
        
    return pd.DataFrame(results)

# ==========================================
# 5. 메인 UI
# ==========================================
_init_default_state()

PRESETS = {
    "SOXL 도전 전략": {"signal_ticker": "SOXL", "trade_ticker": "SOXL", "offset_cl_buy": 1, "buy_operator": ">", "offset_ma_buy": 1, "ma_buy": 20, "offset_cl_sell": 1, "sell_operator": ">", "offset_ma_sell": 20, "ma_sell": 10, "use_trend_in_buy": True, "use_trend_in_sell": True, "offset_compare_short": 10, "ma_compare_short": 5, "offset_compare_long": 20, "ma_compare_long": 5, "stop_loss_pct": 0.0, "take_profit_pct": 0.0},
    "SOXL 안전 전략": {"signal_ticker": "SOXL", "trade_ticker": "SOXL", "offset_cl_buy": 20, "buy_operator": ">", "offset_ma_buy": 50, "ma_buy": 10, "offset_cl_sell": 50, "sell_operator": ">", "offset_ma_sell": 1, "ma_sell": 10, "use_trend_in_buy": True, "use_trend_in_sell": True, "offset_compare_short": 20, "ma_compare_short": 10, "offset_compare_long": 20, "ma_compare_long": 1, "stop_loss_pct": 35.0, "take_profit_pct": 15.0},
    "TSLL 안전 전략": {"signal_ticker": "TSLL", "trade_ticker": "TSLL", "offset_cl_buy": 20, "buy_operator": "<", "offset_ma_buy": 5, "ma_buy": 10, "offset_cl_sell": 1, "sell_operator": ">", "offset_ma_sell": 1, "ma_sell": 60, "use_trend_in_buy": True, "use_trend_in_sell": True, "offset_compare_short": 20, "ma_compare_short": 50, "offset_compare_long": 20, "ma_compare_long": 5, "stop_loss_pct": 0.0, "take_profit_pct": 20.0},
    "GGLL 전략": {"signal_ticker": "GGLL", "trade_ticker": "GGLL", "offset_cl_buy": 1, "buy_operator": "<", "offset_ma_buy": 1, "ma_buy": 20, "offset_cl_sell": 20, "sell_operator": "<", "offset_ma_sell": 20, "ma_sell": 50, "use_trend_in_buy": True, "use_trend_in_sell": True, "offset_compare_short": 20, "ma_compare_short": 1, "offset_compare_long": 50, "ma_compare_long": 1, "stop_loss_pct": 15.0, "take_profit_pct": 0.0},
    "GGLL 안전 전략": {"signal_ticker": "GGLL", "trade_ticker": "GGLL", "offset_cl_buy": 10, "buy_operator": ">", "offset_ma_buy": 50, "ma_buy": 5, "offset_cl_sell": 10, "sell_operator": "<", "offset_ma_sell": 20, "ma_sell": 20, "use_trend_in_buy": True, "use_trend_in_sell": True, "offset_compare_short": 10, "ma_compare_short": 20, "offset_compare_long": 50, "ma_compare_long": 10, "stop_loss_pct": 20.0, "take_profit_pct": 20.0},
    "BITX 전략": {"signal_ticker": "BITX", "trade_ticker": "BITX", "offset_cl_buy": 16, "buy_operator": ">", "offset_ma_buy": 26, "ma_buy": 5, "offset_cl_sell": 26, "sell_operator": ">", "offset_ma_sell": 2, "ma_sell": 15, "use_trend_in_buy": True, "use_trend_in_sell": True, "offset_compare_short": 26, "ma_compare_short": 15, "offset_compare_long": 6, "ma_compare_long": 15, "stop_loss_pct": 30.0, "take_profit_pct": 0.0},
    "TQQQ 도전 전략": {"signal_ticker": "TQQQ", "trade_ticker": "TQQQ", "offset_cl_buy": 50, "buy_operator": ">", "offset_ma_buy": 10, "ma_buy": 1, "offset_cl_sell": 50, "sell_operator": ">", "offset_ma_sell": 1, "ma_sell": 1, "use_trend_in_buy": True, "use_trend_in_sell": True, "offset_compare_short": 1, "ma_compare_short": 50, "offset_compare_long": 10, "ma_compare_long": 1, "stop_loss_pct": 15.0, "take_profit_pct": 25.0},
    "TQQQ 안전 전략": {"signal_ticker": "TQQQ", "trade_ticker": "TQQQ", "offset_cl_buy": 10, "buy_operator": "<", "offset_ma_buy": 50, "ma_buy": 20, "offset_cl_sell": 50, "sell_operator": ">", "offset_ma_sell": 10, "ma_sell": 20, "use_trend_in_buy": True, "use_trend_in_sell": True, "offset_compare_short": 10, "ma_compare_short": 50, "offset_compare_long": 20, "ma_compare_long": 20, "stop_loss_pct": 25.0, "take_profit_pct": 25.0},
    "BITX-TQQQ 안전": {"signal_ticker": "BITX", "trade_ticker": "TQQQ", "offset_cl_buy": 10, "buy_operator": ">", "offset_ma_buy": 10, "ma_buy": 20, "offset_cl_sell": 50, "sell_operator": ">", "offset_ma_sell": 1, "ma_sell": 5, "use_trend_in_buy": False, "use_trend_in_sell": True, "offset_compare_short": 50, "ma_compare_short": 5, "offset_compare_long": 1, "ma_compare_long": 50, "stop_loss_pct": 0.0, "take_profit_pct": 15.0},
    "453850 ACE 미국30년국채 전략": {"signal_ticker": "453850", "trade_ticker": "453850", "offset_cl_buy": 16, "buy_operator": "<", "offset_ma_buy": 26, "ma_buy": 15, "offset_cl_sell": 26, "sell_operator": ">", "offset_ma_sell": 2, "ma_sell": 20, "use_trend_in_buy": True, "use_trend_in_sell": False, "offset_compare_short": 2, "ma_compare_short": 15, "offset_compare_long": 26, "ma_compare_long": 15, "stop_loss_pct": 0.0, "take_profit_pct": 10.0},
    "465580 ACE미국빅테크TOP7PLUS": {"signal_ticker": "465580", "trade_ticker": "465580", "offset_cl_buy": 2, "buy_operator": ">", "offset_ma_buy": 2, "ma_buy": 5, "offset_cl_sell": 2, "sell_operator": "<", "offset_ma_sell": 2, "ma_sell": 25, "use_trend_in_buy": False, "use_trend_in_sell": True, "offset_compare_short": 6, "ma_compare_short": 10, "offset_compare_long": 2, "ma_compare_long": 10, "stop_loss_pct": 0.0, "take_profit_pct": 10.0},
    "390390 KODEX미국반도체": {"signal_ticker": "390390", "trade_ticker": "390390", "offset_cl_buy": 6, "buy_operator": "<", "offset_ma_buy": 2, "ma_buy": 5, "offset_cl_sell": 26, "sell_operator": ">", "offset_ma_sell": 2, "ma_sell": 20, "use_trend_in_buy": False, "use_trend_in_sell": True, "offset_compare_short": 6, "ma_compare_short": 25, "offset_compare_long": 2, "ma_compare_long": 25, "stop_loss_pct": 0.0, "take_profit_pct": 10.0},
    "371460 TIGER차이나전기차SOLACTIVE": {"signal_ticker": "371460", "trade_ticker": "371460", "offset_cl_buy": 2, "buy_operator": ">", "offset_ma_buy": 6, "ma_buy": 10, "offset_cl_sell": 16, "sell_operator": ">", "offset_ma_sell": 2, "ma_sell": 5, "use_trend_in_buy": True, "use_trend_in_sell": True, "offset_compare_short": 6, "ma_compare_short": 15, "offset_compare_long": 16, "ma_compare_long": 10, "stop_loss_pct": 0.0, "take_profit_pct": 10.0},
    "483280 AITOP10커브드콜": {"signal_ticker": "483280", "trade_ticker": "483280", "offset_cl_buy": 26, "buy_operator": ">", "offset_ma_buy": 26, "ma_buy": 20, "offset_cl_sell": 26, "sell_operator": ">", "offset_ma_sell": 6, "ma_sell": 20, "use_trend_in_buy": True, "use_trend_in_sell": True, "offset_compare_short": 2, "ma_compare_short": 20, "offset_compare_long": 16, "ma_compare_long": 5, "stop_loss_pct": 0.0, "take_profit_pct": 0.0},
}
st.session_state["ALL_PRESETS_DATA"] = PRESETS

with st.sidebar:
    st.header("⚙️ 설정 & Gemini")
    api_key_input = st.text_input("Gemini API Key", type="password", key="gemini_key_input")
    if api_key_input: 
        st.session_state["gemini_api_key"] = api_key_input
        try:
            genai.configure(api_key=api_key_input)
            models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
            st.session_state["selected_model_name"] = st.selectbox("🤖 모델 선택", models, index=0)
        except: st.error("모델 로드 실패")
    
    st.divider()
    with st.expander("💾 전략 저장/삭제"):
        save_name = st.text_input("전략 이름")
        if st.button("현재 설정 저장"):
            if save_name:
                params = {k: st.session_state[k] for k in ["signal_ticker_input","trade_ticker_input","ma_buy","offset_ma_buy","offset_cl_buy","buy_operator","ma_sell","offset_ma_sell","offset_cl_sell","sell_operator","use_trend_in_buy","use_trend_in_sell","ma_compare_short","ma_compare_long","offset_compare_short","offset_compare_long","stop_loss_pct","take_profit_pct","min_hold_days","use_market_filter","market_ticker_input","market_ma_period","use_bollinger","bb_period","bb_std","bb_entry_type","bb_exit_type"]}
                save_strategy_to_file(save_name, params)
                st.rerun()
        
        del_name = st.selectbox("삭제할 전략", list(load_saved_strategies().keys())) if load_saved_strategies() else None
        if del_name and st.button("삭제"):
            delete_strategy_from_file(del_name)
            st.rerun()

    st.divider()
    selected_preset = st.selectbox(
        "🎯 프리셋", 
        ["직접 설정"] + list(PRESETS.keys()), 
        key="preset_name_selector", 
        on_change=_on_preset_change
    )

col1, col2, col3 = st.columns(3)
signal_ticker = col1.text_input("시그널 티커", key="signal_ticker_input")
trade_ticker = col2.text_input("매매 티커", key="trade_ticker_input")
market_ticker = col3.text_input("시장 티커 (옵션)", key="market_ticker_input", help="예: SPY")

col4, col5 = st.columns(2)
start_date = col4.date_input("시작일", value=datetime.date(2020, 1, 1))
end_date = col5.date_input("종료일", value=datetime.date.today())

with st.expander("📈 상세 설정 (Offset, 비용 등)", expanded=True):
    tabs = st.tabs(["📊 이평선 설정", "🚦 시장 필터", "🌊 볼린저 밴드", "🛡️ 리스크/기타"])

    with tabs[0]:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### 📥 매수")
            ma_buy = st.number_input("매수 이평", key="ma_buy", step=1, min_value=1)
            offset_ma_buy = st.number_input("매수 이평 Offset", key="offset_ma_buy", step=1)
            offset_cl_buy = st.number_input("매수 종가 Offset", key="offset_cl_buy", step=1)
            buy_operator = st.selectbox("매수 부호", [">", "<"], key="buy_operator")
            use_trend_in_buy = st.checkbox("매수 추세 필터", key="use_trend_in_buy")
        with c2:
            st.markdown("#### 📤 매도")
            ma_sell = st.number_input("매도 이평", key="ma_sell", step=1, min_value=1)
            offset_ma_sell = st.number_input("매도 이평 Offset", key="offset_ma_sell", step=1)
            offset_cl_sell = st.number_input("매도 종가 Offset", key="offset_cl_sell", step=1)
            sell_operator = st.selectbox("매도 부호", ["<", ">"], key="sell_operator")
            use_trend_in_sell = st.checkbox("매도 역추세 필터", key="use_trend_in_sell")
        
        st.divider()
        c3, c4 = st.columns(2)
        with c3:
            st.markdown("#### 📈 추세선")
            ma_compare_short = st.number_input("추세 Short", key="ma_compare_short", step=1, min_value=1)
            offset_compare_short = st.number_input("추세 Short Offset", key="offset_compare_short", step=1)
        with c4:
            st.markdown("#### .")
            ma_compare_long = st.number_input("추세 Long", key="ma_compare_long", step=1, min_value=1)
            offset_compare_long = st.number_input("추세 Long Offset", key="offset_compare_long", step=1)

    with tabs[1]:
        st.markdown("#### 🚦 시장 필터 (Market Filter)")
        st.write("시장 지수(예: SPY)가 이평선 위에 있을 때만 매수합니다.")
        use_market_filter = st.checkbox("시장 필터 사용", key="use_market_filter")
        market_ma_period = st.number_input("시장 이평선 기간", value=200, step=10, key="market_ma_period")

    with tabs[2]:
        st.markdown("#### 🌊 볼린저 밴드 (Volatility Breakout)")
        st.write("이평선 매매 대신 볼린저 밴드 돌파 전략을 사용합니다.")
        use_bollinger = st.checkbox("볼린저 밴드 사용", key="use_bollinger")
        c_b1, c_b2 = st.columns(2)
        bb_period = c_b1.number_input("밴드 기간", value=20, key="bb_period")
        bb_std = c_b2.number_input("밴드 승수 (Std Dev)", value=2.0, step=0.1, key="bb_std")
        bb_entry_type = st.selectbox("매수 기준", ["상단선 돌파 (추세)", "하단선 이탈 (역추세)", "중심선 돌파"], key="bb_entry_type")
        bb_exit_type = st.selectbox("매도 기준", ["중심선(MA) 이탈", "상단선 복귀", "하단선 이탈"], key="bb_exit_type")
        if use_bollinger:
            st.info("ℹ️ 활성화 시 '이평선 매매' 조건은 무시됩니다.")

    with tabs[3]:
        c5, c6 = st.columns(2)
        with c5:
            st.markdown("#### 🛡️ 리스크")
            stop_loss_pct = st.number_input("손절 (%)", step=0.5, key="stop_loss_pct")
            take_profit_pct = st.number_input("익절 (%)", step=0.5, key="take_profit_pct")
            min_hold_days = st.number_input("최소 보유일", step=1, key="min_hold_days")
        with c6:
            st.markdown("#### ⚙️ 기타")
            strategy_behavior = st.selectbox("행동 패턴", ["1. 포지션 없으면 매수 / 보유 중이면 매도", "2. 매수 우선", "3. 관망"], key="strategy_behavior")
            fee_bps = st.number_input("수수료 (bps)", value=25, step=1, key="fee_bps")
            slip_bps = st.number_input("슬리피지 (bps)", value=1, step=1, key="slip_bps")
            seed = st.number_input("랜덤 시드", value=0, step=1)
            if seed > 0: random.seed(seed)
        
        st.divider()
        st.markdown("#### 🔮 보조지표 설정")
        c_r1, c_r2 = st.columns(2)
        rsi_p = c_r1.number_input("RSI 기간 (Period)", 14, step=1, key="rsi_period")
        u_rsi = st.checkbox("RSI 필터 적용 (매수시 과열 방지)", key="use_rsi_filter")
        if u_rsi:
            rsi_max = c_r2.number_input("RSI 과매수 기준", 70, key="rsi_max")

tab1, tab2, tab3, tab4 = st.tabs(["🎯 시그널", "📚 PRESETS", "🧪 백테스트", "🧬 실험실"])

with tab1:
    if st.button("📌 오늘의 매매 시그널 확인", type="primary", use_container_width=True):
        base, x_sig, x_trd, ma_dict, x_mkt, ma_mkt_arr = prepare_base(signal_ticker, trade_ticker, market_ticker, start_date, end_date, [ma_buy, ma_sell], market_ma_period)
        if base is not None:
             check_signal_today(base, ma_buy, offset_ma_buy, ma_sell, offset_ma_sell, offset_cl_buy, offset_cl_sell, ma_compare_short, ma_compare_long, offset_compare_short, offset_compare_long, buy_operator, sell_operator, use_trend_in_buy, use_trend_in_sell,
                                use_market_filter, market_ticker, market_ma_period, 
                                use_bollinger, bb_period, bb_std, bb_entry_type, bb_exit_type)
        else: st.error("데이터 로딩 실패")

with tab2:
    if st.button("📚 모든 프리셋 일괄 점검"):
        rows = []
        with st.spinner("모든 전략을 시뮬레이션 중입니다..."):
            for name, p in PRESETS.items():
                t = p.get("signal_ticker", p.get("trade_ticker"))
                res = summarize_signal_today(get_data(t, start_date, end_date), p)
                rows.append({
                    "전략": name, "티커": t, "시그널": res["label"], 
                    "최근 BUY": res["last_buy"], "최근 SELL": res["last_sell"], "최근 HOLD": res["last_hold"]
                })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

with tab3:
    if st.button("✅ 백테스트 실행 (종가매매)", type="primary", use_container_width=True):
        p_ma_buy = int(ma_buy)
        p_ma_sell = int(ma_sell)
        p_ma_compare_short = int(ma_compare_short) if ma_compare_short else 0
        p_ma_compare_long = int(ma_compare_long) if ma_compare_long else 0
        
        ma_pool = [p_ma_buy, p_ma_sell, p_ma_compare_short, p_ma_compare_long]
        base, x_sig, x_trd, ma_dict, x_mkt, ma_mkt_arr = prepare_base(signal_ticker, trade_ticker, market_ticker, start_date, end_date, ma_pool, market_ma_period)
        
        if base is not None:
            with st.spinner("과거 데이터를 한 땀 한 땀 분석 중..."):
                p_use_rsi = st.session_state.get("use_rsi_filter", False)
                p_rsi_period = st.session_state.get("rsi_period", 14)
                p_rsi_max = st.session_state.get("rsi_max", 70)

                res = backtest_fast(base, x_sig, x_trd, ma_dict, p_ma_buy, offset_ma_buy, p_ma_sell, offset_ma_sell, offset_cl_buy, offset_cl_sell, p_ma_compare_short, p_ma_compare_long, offset_compare_short, offset_compare_long, 5000000, stop_loss_pct, take_profit_pct, strategy_behavior, min_hold_days, fee_bps, slip_bps, use_trend_in_buy, use_trend_in_sell, buy_operator, sell_operator, 
                                use_rsi_filter=p_use_rsi, rsi_period=p_rsi_period, rsi_min=30, rsi_max=p_rsi_max,
                                use_market_filter=use_market_filter, x_mkt=x_mkt, ma_mkt_arr=ma_mkt_arr,
                                use_bollinger=use_bollinger, bb_period=bb_period, bb_std=bb_std, 
                                bb_entry_type=bb_entry_type, bb_exit_type=bb_exit_type)
            st.session_state["bt_result"] = res
            if "ai_analysis" in st.session_state: del st.session_state["ai_analysis"]
            st.rerun()
        else: st.error("데이터 로딩 실패")

    if "bt_result" in st.session_state:
        res = st.session_state["bt_result"]
        if res:
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("총 수익률", f"{res['수익률 (%)']}%", delta_color="normal")
            k2.metric("MDD (최대낙폭)", f"{res['MDD (%)']}%", delta_color="inverse")
            k3.metric("승률", f"{res['승률 (%)']}%")
            k4.metric("Profit Factor", res['Profit Factor'])
            
            df_log = pd.DataFrame(res['매매 로그'])
            if not df_log.empty:
                initial_price = df_log['종가'].iloc[0]
                benchmark = (df_log['종가'] / initial_price) * 5000000
                drawdown = (df_log['자산'] - df_log['자산'].cummax()) / df_log['자산'].cummax() * 100

                chart_data = res.get("차트데이터", {})
                base_df = chart_data.get("base")
                
                fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.5, 0.25, 0.25], 
                                    subplot_titles=("주가 & 매매타점 (Candle + MA)", "내 자산 vs 보유 전략 (Equity)", "MDD (%)"))

                if base_df is not None:
                    fig.add_trace(go.Candlestick(x=base_df['Date'], open=base_df['Open_sig'], high=base_df['High_sig'], low=base_df['Low_sig'], close=base_df['Close_sig'], name='가격(Signal)'), row=1, col=1)
                    
                    if use_bollinger and chart_data.get("bb_up") is not None:
                        fig.add_trace(go.Scatter(x=base_df['Date'], y=chart_data['bb_up'], name='BB 상단', line=dict(color='gray', width=1, dash='dot')), row=1, col=1)
                        fig.add_trace(go.Scatter(x=base_df['Date'], y=chart_data['bb_lo'], name='BB 하단', line=dict(color='gray', width=1, dash='dot'), fill='tonexty'), row=1, col=1)
                    else:
                        fig.add_trace(go.Scatter(x=base_df['Date'], y=chart_data['ma_buy_arr'], name='매수 기준선(MA)', line=dict(color='orange', width=1)), row=1, col=1)
                        fig.add_trace(go.Scatter(x=base_df['Date'], y=chart_data['ma_sell_arr'], name='매도 기준선(MA)', line=dict(color='blue', width=1, dash='dot')), row=1, col=1)

                buys = df_log[df_log['신호']=='BUY']
                
                # [FIXED] 필터링 조건에서 컬럼 존재 여부 확인 없이 사용하여 에러 발생 가능성 차단
                # 모든 로그에는 '손절발동' 키가 있으므로 안전
                sells_reg = df_log[(df_log['신호']=='SELL') & (df_log['손절발동']==False) & (df_log['익절발동']==False)]
                sl = df_log[df_log['손절발동']==True]
                tp = df_log[df_log['익절발동']==True]

                fig.add_trace(go.Scatter(x=buys['날짜'], y=buys['종가'], mode='markers', marker=dict(color='#00FF00', symbol='triangle-up', size=12), name='매수 체결'), row=1, col=1)
                fig.add_trace(go.Scatter(x=sells_reg['날짜'], y=sells_reg['종가'], mode='markers', marker=dict(color='red', symbol='triangle-down', size=12), name='매도 체결'), row=1, col=1)
                fig.add_trace(go.Scatter(x=sl['날짜'], y=sl['종가'], mode='markers', marker=dict(color='purple', symbol='x', size=12), name='손절'), row=1, col=1)
                fig.add_trace(go.Scatter(x=tp['날짜'], y=tp['종가'], mode='markers', marker=dict(color='gold', symbol='star', size=15), name='익절'), row=1, col=1)

                fig.add_trace(go.Scatter(x=df_log['날짜'], y=df_log['자산'], name='내 전략 자산', line=dict(color='#00F0FF', width=2)), row=2, col=1)
                fig.add_trace(go.Scatter(x=df_log['날짜'], y=benchmark, name='단순 보유(Buy&Hold)', line=dict(color='gray', dash='dot')), row=2, col=1)

                fig.add_trace(go.Scatter(x=df_log['날짜'], y=drawdown, name='MDD', line=dict(color='#FF4B4B', width=1), fill='tozeroy'), row=3, col=1)

                fig.update_layout(height=900, template="plotly_dark", hovermode="x unified", xaxis_rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("### 📅 월별 수익률 Heatmap")
                df_log['Year'] = df_log['날짜'].dt.year
                df_log['Month'] = df_log['날짜'].dt.month
                df_log['Returns'] = df_log['자산'].pct_change()
                monthly_ret = df_log.groupby(['Year', 'Month'])['Returns'].apply(lambda x: (x + 1).prod() - 1).reset_index()
                pivot_ret = monthly_ret.pivot(index='Year', columns='Month', values='Returns')
                fig_heat = go.Figure(data=go.Heatmap(
                    z=pivot_ret.values * 100, x=pivot_ret.columns, y=pivot_ret.index,
                    colorscale='RdBu', zmid=0, texttemplate="%{z:.1f}%"
                ))
                fig_heat.update_layout(height=400, margin=dict(t=30, b=30))
                st.plotly_chart(fig_heat, use_container_width=True)

                # tab3 결과 화면 가장 아래쪽에 추가
                st.divider()
                st.markdown("### 🤖 제미니 퀀트 컨설턴트 (1:1 대화)")
        
                # 채팅 기록 표시용 컨테이너
                chat_container = st.container(height=300)
                for msg in st.session_state["chat_history"]:
                    with chat_container.chat_message(msg["role"]):
                        st.write(msg["content"])

                # 채팅 입력창
                if prompt := st.chat_input("전략에 대해 질문하세요! (예: 왜 보류 등급이야?, 승률 높이는 법?)"):
                    st.session_state["chat_history"].append({"role": "user", "content": prompt})
                    with chat_container.chat_message("user"): st.write(prompt)
            
                    with chat_container.chat_message("assistant"):
                        current_p = f"매수:{ma_buy}MA, 매도:{ma_sell}MA, 손절:{stop_loss_pct}%"
                        response = ask_gemini_chat(prompt, res, current_p, trade_ticker, st.session_state["gemini_api_key"], st.session_state.get("selected_model_name"))
                        st.write(response)
                        st.session_state["chat_history"].append({"role": "assistant", "content": response})

                st.markdown("### 💾 결과 저장")
                csv = df_log.to_csv(index=False).encode('utf-8-sig')
                st.download_button(
                    label="📥 매매 로그 다운로드 (CSV)",
                    data=csv,
                    file_name=f'backtest_log_{trade_ticker}_{datetime.date.today()}.csv',
                    mime='text/csv',
                )

                st.divider()
                st.markdown("### 🤖 Gemini AI 전략 컨설팅")
                if st.button("✨ AI에게 분석 및 개선점 물어보기", type="primary"):
                    sl_txt = f"{stop_loss_pct}%" if stop_loss_pct > 0 else "미설정"
                    tp_txt = f"{take_profit_pct}%" if take_profit_pct > 0 else "미설정"
                    current_params = f"매수: {ma_buy}일 이평, 매도: {ma_sell}일 이평, 손절: {sl_txt}, 익절: {tp_txt}"
                    anl = ask_gemini_analysis(res, current_params, trade_ticker, st.session_state.get("gemini_api_key"), st.session_state.get("selected_model_name", "gemini-1.5-flash"))
                    st.session_state["ai_analysis"] = anl       
                
                if "ai_analysis" in st.session_state:
                    st.info(st.session_state["ai_analysis"])
                
                with st.expander("📝 상세 로그 보기"):
                    st.dataframe(df_log, use_container_width=True)
        else:
            st.warning("⚠️ 매매 신호가 발생하지 않았습니다. (조건을 완화하거나 기간을 늘려보세요)")

with tab4:
    st.markdown("### 🧬 전략 파라미터 자동 최적화 (Grid Search)")
    st.caption("여러 설정을 자동으로 돌려보고 가장 좋은 수익률을 찾아냅니다.")
    
    with st.expander("🔎 필터 및 정렬 설정", expanded=True):
        c1, c2 = st.columns(2)
        sort_metric = c1.selectbox("정렬 기준", ["Full_수익률(%)", "Test_수익률(%)", "Full_MDD(%)", "Full_승률(%)"])
        top_n = c2.slider("표시할 상위 개수", 1, 50, 10)
        
        c3, c4 = st.columns(2)
        min_trades = c3.number_input("최소 매매 횟수", 0, 100, 5)
        min_win = c4.number_input("최소 승률 (%)", 0.0, 100.0, 50.0)
        
        c5, c6 = st.columns(2)
        min_train_ret = c5.number_input("최소 Train 수익률 (%)", -100.0, 1000.0, 0.0)
        min_test_ret = c6.number_input("최소 Test 수익률 (%)", -100.0, 1000.0, 0.0)
        
        # [MODIFIED] 절대값 설명 추가
        limit_mdd = st.number_input("최대 낙폭(MDD) 한계 (%, 절대값)", 
                                    min_value=0.0, max_value=100.0, value=0.0, step=1.0,
                                    help="예: 20을 입력하면 -20%보다 낙폭이 큰 전략은 제외합니다.")

    colL, colR = st.columns(2)
    with colL:
        st.markdown("#### 1. 매수/매도 조건")
        cand_off_cl_buy = st.text_input("매수 종가 Offset", "1, 5, 10, 20, 50")
        cand_buy_op = st.text_input("매수 부호", "<,>")
        cand_off_ma_buy = st.text_input("매수 이평 Offset", "1, 5, 10, 20, 50")
        cand_ma_buy = st.text_input("매수 이평 (MA Buy)", "1, 5, 10, 20, 50, 60, 120")
        
        st.divider()
        cand_off_cl_sell = st.text_input("매도 종가 Offset", "1, 5, 10, 20, 50")
        cand_sell_op = st.text_input("매도 부호", "<,>")
        cand_off_ma_sell = st.text_input("매도 이평 Offset", "1, 5, 10, 20, 50")
        cand_ma_sell = st.text_input("매도 이평 (MA Sell)", "1, 5, 10, 20, 50, 60, 120")

    with colR:
        st.markdown("#### 2. 추세 & 리스크")
        cand_use_tr_buy = st.text_input("매수 추세필터 (True, False)", "True, False")
        cand_use_tr_sell = st.text_input("매도 역추세필터", "True")
        
        cand_ma_s = st.text_input("추세 Short 후보", "1, 5, 10, 20, 50, 60, 120")
        cand_ma_l = st.text_input("추세 Long 후보", "1, 5, 10, 20, 50, 60, 120")
        cand_off_s = st.text_input("추세 Short Offset", "1, 5, 10, 20, 50")
        cand_off_l = st.text_input("추세 Long Offset", "1, 5, 10, 20, 50")
        
        st.divider()
        cand_stop = st.text_input("손절(%) 후보", "0, 5, 10, 20")
        cand_take = st.text_input("익절(%) 후보", "0, 10, 20")

    n_trials = st.number_input("시도 횟수", 10, 500, 50)
    split_ratio = st.slider("Train 비율", 0.5, 0.9, 0.7)
    
    if st.button("🚀 최적 조합 찾기 시작"):
        choices = {
            "ma_buy": _parse_choices(cand_ma_buy, "int"), "offset_ma_buy": _parse_choices(cand_off_ma_buy, "int"),
            "offset_cl_buy": _parse_choices(cand_off_cl_buy, "int"), "buy_operator": _parse_choices(cand_buy_op, "str"),
            "ma_sell": _parse_choices(cand_ma_sell, "int"), "offset_ma_sell": _parse_choices(cand_off_ma_sell, "int"),
            "offset_cl_sell": _parse_choices(cand_off_cl_sell, "int"), "sell_operator": _parse_choices(cand_sell_op, "str"),
            "use_trend_in_buy": _parse_choices(cand_use_tr_buy, "bool"), "use_trend_in_sell": _parse_choices(cand_use_tr_sell, "bool"),
            "ma_compare_short": _parse_choices(cand_ma_s, "int"), "ma_compare_long": _parse_choices(cand_ma_l, "int"),
            "offset_compare_short": _parse_choices(cand_off_s, "int"), "offset_compare_long": _parse_choices(cand_off_l, "int"),
            "stop_loss_pct": _parse_choices(cand_stop, "float"), "take_profit_pct": _parse_choices(cand_take, "float"),
        }
        
        constraints = {
            "min_trades": min_trades,
            "min_winrate": min_win,
            "limit_mdd": limit_mdd,
            "min_train_ret": min_train_ret,
            "min_test_ret": min_test_ret
        }
        
        with st.spinner("AI가 최적의 파라미터를 탐색 중입니다..."):
            df_opt = auto_search_train_test(
                signal_ticker, trade_ticker, start_date, end_date, split_ratio, choices, 
                n_trials=int(n_trials), initial_cash=5000000, 
                fee_bps=fee_bps, slip_bps=slip_bps, strategy_behavior=strategy_behavior, min_hold_days=min_hold_days,
                constraints=constraints
            )
            
            if not df_opt.empty:
                for col in df_opt.columns:
                    df_opt[col] = pd.to_numeric(df_opt[col], errors='ignore')
                df_opt = df_opt.round(2)

                st.session_state['opt_results'] = df_opt 
                st.session_state['sort_metric'] = sort_metric
            else:
                st.warning("조건을 만족하는 결과가 없습니다.")

    if 'opt_results' in st.session_state:
        df_show = st.session_state['opt_results'].sort_values(st.session_state['sort_metric'], ascending=False).head(top_n)
        
        st.markdown("#### 🏆 상위 결과 (적용 버튼을 누르면 즉시 백테스트 실행)")
        
        for i, row in df_show.iterrows():
            c1, c2 = st.columns([4, 1])
            with c1:
                st.dataframe(
                    pd.DataFrame([row]), 
                    hide_index=True,
                    column_config={
                        "Full_수익률(%)": st.column_config.NumberColumn(format="%.2f%%"),
                        "Test_수익률(%)": st.column_config.NumberColumn(format="%.2f%%"),
                        "Train_수익률(%)": st.column_config.NumberColumn(format="%.2f%%"),
                        "Full_MDD(%)": st.column_config.NumberColumn(format="%.2f%%"),
                        "Full_승률(%)": st.column_config.NumberColumn(format="%.2f%%"),
                    },
                    use_container_width=True
                )
            with c2:
                if st.button(f"🥇 적용하기 #{i}", key=f"apply_{i}", on_click=apply_opt_params, args=(row,)):
                    st.rerun()
