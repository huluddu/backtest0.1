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
st.set_page_config(page_title="시그널 대시보드 Ultimate (Final v2)", page_icon="📈", layout="wide")

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
    defaults = {
        "signal_ticker_input": "SOXL", "trade_ticker_input": "SOXL",
        "buy_operator": ">", "sell_operator": "<",
        "strategy_behavior": "1. 포지션 없으면 매수 / 보유 중이면 매도",
        "offset_cl_buy": 0, "offset_cl_sell": 0,
        "offset_ma_buy": 0, "offset_ma_sell": 0,
        "ma_buy": 50, "ma_sell": 10,
        "use_trend_in_buy": True, "use_trend_in_sell": False,
        "ma_compare_short": 20, "ma_compare_long": 50,
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
            "auto_run_trigger": True
        }
        for k, v in updates.items(): st.session_state[k] = v
        st.session_state["preset_name_selector"] = "직접 설정"
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
            df = stock.get_etf_ohlcv_by_date(s, e, code)
            if df is None or df.empty: df = stock.get_market_ohlcv_by_date(s, e, code)
            if not df.empty:
                df = df.reset_index().rename(columns={"날짜":"Date","시가":"Open","고가":"High","저가":"Low","종가":"Close"})
        else:
            df = yf.download(t, start=start_date, end=end_date, progress=False, auto_adjust=False)
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
def prepare_base(signal_ticker, trade_ticker, start_date, end_date, ma_pool):
    sig = get_data(signal_ticker, start_date, end_date).sort_values("Date")
    trd = get_data(trade_ticker,  start_date, end_date).sort_values("Date")
    if sig.empty or trd.empty: return None, None, None, None
    sig = sig.rename(columns={"Close": "Close_sig"})[["Date", "Close_sig"]]
    trd = trd.rename(columns={"Open": "Open_trd", "High": "High_trd", "Low": "Low_trd", "Close": "Close_trd"})
    base = pd.merge(sig, trd, on="Date", how="inner").dropna().reset_index(drop=True)
    x_sig = base["Close_sig"].to_numpy(dtype=float)
    x_trd = base["Close_trd"].to_numpy(dtype=float)
    ma_dict_sig = {}
    for w in sorted(set([int(w) for w in ma_pool if w and w > 0])):
        ma_dict_sig[w] = _fast_ma(x_sig, w)
    return base, x_sig, x_trd, ma_dict_sig

# ==========================================
# 3. 로직 함수 (보조지표 포함)
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
        당신은 월스트리트의 전문 퀀트 트레이더입니다. 아래 백테스트 결과를 한국어로 냉철하게 분석해주세요.
        
        [대상 자산]: {ticker}
        [전략 파라미터]: {params}
        
        [백테스트 성과]
        - 수익률: {summary.get('수익률 (%)')}%
        - MDD (최대 낙폭): {summary.get('MDD (%)')}%
        - 승률: {summary.get('승률 (%)')}%
        - 총 매매 횟수: {summary.get('총 매매 횟수')}회
        - Profit Factor: {summary.get('Profit Factor')}
        
        [분석 요청 사항]
        1. 🛡️ **리스크 평가**: 이 전략이 폭락장에서도 버틸 수 있는지, MDD가 적절한지 평가하세요.
        2. 💰 **수익성 평가**: 단순 보유(Buy&Hold) 대비 매매 비용을 고려했을 때 유효한지 평가하세요.
        3. 💡 **개선 아이디어**: 파라미터(이평선, 손절 등)를 어떻게 수정하면 더 나을지 구체적으로 제안하세요.
        4. ⚖️ **종합 의견**: 실전 투자에 바로 사용해도 될까요? (강력 추천 / 보류 / 비추천)
        """
        with st.spinner("🤖 Gemini가 전략을 분석 중입니다..."):
            response = model.generate_content(prompt)
            return response.text
    except Exception as e: return f"❌ Gemini 분석 오류: {e}"

def check_signal_today(df, ma_buy, offset_ma_buy, ma_sell, offset_ma_sell, offset_cl_buy, offset_cl_sell, ma_compare_short, ma_compare_long, offset_compare_short, offset_compare_long, buy_operator, sell_operator, use_trend_in_buy, use_trend_in_sell):
    if df.empty: st.warning("데이터 없음"); return
    
    # [에러 방지] 정수 변환
    ma_buy, ma_sell = int(ma_buy), int(ma_sell)
    offset_ma_buy, offset_ma_sell = int(offset_ma_buy), int(offset_ma_sell)
    offset_cl_buy, offset_cl_sell = int(offset_cl_buy), int(offset_cl_sell)
    ma_compare_short = int(ma_compare_short) if ma_compare_short else 0
    ma_compare_long = int(ma_compare_long) if ma_compare_long else 0

    df = df.copy().sort_values("Date").reset_index(drop=True)
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df["MA_BUY"], df["MA_SELL"] = df["Close"].rolling(ma_buy).mean(), df["Close"].rolling(ma_sell).mean()
    if ma_compare_short and ma_compare_long:
        df["MA_SHORT"], df["MA_LONG"] = df["Close"].rolling(ma_compare_short).mean(), df["Close"].rolling(ma_compare_long).mean()
    
    i = len(df) - 1
    try:
        cl_b, ma_b = float(df["Close"].iloc[i - offset_cl_buy]), float(df["MA_BUY"].iloc[i - offset_ma_buy])
        cl_s, ma_s = float(df["Close"].iloc[i - offset_cl_sell]), float(df["MA_SELL"].iloc[i - offset_ma_sell])
        ref_date = df["Date"].iloc[-1].strftime('%Y-%m-%d')
        
        trend_ok, trend_msg = True, "비활성화"
        if (use_trend_in_buy or use_trend_in_sell) and "MA_SHORT" in df.columns:
            ms, ml = float(df["MA_SHORT"].iloc[i - offset_compare_short]), float(df["MA_LONG"].iloc[i - offset_compare_long])
            trend_ok = ms >= ml
            trend_msg = f"{ms:.2f} vs {ml:.2f} ({'매수추세' if trend_ok else '매도추세'})"

        buy_base = (cl_b > ma_b) if (buy_operator == ">") else (cl_b < ma_b)
        sell_base = (cl_s < ma_s) if (sell_operator == "<") else (cl_s > ma_s)
        buy_ok = (buy_base and trend_ok) if use_trend_in_buy else buy_base
        sell_ok = (sell_base and (not trend_ok)) if use_trend_in_sell else sell_base
        
        st.subheader(f"📌 오늘 시그널 ({ref_date})")
        st.write(f"📈 추세: {trend_msg}")
        st.write(f"💡 매수: {cl_b:.2f} {buy_operator} {ma_b:.2f} {'+추세' if use_trend_in_buy else ''} → {'✅' if buy_ok else '❌'}")
        st.write(f"💡 매도: {cl_s:.2f} {sell_operator} {ma_s:.2f} {'+역추세' if use_trend_in_sell else ''} → {'✅' if sell_ok else '❌'}")
        
        if buy_ok: st.success("📈 매수 시그널!")
        elif sell_ok: st.error("📉 매도 시그널!")
        else: st.info("⏸ 관망")
    except: st.error("데이터 부족")

def summarize_signal_today(df, p):
    if df is None or df.empty: return {"label": "N/A", "last_buy": "-", "last_sell": "-", "last_hold": "-"}
    
    ma_buy, ma_sell = int(p.get("ma_buy", 50)), int(p.get("ma_sell", 10))
    offset_ma_buy, offset_ma_sell = int(p.get("offset_ma_buy", 50)), int(p.get("offset_ma_sell", 50))
    offset_cl_buy, offset_cl_sell = int(p.get("offset_cl_buy", 1)), int(p.get("offset_cl_sell", 50))
    buy_op, sell_op = p.get("buy_operator", ">"), p.get("sell_operator", "<")
    use_trend_buy, use_trend_sell = bool(p.get("use_trend_in_buy", True)), bool(p.get("use_trend_in_sell", False))
    ma_s, ma_l = int(p.get("ma_compare_short", 20)), int(p.get("ma_compare_long", 50))
    off_s, off_l = int(p.get("offset_compare_short", 0)), int(p.get("offset_compare_long", 0))

    df = df.copy().sort_values("Date").reset_index(drop=True)
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df["MA_BUY"], df["MA_SELL"] = df["Close"].rolling(ma_buy).mean(), df["Close"].rolling(ma_sell).mean()
    if ma_s and ma_l: df["MA_S"], df["MA_L"] = df["Close"].rolling(ma_s).mean(), df["Close"].rolling(ma_l).mean()

    safe_start = max(offset_cl_buy, offset_ma_buy, offset_cl_sell, offset_ma_sell, off_s, off_l) + 1
    last_buy, last_sell, last_hold = None, None, None
    
    for j in range(len(df)-1, safe_start, -1):
        try:
            cb, mb = df["Close"].iloc[j-offset_cl_buy], df["MA_BUY"].iloc[j-offset_ma_buy]
            cs, ms = df["Close"].iloc[j-offset_cl_sell], df["MA_SELL"].iloc[j-offset_ma_sell]
            t_ok = True
            if ma_s and ma_l and "MA_S" in df.columns:
                t_ok = df["MA_S"].iloc[j-off_s] >= df["MA_L"].iloc[j-off_l]
            b_cond = (cb > mb) if buy_op == ">" else (cb < mb)
            s_cond = (cs < ms) if sell_op == "<" else (cs > ms)
            is_buy = (b_cond and t_ok) if use_trend_buy else b_cond
            is_sell = (s_cond and (not t_ok)) if use_trend_sell else s_cond
            d_str = df["Date"].iloc[j].strftime("%Y-%m-%d")
            if last_buy is None and is_buy: last_buy = d_str
            if last_sell is None and is_sell: last_sell = d_str
            if last_hold is None and (not is_buy and not is_sell): last_hold = d_str
            if last_buy and last_sell and last_hold: break
        except: continue

    label = "HOLD"
    try:
        i = len(df)-1
        cb, mb = df["Close"].iloc[i-offset_cl_buy], df["MA_BUY"].iloc[i-offset_ma_buy]
        cs, ms = df["Close"].iloc[i-offset_cl_sell], df["MA_SELL"].iloc[i-offset_ma_sell]
        t_ok = True
        if ma_s and ma_l and "MA_S" in df.columns: t_ok = df["MA_S"].iloc[i-off_s] >= df["MA_L"].iloc[i-off_l]
        b_cond = (cb > mb) if buy_op == ">" else (cb < mb)
        s_cond = (cs < ms) if sell_op == "<" else (cs > ms)
        is_buy = (b_cond and t_ok) if use_trend_buy else b_cond
        is_sell = (s_cond and (not t_ok)) if use_trend_sell else s_cond
        if is_buy and is_sell: label = "BUY & SELL"
        elif is_buy: label = "BUY"
        elif is_sell: label = "SELL"
    except: pass
    return {"label": label, "last_buy": last_buy, "last_sell": last_sell, "last_hold": last_hold}

def backtest_fast(base, x_sig, x_trd, ma_dict_sig, ma_buy, offset_ma_buy, ma_sell, offset_ma_sell, offset_cl_buy, offset_cl_sell, ma_compare_short, ma_compare_long, offset_compare_short, offset_compare_long, initial_cash, stop_loss_pct, take_profit_pct, strategy_behavior, min_hold_days, fee_bps, slip_bps, use_trend_in_buy, use_trend_in_sell, buy_operator, sell_operator, 
                  use_rsi_filter=False, rsi_period=14, rsi_min=30, rsi_max=70):
    n = len(base)
    if n == 0: return {}
    
    # [중요] 정수형 키로 변환하여 조회
    ma_buy, ma_sell = int(ma_buy), int(ma_sell)
    ma_buy_arr, ma_sell_arr = ma_dict_sig.get(ma_buy), ma_dict_sig.get(ma_sell)
    
    ma_s_arr = ma_dict_sig.get(int(ma_compare_short)) if ma_compare_short else None
    ma_l_arr = ma_dict_sig.get(int(ma_compare_long)) if ma_compare_long else None

    # 배열이 없으면 에러 방지를 위해 원본 사용 (혹은 빈 배열)
    if ma_buy_arr is None: ma_buy_arr = x_sig
    if ma_sell_arr is None: ma_sell_arr = x_sig

    rsi_arr = None
    if use_rsi_filter:
        rsi_arr = calculate_indicators(x_sig, int(rsi_period))
    
    max_offset = max(ma_buy, ma_sell, offset_ma_buy, offset_ma_sell, offset_cl_buy, offset_cl_sell, (offset_compare_short or 0), (offset_compare_long or 0), (rsi_period if use_rsi_filter else 0))
    idx0 = int(max_offset) + 1

    xO, xH, xL, xC_trd = base["Open_trd"].values, base["High_trd"].values, base["Low_trd"].values, x_trd
    cash, position, hold_days = float(initial_cash), 0.0, 0
    entry_price = 0.0 
    logs, asset_curve = [], []
    sb = str(strategy_behavior)[:1]

    def _fill_buy(px): return px * (1 + (slip_bps + fee_bps)/10000.0)
    def _fill_sell(px): return px * (1 - (slip_bps + fee_bps)/10000.0)

    for i in range(idx0, n):
        just_bought = False
        exec_price, signal, reason = None, "HOLD", None
        open_today, high_today, low_today, close_today = xO[i], xH[i], xL[i], xC_trd[i]

        try:
            cl_b, ma_b = float(x_sig[i - offset_cl_buy]), float(ma_buy_arr[i - offset_ma_buy])
            cl_s, ma_s = float(x_sig[i - offset_cl_sell]), float(ma_sell_arr[i - offset_ma_sell])
        except: 
            asset_curve.append(cash + position * close_today)
            continue

        trend_ok = True
        if ma_s_arr is not None and ma_l_arr is not None:
            try:
                ms, ml = ma_s_arr[i - offset_compare_short], ma_l_arr[i - offset_compare_long]
                trend_ok = (ms >= ml)
            except: pass

        buy_base = (cl_b > ma_b) if buy_operator == ">" else (cl_b < ma_b)
        sell_base = (cl_s < ma_s) if (sell_operator == "<") else (cl_s > ma_s)
        buy_cond = (buy_base and trend_ok) if use_trend_in_buy else buy_base
        sell_cond = (sell_base and (not trend_ok)) if use_trend_in_sell else sell_base

        if use_rsi_filter and buy_cond and rsi_arr is not None:
            if rsi_arr[i-1] > rsi_max: buy_cond = False

        stop_hit, take_hit = False, False
        if position > 0 and entry_price > 0:
            if stop_loss_pct > 0:
                sl_price = entry_price * (1 - stop_loss_pct / 100)
                if low_today <= sl_price:
                    stop_hit = True
                    exec_price = open_today if open_today < sl_price else sl_price
            
            if take_profit_pct > 0 and not stop_hit:
                tp_price = entry_price * (1 + take_profit_pct / 100)
                if high_today >= tp_price:
                    take_hit = True
                    exec_price = open_today if open_today > tp_price else tp_price
            
            if stop_hit or take_hit:
                fill = _fill_sell(exec_price)
                cash = position * fill
                position = 0.0
                entry_price = 0.0
                signal = "SELL"; reason = "손절" if stop_hit else "익절"

        if position > 0 and signal == "HOLD":
            if sell_cond and hold_days >= int(min_hold_days):
                base_px = open_today
                fill = _fill_sell(base_px)
                cash = position * fill
                position = 0.0
                entry_price = 0.0
                signal = "SELL"; reason = "전략매도"; exec_price = base_px

        if position == 0 and signal == "HOLD":
            do_buy = False
            if sb == "1": do_buy = buy_cond
            elif sb == "2": do_buy = buy_cond and not sell_cond
            elif sb == "3": do_buy = buy_cond and not sell_cond
            
            if do_buy:
                base_px = open_today
                fill = _fill_buy(base_px)
                position = cash / fill
                entry_price = base_px
                cash = 0.0
                signal = "BUY"; reason = "전략매수"; exec_price = base_px
                just_bought = True

        if position > 0 and not just_bought: hold_days += 1
        else: hold_days = 0

        total = cash + (position * close_today)
        asset_curve.append(total)
        logs.append({
            "날짜": base["Date"].iloc[i], "종가": close_today, "신호": signal, "체결가": exec_price,
            "자산": total, "이유": reason, "손절발동": stop_hit, "익절발동": take_hit, 
            "RSI": rsi_arr[i] if (use_rsi_filter and rsi_arr is not None and i < len(rsi_arr)) else None
        })

    if not logs: return {}
    final_asset = asset_curve[-1]
    s = pd.Series(asset_curve)
    mdd = ((s - s.cummax()) / s.cummax()).min() * 100
    
    buy_cache = None
    g_profit, g_loss, wins = 0, 0, 0
    df_res = pd.DataFrame(logs)
    for r in logs:
        if r['신호'] == 'BUY': buy_cache = r
        elif r['신호'] == 'SELL' and buy_cache:
            pb = buy_cache['체결가'] or buy_cache['종가']
            ps = r['체결가'] or r['종가']
            ret = (ps - pb) / pb
            if ret > 0: wins += 1; g_profit += ret
            else: g_loss += abs(ret)
            buy_cache = None
    
    total_trades = wins + (len(df_res[df_res['신호']=='SELL']) - wins)
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0.0
    pf = (g_profit / g_loss) if g_loss > 0 else 999.0

    return {
        "수익률 (%)": round((final_asset - initial_cash)/initial_cash*100, 2),
        "MDD (%)": round(mdd, 2), "승률 (%)": round(win_rate, 2),
        "Profit Factor": round(pf, 2), "총 매매 횟수": total_trades,
        "최종 자산": round(final_asset), "매매 로그": logs
    }

def auto_search_train_test(signal_ticker, trade_ticker, start_date, end_date, split_ratio, choices_dict, n_trials=50, initial_cash=5000000, fee_bps=0, slip_bps=0, strategy_behavior="1", min_hold_days=0, constraints=None, **kwargs):
    ma_pool = set([5, 10, 20, 60, 120])
    for k in ["ma_buy", "ma_sell", "ma_compare_short", "ma_compare_long"]:
        for v in choices_dict.get(k, []):
            try:
                if int(v) > 0: ma_pool.add(int(v))
            except: pass
            
    base_full, x_sig_full, x_trd_full, ma_dict = prepare_base(signal_ticker, trade_ticker, start_date, end_date, list(ma_pool))
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
        if limit_mdd > 0 and res_full.get('MDD (%)', 0) < -abs(limit_mdd): continue

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
PRESETS.update(load_saved_strategies())
st.session_state["ALL_PRESETS_DATA"] = PRESETS

with st.sidebar:
    st.header("⚙️ 설정 & Gemini")
    api_key_input = st.text_input("Gemini API Key", type="password", key="gemini_key_input")
    if api_key_input: 
        st.session_state["gemini_api_key"] = api_key_input
        try:
            genai.configure(api_key=api_key_input)
            models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
            idx = 0
            for i, m in enumerate(models):
                if "gemini-1.5-flash" in m: idx = i; break
            selected_model = st.selectbox("🤖 모델 선택", models, index=idx)
            st.session_state["selected_model_name"] = selected_model
        except: st.error("모델 로드 실패")
    
    st.divider()
    with st.expander("💾 전략 저장/삭제"):
        save_name = st.text_input("전략 이름")
        if st.button("현재 설정 저장"):
            if save_name:
                params = {k: st.session_state[k] for k in ["signal_ticker_input","trade_ticker_input","ma_buy","offset_ma_buy","offset_cl_buy","buy_operator","ma_sell","offset_ma_sell","offset_cl_sell","sell_operator","use_trend_in_buy","use_trend_in_sell","ma_compare_short","ma_compare_long","offset_compare_short","offset_compare_long","stop_loss_pct","take_profit_pct","min_hold_days"]}
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

col1, col2 = st.columns(2)
signal_ticker = col1.text_input("시그널 티커", key="signal_ticker_input")
trade_ticker = col2.text_input("매매 티커", key="trade_ticker_input")
col3, col4 = st.columns(2)
start_date = col3.date_input("시작일", value=datetime.date(2020, 1, 1))
end_date = col4.date_input("종료일", value=datetime.date.today())

with st.expander("📈 상세 설정 (Offset, 비용 등)", expanded=True):
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
        offset_compare_long = st.number_input("
