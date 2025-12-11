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
st.set_page_config(page_title="시그널 대시보드 Ultimate", page_icon="📈", layout="wide")
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
                try: df = df.xs(t, axis=1, level=1) if t in df.columns.levels[1] else df.droplevel(1, axis=1)
                except: df = df.droplevel(1, axis=1)
            df = df.reset_index()
            if "Datetime" in df.columns: df.rename(columns={"Datetime": "Date"}, inplace=True)
            if "Date" in df.columns and pd.api.types.is_datetime64_any_dtype(df["Date"]):
                df["Date"] = df["Date"].dt.tz_localize(None)

        if df is None or df.empty: return pd.DataFrame(columns=["Date", "Open", "High", "Low", "Close"])
        cols = ["Open", "High", "Low", "Close"]
        df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')
        return df[["Date", "Open", "High", "Low", "Close"]].dropna()
    except: return pd.DataFrame(columns=["Date", "Open", "High", "Low", "Close"])

@st.cache_data(show_spinner=False, ttl=30)
def get_yf_1m_grouped_close(ticker: str, tz: str, session_start: str, session_end: str):
    try:
        df = yf.download(ticker, period="5d", interval="1m", auto_adjust=False, progress=False)
        if df.empty: return pd.Series(dtype=float), None, None
        if isinstance(df.columns, pd.MultiIndex): df = df.droplevel(1, axis=1)
        if df.index.tz is None: df.index = df.index.tz_localize("UTC")
        df = df.tz_convert(tz).between_time(session_start, session_end).copy()
        if df.empty: return pd.Series(dtype=float), None, None
        df["session"] = df.index.date
        return df.groupby("session")["Close"].last(), float(df.iloc[-1]["Close"]), df.iloc[-1].name
    except: return pd.Series(dtype=float), None, None

@st.cache_data(show_spinner=False, ttl=1800)
def prepare_base(signal_ticker, trade_ticker, start_date, end_date, ma_pool):
    sig = get_data(signal_ticker, start_date, end_date).sort_values("Date")
    trd = get_data(trade_ticker,  start_date, end_date).sort_values("Date")
    if sig.empty or trd.empty: return None, None, None, None
    sig = sig.rename(columns={"Close": "Close_sig"})[["Date", "Close_sig"]]
    trd = trd.rename(columns={"Open": "Open_trd", "High": "High_trd", "Low": "Low_trd", "Close": "Close_trd"})
    base = pd.merge(sig, trd, on="Date", how="inner").dropna().reset_index(drop=True)
    x_sig = base["Close_sig"].to_numpy(dtype=float)
    x_trd = base["Close_trd"].to_numpy(dtype=float)
    ma_dict_sig = {}
    for w in sorted(set([w for w in ma_pool if w and w > 0])):
        ma_dict_sig[w] = _fast_ma(x_sig, w)
    return base, x_sig, x_trd, ma_dict_sig

# ==========================================
# 3. 로직 함수 (보조지표 포함)
# ==========================================
def calculate_indicators(close_data, rsi_period, bb_period, bb_std):
    df = pd.DataFrame({'close': close_data})
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    mid = df['close'].rolling(window=bb_period).mean()
    std = df['close'].rolling(window=bb_period).std()
    upper = mid + (bb_std * std)
    lower = mid - (bb_std * std)
    return rsi.to_numpy(), upper.to_numpy(), lower.to_numpy()

def ask_gemini_analysis(summary, params, ticker, api_key, model_name):
    if not api_key: return "⚠️ API Key가 없습니다."
    try:
        genai.configure(api_key=api_key)
        m_name = model_name if model_name and model_name.strip() else "gemini-pro"
        model = genai.GenerativeModel(m_name)
        prompt = f"""
        전문 퀀트 투자자 관점에서 분석해주세요.
        [전략: {ticker}] {params}
        [결과] 수익률: {summary.get('수익률 (%)')}%, MDD: {summary.get('MDD (%)')}%, 승률: {summary.get('승률 (%)')}%
        1. 리스크 분석
        2. 실전 투자 적합성
        3. 파라미터 개선 제안
        """
        with st.spinner("🤖 분석 중..."): return model.generate_content(prompt).text
    except Exception as e: return f"❌ 오류: {e}"

def check_signal_today(df, ma_buy, offset_ma_buy, ma_sell, offset_ma_sell, offset_cl_buy, offset_cl_sell, ma_compare_short, ma_compare_long, offset_compare_short, offset_compare_long, buy_operator, sell_operator, use_trend_in_buy, use_trend_in_sell):
    if df.empty: st.warning("데이터 없음"); return
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
                  use_rsi_filter=False, rsi_period=14, rsi_min=30, rsi_max=70,
                  use_bb_filter=False, bb_period=20, bb_std=2.0):
    n = len(base)
    if n == 0: return {}
    ma_buy_arr, ma_sell_arr = ma_dict_sig.get(ma_buy), ma_dict_sig.get(ma_sell)
    ma_s_arr = ma_dict_sig.get(ma_compare_short) if ma_compare_short else None
    ma_l_arr = ma_dict_sig.get(ma_compare_long) if ma_compare_long else None

    rsi_arr, bb_up, bb_lo = None, None, None
    if use_rsi_filter or use_bb_filter:
        rsi_arr, bb_up, bb_lo = calculate_indicators(x_sig, rsi_period, bb_period, bb_std)
    
    idx0 = max((ma_buy or 1), (ma_sell or 1), offset_ma_buy, offset_ma_sell, offset_cl_buy, offset_cl_sell, (offset_compare_short or 0), (offset_compare_long or 0), (rsi_period if use_rsi_filter else 0), (bb_period if use_bb_filter else 0)) + 1
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
            ms, ml = ma_s_arr[i - offset_compare_short], ma_l_arr[i - offset_compare_long]
            trend_ok = (ms >= ml)

        buy_base = (cl_b > ma_b) if buy_operator == ">" else (cl_b < ma_b)
        sell_base = (cl_s < ma_s) if (sell_operator == "<") else (cl_s > ma_s)
        buy_cond = (buy_base and trend_ok) if use_trend_in_buy else buy_base
        sell_cond = (sell_base and (not trend_ok)) if use_trend_in_sell else sell_base

        if use_rsi_filter and buy_cond:
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
            "RSI": rsi_arr[i] if use_rsi_filter and i < len(rsi_arr) else None
        })

    if not logs: return {}
    final_asset = asset_curve[-1]
    s = pd.Series(asset_curve)
    mdd = ((s - s.cummax()) / s.cummax()).min() * 100
    
    trade_pairs, buy_cache = [], None
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
            if isinstance(v, int) and v > 0: ma_pool.add(v)
            
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
    # ✅ 신규 필터 추가
    min_train_r = constraints.get("min_train_ret", -999.0)
    min_test_r = constraints.get("min_test_ret", -999.0)

    for _ in range(int(n_trials)):
        p = {}
        for k in choices_dict.keys():
            arr = choices_dict[k]
            p[k] = random.choice(arr) if arr else defaults.get(k)
        
        common_args = {
            "ma_dict_sig": ma_dict,
            "ma_buy": p.get('ma_buy', 50), "offset_ma_buy": p.get('offset_ma_buy', 0),
            "ma_sell": p.get('ma_sell', 10), "offset_ma_sell": p.get('offset_ma_sell', 0),
            "offset_cl_buy": p.get('offset_cl_buy', 0), "offset_cl_sell": p.get('offset_cl_sell', 0),
            "ma_compare_short": p.get('ma_compare_short'), "ma_compare_long": p.get('ma_compare_long'),
            "offset_compare_short": p.get('offset_compare_short', 0), "offset_compare_long": p.get('offset_compare_long', 0),
            "initial_cash": initial_cash, "stop_loss_pct": p.get('stop_loss_pct', 0), "take_profit_pct": p.get('take_profit_pct', 0),
            "strategy_behavior": strategy_behavior, "min_hold_days": min_hold_days, "fee_bps": fee_bps, "slip_bps": slip_bps,
            "use_trend_in_buy": p.get('use_trend_in_buy', True), "use_trend_in_sell": p.get('use_trend_in_sell', False),
            "buy_operator": p.get('buy_operator', '>'), "sell_operator": p.get('sell_operator', '<')
        }

        # Full Test
        res_full = backtest_fast(base_full, x_sig_full, x_trd_full, **common_args)
        if not res_full: continue
        
        # 필터링
        if res_full.get('총 매매 횟수', 0) < min_tr: continue
        if res_full.get('승률 (%)', 0) < min_wr: continue
        if limit_mdd > 0 and res_full.get('MDD (%)', 0) < -abs(limit_mdd): continue

        res_tr = backtest_fast(base_tr, x_sig_tr, x_trd_tr, **common_args)
        if res_tr.get('수익률 (%)', -999) < min_train_r: continue # Train 수익률 필터

        res_te = backtest_fast(base_te, x_sig_te, x_trd_te, **common_args)
        if res_te.get('수익률 (%)', -999) < min_test_r: continue # Test 수익률 필터

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
    "TSLL 안전 전략": {"signal_ticker": "TSLL", "trade_ticker": "TSLL", "offset_cl_buy": 20, "buy_operator": "<", "offset_ma_buy": 50, "ma_buy": 20, "offset_cl_sell": 50, "sell_operator": ">", "offset_ma_sell": 20, "ma_sell": 10, "use_trend_in_buy": True, "use_trend_in_sell": True, "offset_compare_short": 1, "ma_compare_short": 10, "offset_compare_long": 50, "ma_compare_long": 50, "stop_loss_pct": 25.0, "take_profit_pct": 15.0},
    "TSLL 도전 전략": {"signal_ticker": "TSLL", "trade_ticker": "TSLL", "offset_cl_buy": 50, "buy_operator": ">", "offset_ma_buy": 50, "ma_buy": 10, "offset_cl_sell": 10, "sell_operator": "<", "offset_ma_sell": 10, "ma_sell": 20, "use_trend_in_buy": True, "use_trend_in_sell": True, "offset_compare_short": 50, "ma_compare_short": 50, "offset_compare_long": 50, "ma_compare_long": 5, "stop_loss_pct": 0.0, "take_profit_pct": 0.0},
    "GGLL 전략": {"signal_ticker": "GGLL", "trade_ticker": "GGLL", "offset_cl_buy": 1, "buy_operator": "<", "offset_ma_buy": 1, "ma_buy": 20, "offset_cl_sell": 20, "sell_operator": "<", "offset_ma_sell": 20, "ma_sell": 50, "use_trend_in_buy": True, "use_trend_in_sell": True, "offset_compare_short": 20, "ma_compare_short": 1, "offset_compare_long": 50, "ma_compare_long": 1, "stop_loss_pct": 15.0, "take_profit_pct": 0.0},
    "GGLL 안전 전략": {"signal_ticker": "GGLL", "trade_ticker": "GGLL", "offset_cl_buy": 10, "buy_operator": ">", "offset_ma_buy": 50, "ma_buy": 5, "offset_cl_sell": 10, "sell_operator": "<", "offset_ma_sell": 20, "ma_sell": 20, "use_trend_in_buy": True, "use_trend_in_sell": True, "offset_compare_short": 10, "ma_compare_short": 20, "offset_compare_long": 50, "ma_compare_long": 10, "stop_loss_pct": 20.0, "take_profit_pct": 20.0},
    "GGLL 안전 전략2": {"signal_ticker": "GGLL", "trade_ticker": "GGLL", "offset_cl_buy": 50, "buy_operator": "<", "offset_ma_buy": 20, "ma_buy": 1, "offset_cl_sell": 20, "sell_operator": "<", "offset_ma_sell": 1, "ma_sell": 1, "use_trend_in_buy": True, "use_trend_in_sell": True, "offset_compare_short": 1, "ma_compare_short": 10, "offset_compare_long": 50, "ma_compare_long": 10, "stop_loss_pct": 20.0, "take_profit_pct": 30.0},
    "BITX 전략": {"signal_ticker": "BITX", "trade_ticker": "BITX", "offset_cl_buy": 16, "buy_operator": ">", "offset_ma_buy": 26, "ma_buy": 5, "offset_cl_sell": 26, "sell_operator": ">", "offset_ma_sell": 2, "ma_sell": 15, "use_trend_in_buy": True, "use_trend_in_sell": True, "offset_compare_short": 26, "ma_compare_short": 15, "offset_compare_long": 6, "ma_compare_long": 15, "stop_loss_pct": 30.0, "take_profit_pct": 0.0},
    "TQQQ 도전 전략": {"signal_ticker": "TQQQ", "trade_ticker": "TQQQ", "offset_cl_buy": 50, "buy_operator": ">", "offset_ma_buy": 10, "ma_buy": 1, "offset_cl_sell": 50, "sell_operator": ">", "offset_ma_sell": 1, "ma_sell": 1, "use_trend_in_buy": True, "use_trend_in_sell": True, "offset_compare_short": 1, "ma_compare_short": 50, "offset_compare_long": 10, "ma_compare_long": 1, "stop_loss_pct": 15.0, "take_profit_pct": 25.0},
    "TQQQ 안전 전략": {"signal_ticker": "TQQQ", "trade_ticker": "TQQQ", "offset_cl_buy": 10, "buy_operator": "<", "offset_ma_buy": 50, "ma_buy": 20, "offset_cl_sell": 50, "sell_operator": ">", "offset_ma_sell": 10, "ma_sell": 20, "use_trend_in_buy": True, "use_trend_in_sell": True, "offset_compare_short": 10, "ma_compare_short": 50, "offset_compare_long": 20, "ma_compare_long": 20, "stop_loss_pct": 25.0, "take_profit_pct": 25.0},
    "BITX-TQQQ 안전": {"signal_ticker": "BITX", "trade_ticker": "TQQQ", "offset_cl_buy": 10, "buy_operator": ">", "offset_ma_buy": 10, "ma_buy": 20, "offset_cl_sell": 50, "sell_operator": ">", "offset_ma_sell": 1, "ma_sell": 5, "use_trend_in_buy": False, "use_trend_in_sell": True, "offset_compare_short": 50, "ma_compare_short": 5, "offset_compare_long": 1, "ma_compare_long": 50, "stop_loss_pct": 0.0, "take_profit_pct": 15.0},
    "BITX-TQQQ 도전": {"signal_ticker": "BITX", "trade_ticker": "TQQQ", "offset_cl_buy": 50, "buy_operator": "<", "offset_ma_buy": 20, "ma_buy": 50, "offset_cl_sell": 50, "sell_operator": ">", "offset_ma_sell": 10, "ma_sell": 5, "use_trend_in_buy": True, "use_trend_in_sell": True, "offset_compare_short": 10, "ma_compare_short": 1, "offset_compare_long": 20, "ma_compare_long": 1, "stop_loss_pct": 25.0, "take_profit_pct": 25.0},
    "BTC-TQQQ": {"signal_ticker": "BTC-USD", "trade_ticker": "TQQQ", "offset_cl_buy": 10, "buy_operator": ">", "offset_ma_buy": 1, "ma_buy": 20, "offset_cl_sell": 50, "sell_operator": "<", "offset_ma_sell": 1, "ma_sell": 10, "use_trend_in_buy": True, "use_trend_in_sell": True, "offset_compare_short": 1, "ma_compare_short": 5, "offset_compare_long": 10, "ma_compare_long": 5, "stop_loss_pct": 15.0, "take_profit_pct": 15.0},
    "BTC-TQQQ2": {"signal_ticker": "BTC-USD", "trade_ticker": "TQQQ", "offset_cl_buy": 50, "buy_operator": ">", "offset_ma_buy": 20, "ma_buy": 1, "offset_cl_sell": 50, "sell_operator": "<", "offset_ma_sell": 20, "ma_sell": 1, "use_trend_in_buy": True, "use_trend_in_sell": True, "offset_compare_short": 1, "ma_compare_short": 10, "offset_compare_long": 20, "ma_compare_long": 10, "stop_loss_pct": 35.0, "take_profit_pct": 25.0},
    "453850 ACE 미국30년국채 전략": {"signal_ticker": "453850", "trade_ticker": "453850", "offset_cl_buy": 16, "buy_operator": "<", "offset_ma_buy": 26, "ma_buy": 15, "offset_cl_sell": 26, "sell_operator": ">", "offset_ma_sell": 2, "ma_sell": 20, "use_trend_in_buy": True, "use_trend_in_sell": False, "offset_compare_short": 2, "ma_compare_short": 15, "offset_compare_long": 26, "ma_compare_long": 15, "stop_loss_pct": 0.0, "take_profit_pct": 10.0},
    "465580 ACE미국빅테크TOP7PLUS": {"signal_ticker": "465580", "trade_ticker": "465580", "offset_cl_buy": 2, "buy_operator": ">", "offset_ma_buy": 2, "ma_buy": 5, "offset_cl_sell": 2, "sell_operator": "<", "offset_ma_sell": 2, "ma_sell": 25, "use_trend_in_buy": False, "use_trend_in_sell": True, "offset_compare_short": 6, "ma_compare_short": 10, "offset_compare_long": 2, "ma_compare_long": 10, "stop_loss_pct": 0.0, "take_profit_pct": 10.0},
    "390390 KODEX미국반도체": {"signal_ticker": "390390", "trade_ticker": "390390", "offset_cl_buy": 6, "buy_operator": "<", "offset_ma_buy": 2, "ma_buy": 5, "offset_cl_sell": 26, "sell_operator": ">", "offset_ma_sell": 2, "ma_sell": 20, "use_trend_in_buy": False, "use_trend_in_sell": True, "offset_compare_short": 6, "ma_compare_short": 25, "offset_compare_long": 2, "ma_compare_long": 25, "stop_loss_pct": 0.0, "take_profit_pct": 10.0},
    "371460 TIGER차이나전기차SOLACTIVE": {"signal_ticker": "371460", "trade_ticker": "371460", "offset_cl_buy": 2, "buy_operator": ">", "offset_ma_buy": 6, "ma_buy": 10, "offset_cl_sell": 16, "sell_operator": ">", "offset_ma_sell": 2, "ma_sell": 5, "use_trend_in_buy": True, "use_trend_in_sell": True, "offset_compare_short": 6, "ma_compare_short": 15, "offset_compare_long": 16, "ma_compare_long": 10, "stop_loss_pct": 0.0, "take_profit_pct": 10.0},
    "483280 AITOP10커브드콜": {"signal_ticker": "483280", "trade_ticker": "483280", "offset_cl_buy": 26, "buy_operator": ">", "offset_ma_buy": 26, "ma_buy": 20, "offset_cl_sell": 26, "sell_operator": ">", "offset_ma_sell": 6, "ma_sell": 20, "use_trend_in_buy": True, "use_trend_in_sell": True, "offset_compare_short": 2, "ma_compare_short": 20, "offset_compare_long": 16, "ma_compare_long": 5, "stop_loss_pct": 0.0, "take_profit_pct": 0.0},
}
PRESETS.update(load_saved_strategies())

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
    selected_preset = st.selectbox("🎯 프리셋", ["직접 설정"] + list(PRESETS.keys()), key="preset_name", on_change=_on_preset_change, args=(PRESETS,))

preset_values = PRESETS.get(selected_preset, {}) if selected_preset != "직접 설정" else {}
col1, col2 = st.columns(2)
signal_ticker = col1.text_input("시그널 티커", value=preset_values.get("signal_ticker", "SOXL"), key="signal_ticker_input")
trade_ticker = col2.text_input("매매 티커", value=preset_values.get("trade_ticker", "SOXL"), key="trade_ticker_input")
col3, col4 = st.columns(2)
start_date = col3.date_input("시작일", value=datetime.date(2020, 1, 1))
end_date = col4.date_input("종료일", value=datetime.date.today())

with st.expander("📈 상세 설정 (Offset, 비용 등)", expanded=True):
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 📥 매수")
        ma_buy = st.number_input("매수 이평", value=preset_values.get("ma_buy", 20), key="ma_buy")
        offset_ma_buy = st.number_input("매수 이평 Offset", value=preset_values.get("offset_ma_buy", 0), key="offset_ma_buy")
        offset_cl_buy = st.number_input("매수 종가 Offset", value=preset_values.get("offset_cl_buy", 0), key="offset_cl_buy")
        buy_operator = st.selectbox("매수 부호", [">", "<"], index=0 if preset_values.get("buy_operator",">")==">" else 1, key="buy_operator")
        use_trend_in_buy = st.checkbox("매수 추세 필터", value=preset_values.get("use_trend_in_buy", True), key="use_trend_in_buy")
    with c2:
        st.markdown("#### 📤 매도")
        ma_sell = st.number_input("매도 이평", value=preset_values.get("ma_sell", 10), key="ma_sell")
        offset_ma_sell = st.number_input("매도 이평 Offset", value=preset_values.get("offset_ma_sell", 0), key="offset_ma_sell")
        offset_cl_sell = st.number_input("매도 종가 Offset", value=preset_values.get("offset_cl_sell", 0), key="offset_cl_sell")
        sell_operator = st.selectbox("매도 부호", ["<", ">"], index=0 if preset_values.get("sell_operator","<")=="<" else 1, key="sell_operator")
        use_trend_in_sell = st.checkbox("매도 역추세 필터", value=preset_values.get("use_trend_in_sell", False), key="use_trend_in_sell")
    
    st.divider()
    c3, c4 = st.columns(2)
    with c3:
        st.markdown("#### 📈 추세선")
        ma_compare_short = st.number_input("추세 Short", value=preset_values.get("ma_compare_short", 20), key="ma_compare_short")
        offset_compare_short = st.number_input("추세 Short Offset", value=preset_values.get("offset_compare_short", 0), key="offset_compare_short")
    with c4:
        st.markdown("#### .")
        ma_compare_long = st.number_input("추세 Long", value=preset_values.get("ma_compare_long", 50), key="ma_compare_long")
        offset_compare_long = st.number_input("추세 Long Offset", value=preset_values.get("offset_compare_long", 0), key="offset_compare_long")

    st.divider()
    c5, c6 = st.columns(2)
    with c5:
        st.markdown("#### 🛡️ 리스크")
        stop_loss_pct = st.number_input("손절 (%)", value=float(preset_values.get("stop_loss_pct", 0.0)), step=0.5, key="stop_loss_pct")
        take_profit_pct = st.number_input("익절 (%)", value=float(preset_values.get("take_profit_pct", 0.0)), step=0.5, key="take_profit_pct")
        min_hold_days = st.number_input("최소 보유일", value=int(preset_values.get("min_hold_days", 0)), step=1, key="min_hold_days")
    with c6:
        st.markdown("#### ⚙️ 기타")
        strategy_behavior = st.selectbox("행동 패턴", ["1. 포지션 없으면 매수 / 보유 중이면 매도", "2. 매수 우선", "3. 관망"], key="strategy_behavior")
        fee_bps = st.number_input("수수료 (bps)", value=25, step=1, key="fee_bps")
        slip_bps = st.number_input("슬리피지 (bps)", value=1, step=1, key="slip_bps")
        seed = st.number_input("랜덤 시드", value=0, step=1)
        if seed > 0: random.seed(seed)

    st.divider()
    # ✅ RSI 설정
    st.markdown("#### 🔮 보조지표 설정")
    c_r1, c_r2 = st.columns(2)
    rsi_p = c_r1.number_input("RSI 기간 (Period)", 14, key="rsi_period")
    u_rsi = st.checkbox("RSI 필터 적용 (매수시 과열 방지)", value=preset_values.get("use_rsi_filter", False), key="use_rsi_filter")
    if u_rsi:
        rsi_max = c_r2.number_input("RSI 과매수 기준", 70, key="rsi_max")

tab1, tab2, tab3, tab4 = st.tabs(["🎯 시그널", "📚 PRESETS", "🧪 백테스트", "🧬 실험실"])

with tab1:
    if st.button("📌 시그널 확인"):
        check_signal_today(get_data(signal_ticker, start_date, end_date), ma_buy, offset_ma_buy, ma_sell, offset_ma_sell, offset_cl_buy, offset_cl_sell, ma_compare_short, ma_compare_long, offset_compare_short, offset_compare_long, buy_operator, sell_operator, use_trend_in_buy, use_trend_in_sell)

with tab2:
    if st.button("📚 일괄 확인"):
        rows = []
        with st.spinner("계산 중..."):
            for name, p in PRESETS.items():
                t = p.get("signal_ticker", p.get("trade_ticker"))
                res = summarize_signal_today(get_data(t, start_date, end_date), p)
                rows.append({
                    "전략": name, "티커": t, "시그널": res["label"], 
                    "최근 BUY": res["last_buy"], "최근 SELL": res["last_sell"], "최근 HOLD": res["last_hold"]
                })
        st.dataframe(pd.DataFrame(rows))

with tab3:
    should_run = False
    if st.button("✅ 백테스트 실행", use_container_width=True): should_run = True
    if st.session_state.get("auto_run_trigger"): should_run = True; st.session_state["auto_run_trigger"] = False 

    if should_run:
        ma_pool = [ma_buy, ma_sell, ma_compare_short, ma_compare_long]
        base, x_sig, x_trd, ma_dict = prepare_base(signal_ticker, trade_ticker, start_date, end_date, ma_pool)
        if base is not None:
            res = backtest_fast(base, x_sig, x_trd, ma_dict, ma_buy, offset_ma_buy, ma_sell, offset_ma_sell, offset_cl_buy, offset_cl_sell, ma_compare_short, ma_compare_long, offset_compare_short, offset_compare_long, 5000000, stop_loss_pct, take_profit_pct, strategy_behavior, min_hold_days, fee_bps, slip_bps, use_trend_in_buy, use_trend_in_sell, buy_operator, sell_operator, 
                                use_rsi_filter=st.session_state.get("use_rsi_filter", False), rsi_period=st.session_state.get("rsi_period", 14), rsi_max=st.session_state.get("rsi_max", 70))
            st.session_state["bt_result"] = res
            if "ai_analysis" in st.session_state: del st.session_state["ai_analysis"]
        else: st.error("데이터 로딩 실패")

    if "bt_result" in st.session_state:
        res = st.session_state["bt_result"]
        if res:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("수익률", f"{res['수익률 (%)']}%")
            c2.metric("MDD", f"{res['MDD (%)']}%")
            c3.metric("승률", f"{res['승률 (%)']}%")
            c4.metric("PF", res['Profit Factor'])
            
            df_log = pd.DataFrame(res['매매 로그'])
            initial_price = df_log['종가'].iloc[0]
            benchmark = (df_log['종가'] / initial_price) * 5000000
            drawdown = (df_log['자산'] - df_log['자산'].cummax()) / df_log['자산'].cummax() * 100

            # 3단 차트
            fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.5, 0.25, 0.25], 
                                subplot_titles=("자산 & Benchmark", "RSI (14)", "MDD (%)"))

            fig.add_trace(go.Scatter(x=df_log['날짜'], y=df_log['자산'], name='내 전략', line=dict(color='#00F0FF', width=2)), row=1, col=1)
            fig.add_trace(go.Scatter(x=df_log['날짜'], y=benchmark, name='Buy & Hold', line=dict(color='gray', dash='dot')), row=1, col=1)
            
            buys = df_log[df_log['신호']=='BUY']
            sells_reg = df_log[(df_log['신호']=='SELL') & (df_log['손절발동']==False) & (df_log['익절발동']==False)]
            sl = df_log[df_log['손절발동']==True]
            tp = df_log[df_log['익절발동']==True]

            fig.add_trace(go.Scatter(x=buys['날짜'], y=buys['자산'], mode='markers', marker=dict(color='#00FF00', symbol='triangle-up', size=10), name='매수'), row=1, col=1)
            fig.add_trace(go.Scatter(x=sells_reg['날짜'], y=sells_reg['자산'], mode='markers', marker=dict(color='red', symbol='triangle-down', size=10), name='매도'), row=1, col=1)
            fig.add_trace(go.Scatter(x=sl['날짜'], y=sl['자산'], mode='markers', marker=dict(color='purple', symbol='x', size=12), name='손절'), row=1, col=1)
            fig.add_trace(go.Scatter(x=tp['날짜'], y=tp['자산'], mode='markers', marker=dict(color='gold', symbol='star', size=12), name='익절'), row=1, col=1)

            if 'RSI' in df_log.columns:
                fig.add_trace(go.Scatter(x=df_log['날짜'], y=df_log['RSI'], name='RSI', line=dict(color='orange', width=1)), row=2, col=1)
                fig.add_hline(y=70, line_dash="dot", line_color="red", row=2, col=1)
                fig.add_hline(y=30, line_dash="dot", line_color="green", row=2, col=1)
                fig.add_hline(y=50, line_dash="dot", line_color="gray", row=2, col=1)

            fig.add_trace(go.Scatter(x=df_log['날짜'], y=drawdown, name='MDD', line=dict(color='#FF4B4B', width=1), fill='tozeroy'), row=3, col=1)

            fig.update_layout(height=800, template="plotly_dark", hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)

            # 월별 수익률 히트맵
            st.markdown("### 📅 월별 수익률")
            df_log['Year'] = df_log['날짜'].dt.year
            df_log['Month'] = df_log['날짜'].dt.month
            df_log['Returns'] = df_log['자산'].pct_change()
            monthly_ret = df_log.groupby(['Year', 'Month'])['Returns'].apply(lambda x: (x + 1).prod() - 1).reset_index()
            pivot_ret = monthly_ret.pivot(index='Year', columns='Month', values='Returns')
            fig_heat = go.Figure(data=go.Heatmap(
                z=pivot_ret.values * 100, x=pivot_ret.columns, y=pivot_ret.index,
                colorscale='RdBu', zmid=0, texttemplate="%{z:.1f}%"
            ))
            fig_heat.update_layout(title="월별 수익률 Heatmap", height=400)
            st.plotly_chart(fig_heat, use_container_width=True)

            if st.button("✨ Gemini 분석"):
                sl_txt = f"{stop_loss_pct}%" if stop_loss_pct > 0 else "미설정"
                tp_txt = f"{take_profit_pct}%" if take_profit_pct > 0 else "미설정"
                current_params = f"매수: {ma_buy}일 이평, 손절: {sl_txt}, 익절: {tp_txt}"
                anl = ask_gemini_analysis(res, current_params, trade_ticker, st.session_state.get("gemini_api_key"), st.session_state.get("selected_model_name", "gemini-pro"))
                st.session_state["ai_analysis"] = anl    
            
            if "ai_analysis" in st.session_state: st.markdown(st.session_state["ai_analysis"])
            with st.expander("로그"): st.dataframe(df_log)

# ✅ [Tab 4: 실험실 업데이트 (필터 및 기본값 수정)]
with tab4:
    st.markdown("### 🧬 전략 파라미터 자동 최적화")
    
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
        
        limit_mdd = st.number_input("최대 낙폭(MDD) 제한 (%) (0=미사용)", 0.0, 100.0, 0.0)

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
    
    if st.button("🚀 최적 조합 찾기"):
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
            "min_train_ret": min_train_ret, # ✅ 신규 필터 적용
            "min_test_ret": min_test_ret    # ✅ 신규 필터 적용
        }
        
        with st.spinner("최적화 진행 중..."):
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
                st.warning("조건을 만족하는 결과가 없습니다. (필터를 완화하거나 시도 횟수를 늘려보세요)")

    if 'opt_results' in st.session_state:
        df_show = st.session_state['opt_results'].sort_values(st.session_state['sort_metric'], ascending=False).head(top_n)
        
        st.markdown("#### 🏆 상위 결과 (적용 버튼을 누르면 즉시 백테스트 실행)")
        
        # 반복문으로 각 행마다 '적용' 버튼 생성
        for i, row in df_show.iterrows():
            c1, c2 = st.columns([4, 1])
            with c1:
                # 주요 지표만 요약해서 보여줌
                st.dataframe(
                    pd.DataFrame([row]), 
                    hide_index=True,
                    column_config={
                        "Full_수익률(%)": st.column_config.NumberColumn(format="%.2f%%"),
                        "Test_수익률(%)": st.column_config.NumberColumn(format="%.2f%%"),
                        "Train_수익률(%)": st.column_config.NumberColumn(format="%.2f%%"),
                        "Full_MDD(%)": st.column_config.NumberColumn(format="%.2f%%"),
                        "Test_MDD(%)": st.column_config.NumberColumn(format="%.2f%%"),
                        "Full_승률(%)": st.column_config.NumberColumn(format="%.2f%%"),
                    }
                )
            with c2:
                # on_click 콜백 사용
                st.button(f"🥇 적용하기 #{i}", key=f"apply_{i}", on_click=apply_opt_params, args=(row,))
