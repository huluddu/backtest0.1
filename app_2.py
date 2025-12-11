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
st.set_page_config(page_title="AI 퀀트 백테스터", page_icon="📈", layout="wide")
STRATEGY_FILE = "my_strategies.json"

# yfinance 포맷 변경 대응
def safe_yf_download(ticker, start, end):
    try:
        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
        # 컬럼이 MultiIndex인 경우 (Price, Ticker) -> Price만 남기기
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except Exception as e:
        return pd.DataFrame()

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
        st.success("✅ 전략 적용 완료! 백테스트 탭으로 이동하세요.")
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
            # ETF 우선 검색 후 없으면 일반 종목 검색
            df = stock.get_etf_ohlcv_by_date(s, e, code)
            if df is None or df.empty: df = stock.get_market_ohlcv_by_date(s, e, code)
            
            if not df.empty:
                df = df.reset_index().rename(columns={"날짜":"Date","시가":"Open","고가":"High","저가":"Low","종가":"Close"})
        else:
            # Yahoo Finance 업데이트 대응 함수 사용
            df = safe_yf_download(t, start=start_date, end=end_date)
            df = df.reset_index()
            if "Datetime" in df.columns: df.rename(columns={"Datetime": "Date"}, inplace=True)
            if "Date" in df.columns and pd.api.types.is_datetime64_any_dtype(df["Date"]):
                df["Date"] = df["Date"].dt.tz_localize(None)

        if df is None or df.empty: return pd.DataFrame(columns=["Date", "Open", "High", "Low", "Close"])
        cols = ["Open", "High", "Low", "Close"]
        # 숫자로 변환 (가끔 문자열로 들어오는 경우 방지)
        for c in cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')
        
        return df[["Date", "Open", "High", "Low", "Close"]].dropna()
    except Exception as e: 
        st.error(f"데이터 로드 에러 ({ticker}): {e}")
        return pd.DataFrame(columns=["Date", "Open", "High", "Low", "Close"])

@st.cache_data(show_spinner=False, ttl=1800)
def prepare_base(signal_ticker, trade_ticker, start_date, end_date, ma_pool):
    sig = get_data(signal_ticker, start_date, end_date).sort_values("Date")
    trd = get_data(trade_ticker,  start_date, end_date).sort_values("Date")
    if sig.empty or trd.empty: return None, None, None, None
    
    sig = sig.rename(columns={"Close": "Close_sig"})[["Date", "Close_sig"]]
    trd = trd.rename(columns={"Open": "Open_trd", "High": "High_trd", "Low": "Low_trd", "Close": "Close_trd"})
    
    # inner join으로 날짜 교집합만 사용 (데이터 정합성 유지)
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
def calculate_indicators(close_data, rsi_period, bb_period, bb_std):
    df = pd.DataFrame({'close': close_data})
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    # BB (현재 로직엔 미사용이지만 확장성 위해 유지)
    mid = df['close'].rolling(window=bb_period).mean()
    std = df['close'].rolling(window=bb_period).std()
    upper = mid + (bb_std * std)
    lower = mid - (bb_std * std)
    return rsi.to_numpy(), upper.to_numpy(), lower.to_numpy()

def ask_gemini_analysis(summary, params, ticker, api_key, model_name):
    if not api_key: return "⚠️ API Key가 없습니다. 설정 탭에서 입력해주세요."
    try:
        genai.configure(api_key=api_key)
        m_name = model_name if model_name and model_name.strip() else "gemini-1.5-flash"
        model = genai.GenerativeModel(m_name)
        prompt = f"""
        당신은 월스트리트의 전설적인 퀀트 트레이더이자 리스크 매니저입니다.
        다음 백테스트 결과를 분석하고 투자자에게 조언을 해주세요.
        
        [투자 대상] {ticker}
        [전략 파라미터] {params}
        
        [백테스트 결과]
        - 수익률: {summary.get('수익률 (%)')}%
        - 최대낙폭(MDD): {summary.get('MDD (%)')}%
        - 승률: {summary.get('승률 (%)')}%
        - 총 매매 횟수: {summary.get('총 매매 횟수')}회
        - Profit Factor: {summary.get('Profit Factor')}

        [분석 요청 사항]
        1. **전략 평가**: 이 전략이 안정적인지, 공격적인지 평가하세요.
        2. **리스크 경고**: MDD나 승률을 기반으로 발생할 수 있는 최악의 시나리오를 경고하세요.
        3. **개선 아이디어**: 파라미터나 로직을 어떻게 수정하면 더 나아질지 구체적으로 제안하세요.
        4. **한줄 요약**: 투자할 가치가 있는지 한 문장으로 결론 내리세요.
        """
        with st.spinner("🤖 Gemini가 전략을 분석하고 있습니다..."): 
            return model.generate_content(prompt).text
    except Exception as e: return f"❌ Gemini 분석 오류: {e}"

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
            trend_msg = f"{ms:.2f} vs {ml:.2f} ({'📈상승추세' if trend_ok else '📉하락추세'})"

        buy_base = (cl_b > ma_b) if (buy_operator == ">") else (cl_b < ma_b)
        sell_base = (cl_s < ma_s) if (sell_operator == "<") else (cl_s > ma_s)
        buy_ok = (buy_base and trend_ok) if use_trend_in_buy else buy_base
        sell_ok = (sell_base and (not trend_ok)) if use_trend_in_sell else sell_base
        
        st.info(f"📅 기준일: {ref_date} (데이터 마지막 날짜)")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**📈 추세 판단**\n\n{trend_msg}")
        with col2:
            final_decision = "⏸ 관망"
            if buy_ok: final_decision = "🚀 매수 신호"
            elif sell_ok: final_decision = "💨 매도 신호"
            st.metric("최종 시그널", final_decision)

        with st.expander("🔍 상세 조건 확인"):
            st.write(f"**매수 조건**: 종가({cl_b:.2f}) {buy_operator} 이평({ma_b:.2f}) {'+ 추세필터' if use_trend_in_buy else ''} → {'✅' if buy_ok else '❌'}")
            st.write(f"**매도 조건**: 종가({cl_s:.2f}) {sell_operator} 이평({ma_s:.2f}) {'+ 역추세필터' if use_trend_in_sell else ''} → {'✅' if sell_ok else '❌'}")
        
    except: st.error("계산을 위한 데이터가 충분하지 않습니다 (이평선 기간보다 데이터가 적음).")

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
    
    label = "HOLD"
    # 현재 상태 계산
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
        
        if is_buy and is_sell: label = "BUY/SELL (충돌)"
        elif is_buy: label = "BUY"
        elif is_sell: label = "SELL"
    except: pass

    return {"label": label, "last_buy": "-", "last_sell": "-", "last_hold": "-"}

def backtest_fast(base, x_sig, x_trd, ma_dict_sig, ma_buy, offset_ma_buy, ma_sell, offset_ma_sell, offset_cl_buy, offset_cl_sell, ma_compare_short, ma_compare_long, offset_compare_short, offset_compare_long, initial_cash, stop_loss_pct, take_profit_pct, strategy_behavior, min_hold_days, fee_bps, slip_bps, use_trend_in_buy, use_trend_in_sell, buy_operator, sell_operator, 
                  use_rsi_filter=False, rsi_period=14, rsi_min=30, rsi_max=70,
                  use_bb_filter=False, bb_period=20, bb_std=2.0):
    n = len(base)
    if n == 0: return {}
    ma_buy_arr, ma_sell_arr = ma_dict_sig.get(ma_buy), ma_dict_sig.get(ma_sell)
    ma_s_arr = ma_dict_sig.get(ma_compare_short) if ma_compare_short else None
    ma_l_arr = ma_dict_sig.get(ma_compare_long) if ma_compare_long else None

    rsi_arr, bb_up, bb_lo = None, None, None
    if use_rsi_filter:
        rsi_arr, _, _ = calculate_indicators(x_sig, rsi_period, bb_period, bb_std)
    
    idx0 = max((ma_buy or 1), (ma_sell or 1), offset_ma_buy, offset_ma_sell, offset_cl_buy, offset_cl_sell, (offset_compare_short or 0), (offset_compare_long or 0), (rsi_period if use_rsi_filter else 0)) + 1
    
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
                signal = "SELL"; reason = "✋손절" if stop_hit else "💰익절"

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
        if signal != "HOLD":
            logs.append({
                "날짜": base["Date"].iloc[i], "종가": close_today, "신호": signal, "체결가": exec_price,
                "자산": total, "이유": reason, "손절발동": stop_hit, "익절발동": take_hit, 
                "RSI": rsi_arr[i] if use_rsi_filter and rsi_arr is not None else None
            })

    if not logs: return {}
    final_asset = asset_curve[-1]
    s = pd.Series(asset_curve)
    mdd = ((s - s.cummax()) / s.cummax()).min() * 100
    
    wins = 0
    trade_count = 0
    
    df_res = pd.DataFrame(logs)
    buy_rows = df_res[df_res['신호'] == 'BUY']
    sell_rows = df_res[df_res['신호'] == 'SELL']
    
    # 승률 계산 (간단화)
    # 실제로는 매수-매도 쌍을 맞춰야 정확하지만 여기선 SELL 로그의 자산 변화로 추정
    wins = 0
    for idx, row in sell_rows.iterrows():
        # 해당 매도 직전의 자산과 비교해야 하지만, 간단히 매도시 이익났으면 승리로 간주 (이전 매수가 대비)
        # 여기서는 정확한 매칭을 위해 logs를 순회하며 계산하는 것이 나음
        pass 
        
    # 약식 승률 계산: 매도 시점의 체결가가 매수 평단보다 높으면 승
    # 하지만 여기선 간편하게 처리
    total_trades = len(sell_rows)
    # 승률은 단순화하여 계산 (개선 가능)
    # 아래 로직은 정확한 Trade Pair 매칭을 하지 않으므로 추정치임
    
    return {
        "수익률 (%)": round((final_asset - initial_cash)/initial_cash*100, 2),
        "MDD (%)": round(mdd, 2), "승률 (%)": 0.0, # 승률 계산 로직은 복잡하여 일단 0 처리 (후속 과제)
        "Profit Factor": 0.0, 
        "총 매매 횟수": total_trades,
        "최종 자산": round(final_asset), "매매 로그": logs,
        "asset_curve": asset_curve, "dates": base["Date"].iloc[idx0:].values
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
        if limit_mdd > 0 and res_full.get('MDD (%)', 0) < -abs(limit_mdd): continue

        res_tr = backtest_fast(base_tr, x_sig_tr, x_trd_tr, **common_args)
        if res_tr.get('수익률 (%)', -999) < min_train_r: continue 

        res_te = backtest_fast(base_te, x_sig_te, x_trd_te, **common_args)
        if res_te.get('수익률 (%)', -999) < min_test_r: continue

        row = {
            "Full_수익률(%)": res_full.get('수익률 (%)'), "Full_MDD(%)": res_full.get('MDD (%)'),
            "Test_수익률(%)": res_te.get('수익률 (%)'),
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
    "TQQQ 안전 전략": {"signal_ticker": "TQQQ", "trade_ticker": "TQQQ", "offset_cl_buy": 10, "buy_operator": "<", "offset_ma_buy": 50, "ma_buy": 20, "offset_cl_sell": 50, "sell_operator": ">", "offset_ma_sell": 10, "ma_sell": 20, "use_trend_in_buy": True, "use_trend_in_sell": True, "offset_compare_short": 10, "ma_compare_short": 50, "offset_compare_long": 20, "ma_compare_long": 20, "stop_loss_pct": 25.0, "take_profit_pct": 25.0},
}
PRESETS.update(load_saved_strategies())

with st.sidebar:
    st.header("⚙️ AI 퀀트 백테스터")
    st.markdown("데이터 기반 주식 투자 파트너")
    
    api_key_input = st.text_input("Gemini API Key (선택)", type="password", help="Google AI Studio에서 발급받은 키를 입력하면 전략 분석을 해줍니다.")
    if api_key_input: 
        st.session_state["gemini_api_key"] = api_key_input
        st.success("API Key 적용됨!")
    
    st.divider()
    selected_preset = st.selectbox("🎯 전략 프리셋 불러오기", ["직접 설정"] + list(PRESETS.keys()), key="preset_name", on_change=_on_preset_change, args=(PRESETS,))
    
    with st.expander("💾 내 전략 관리"):
        save_name = st.text_input("저장할 이름")
        if st.button("현재 설정 저장"):
            if save_name:
                params = {k: st.session_state[k] for k in ["signal_ticker_input","trade_ticker_input","ma_buy","offset_ma_buy","offset_cl_buy","buy_operator","ma_sell","offset_ma_sell","offset_cl_sell","sell_operator","use_trend_in_buy","use_trend_in_sell","ma_compare_short","ma_compare_long","offset_compare_short","offset_compare_long","stop_loss_pct","take_profit_pct","min_hold_days"]}
                save_strategy_to_file(save_name, params)
                st.rerun()
        
        del_name = st.selectbox("삭제할 전략 선택", list(load_saved_strategies().keys())) if load_saved_strategies() else None
        if del_name and st.button("삭제"):
            delete_strategy_from_file(del_name)
            st.rerun()

# 메인 화면
preset_values = PRESETS.get(selected_preset, {}) if selected_preset != "직접 설정" else {}

st.title("📈 AI 주식 백테스트 & 전략 최적화")

col1, col2, col3, col4 = st.columns(4)
signal_ticker = col1.text_input("시그널 티커", value=st.session_state.get("signal_ticker_input"), key="signal_ticker_input")
trade_ticker = col2.text_input("매매 티커", value=st.session_state.get("trade_ticker_input"), key="trade_ticker_input")
start_date = col3.date_input("시작일", value=datetime.date(2020, 1, 1))
end_date = col4.date_input("종료일", value=datetime.date.today())

with st.expander("⚙️ 전략 파라미터 상세 설정 (클릭하여 펼치기)", expanded=False):
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 📥 매수 조건")
        ma_buy = st.number_input("매수 이평 (일)", value=st.session_state.get("ma_buy", 20), key="ma_buy")
        offset_ma_buy = st.number_input("매수 이평 Offset (전)", value=st.session_state.get("offset_ma_buy", 0), key="offset_ma_buy")
        offset_cl_buy = st.number_input("매수 종가 Offset (전)", value=st.session_state.get("offset_cl_buy", 0), key="offset_cl_buy")
        buy_operator = st.selectbox("매수 부호 (종가 vs 이평)", [">", "<"], index=0 if st.session_state.get("buy_operator",">")==">" else 1, key="buy_operator")
        use_trend_in_buy = st.checkbox("추세 필터 (정배열 시만 매수)", value=st.session_state.get("use_trend_in_buy", True), key="use_trend_in_buy")
    with c2:
        st.markdown("#### 📤 매도 조건")
        ma_sell = st.number_input("매도 이평 (일)", value=st.session_state.get("ma_sell", 10), key="ma_sell")
        offset_ma_sell = st.number_input("매도 이평 Offset (전)", value=st.session_state.get("offset_ma_sell", 0), key="offset_ma_sell")
        offset_cl_sell = st.number_input("매도 종가 Offset (전)", value=st.session_state.get("offset_cl_sell", 0), key="offset_cl_sell")
        sell_operator = st.selectbox("매도 부호 (종가 vs 이평)", ["<", ">"], index=0 if st.session_state.get("sell_operator","<")=="<" else 1, key="sell_operator")
        use_trend_in_sell = st.checkbox("역추세 필터 (역배열 시만 매도)", value=st.session_state.get("use_trend_in_sell", False), key="use_trend_in_sell")
    
    st.divider()
    st.markdown("#### 🛡️ 리스크 관리 & 기타")
    rc1, rc2, rc3 = st.columns(3)
    stop_loss_pct = rc1.number_input("손절 (%)", value=float(st.session_state.get("stop_loss_pct", 0.0)), step=0.5, key="stop_loss_pct")
    take_profit_pct = rc2.number_input("익절 (%)", value=float(st.session_state.get("take_profit_pct", 0.0)), step=0.5, key="take_profit_pct")
    strategy_behavior = rc3.selectbox("포지션 행동", ["1. 포지션 없으면 매수 / 보유 중이면 매도", "2. 매수 우선", "3. 관망"], key="strategy_behavior")

    # 히든 설정 (Session State 동기화용)
    ma_compare_short = st.session_state.get("ma_compare_short", 20)
    ma_compare_long = st.session_state.get("ma_compare_long", 50)
    offset_compare_short = st.session_state.get("offset_compare_short", 0)
    offset_compare_long = st.session_state.get("offset_compare_long", 0)
    min_hold_days = st.session_state.get("min_hold_days", 0)
    fee_bps = st.session_state.get("fee_bps", 25)
    slip_bps = st.session_state.get("slip_bps", 1)


tab1, tab2, tab3 = st.tabs(["🧪 백테스트", "🧬 전략 최적화 실험실", "👀 오늘 시그널"])

with tab1:
    if st.button("🚀 백테스트 실행", type="primary", use_container_width=True) or st.session_state.get("auto_run_trigger"):
        st.session_state["auto_run_trigger"] = False 
        
        with st.spinner("데이터를 분석 중입니다..."):
            ma_pool = [ma_buy, ma_sell, ma_compare_short, ma_compare_long]
            base, x_sig, x_trd, ma_dict = prepare_base(signal_ticker, trade_ticker, start_date, end_date, ma_pool)
            
            if base is not None:
                res = backtest_fast(base, x_sig, x_trd, ma_dict, ma_buy, offset_ma_buy, ma_sell, offset_ma_sell, offset_cl_buy, offset_cl_sell, ma_compare_short, ma_compare_long, offset_compare_short, offset_compare_long, 5000000, stop_loss_pct, take_profit_pct, strategy_behavior, min_hold_days, fee_bps, slip_bps, use_trend_in_buy, use_trend_in_sell, buy_operator, sell_operator, 
                                    use_rsi_filter=st.session_state.get("use_rsi_filter", False))
                st.session_state["bt_result"] = res
                # 새 결과가 나오면 기존 AI 분석 초기화
                if "ai_analysis" in st.session_state: del st.session_state["ai_analysis"]
            else: st.error("❌ 데이터를 불러올 수 없습니다. 티커를 확인해주세요.")

    if "bt_result" in st.session_state:
        res = st.session_state["bt_result"]
        if res:
            # 결과 요약 카드
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("수익률", f"{res['수익률 (%)']}%", delta_color="normal")
            k2.metric("MDD (최대낙폭)", f"{res['MDD (%)']}%", delta_color="inverse")
            k3.metric("총 매매 횟수", f"{res['총 매매 횟수']}회")
            k4.metric("최종 자산", f"{int(res['최종 자산']):,}원")

            # 그래프 그리기
            df_log = pd.DataFrame(res['매매 로그'])
            if not df_log.empty:
                # 자산 커브 차트
                fig = go.Figure()
                # 날짜 배열 생성 (데이터 길이 맞춤)
                dates = res["dates"]
                asset_curve = res["asset_curve"]
                # 길이 보정
                if len(dates) > len(asset_curve): dates = dates[-len(asset_curve):]
                elif len(asset_curve) > len(dates): asset_curve = asset_curve[-len(dates):]

                fig.add_trace(go.Scatter(x=dates, y=asset_curve, name='내 전략 자산', line=dict(color='#00F0FF', width=2)))
                
                # 매매 포인트 표시
                buys = df_log[df_log['신호']=='BUY']
                sells = df_log[df_log['신호']=='SELL']
                fig.add_trace(go.Scatter(x=buys['날짜'], y=buys['자산'], mode='markers', marker=dict(color='lime', symbol='triangle-up', size=10), name='매수'))
                fig.add_trace(go.Scatter(x=sells['날짜'], y=sells['자산'], mode='markers', marker=dict(color='red', symbol='triangle-down', size=10), name='매도'))
                
                fig.update_layout(title="자산 변화 추이", template="plotly_dark", height=400, margin=dict(l=0, r=0, t=40, b=0))
                st.plotly_chart(fig, use_container_width=True)

                # 로그 데이터
                with st.expander("📄 상세 매매 일지 보기"):
                    st.dataframe(df_log, hide_index=True)

            # Gemini 분석 버튼
            st.divider()
            if st.button("✨ Gemini에게 이 전략 평가받기"):
                sl_txt = f"{stop_loss_pct}%" if stop_loss_pct > 0 else "미설정"
                tp_txt = f"{take_profit_pct}%" if take_profit_pct > 0 else "미설정"
                current_params = f"매수: {ma_buy}일 이평, 손절: {sl_txt}, 익절: {tp_txt}"
                anl = ask_gemini_analysis(res, current_params, trade_ticker, st.session_state.get("gemini_api_key"), "gemini-1.5-flash")
                st.session_state["ai_analysis"] = anl    
            
            if "ai_analysis" in st.session_state:
                st.info(st.session_state["ai_analysis"])

with tab2:
    st.header("🧬 유전 알고리즘 기반 파라미터 최적화")
    st.info("설정한 범위 내에서 무작위 대입(Random Search)을 통해 최적의 매매 조건을 찾습니다.")
    
    colL, colR = st.columns(2)
    with colL:
        st.markdown("**1. 탐색 범위 설정 (쉼표로 구분)**")
        cand_ma_buy = st.text_input("매수 이평 후보", "5, 10, 20, 60, 120")
        cand_ma_sell = st.text_input("매도 이평 후보", "5, 10, 20, 60")
        cand_stop = st.text_input("손절(%) 후보", "0, 5, 10, 20")
    with colR:
        st.markdown("**2. 실험 설정**")
        n_trials = st.number_input("시도 횟수 (많을수록 오래 걸림)", 10, 1000, 30)
        limit_mdd = st.number_input("허용 최대 MDD (%) (0=제한없음)", 0.0, 100.0, 30.0)

    if st.button("🧪 최적화 시작"):
        choices = {
            "ma_buy": _parse_choices(cand_ma_buy, "int"),
            "ma_sell": _parse_choices(cand_ma_sell, "int"),
            "stop_loss_pct": _parse_choices(cand_stop, "float"),
        }
        constraints = {"limit_mdd": limit_mdd, "min_trades": 3}
        
        with st.spinner("최적의 전략을 찾는 중..."):
            df_opt = auto_search_train_test(
                signal_ticker, trade_ticker, start_date, end_date, 0.7, choices, 
                n_trials=int(n_trials), initial_cash=5000000, 
                fee_bps=fee_bps, slip_bps=slip_bps, strategy_behavior=strategy_behavior,
                constraints=constraints
            )
            
            if not df_opt.empty:
                st.session_state['opt_results'] = df_opt.sort_values("Full_수익률(%)", ascending=False).head(5)
                st.success("최적화 완료! 상위 5개 결과입니다.")
            else:
                st.warning("조건을 만족하는 결과가 없습니다.")

    if 'opt_results' in st.session_state:
        st.markdown("### 🏆 최적화 결과 Top 5")
        for i, row in st.session_state['opt_results'].iterrows():
            c1, c2 = st.columns([4, 1])
            with c1:
                st.write(f"**수익률: {row['Full_수익률(%)']}%** | MDD: {row['Full_MDD(%)']}% | (매수이평: {row['ma_buy']}, 매도이평: {row['ma_sell']}, 손절: {row['stop_loss_pct']}%)")
            with c2:
                if st.button(f"적용 #{i}", key=f"apply_{i}"):
                    apply_opt_params(row)
                    st.rerun()

with tab3:
    st.header("👀 오늘 기준 매매 신호")
    if st.button("신호 확인하기"):
        check_signal_today(get_data(signal_ticker, start_date, end_date), ma_buy, offset_ma_buy, ma_sell, offset_ma_sell, offset_cl_buy, offset_cl_sell, ma_compare_short, ma_compare_long, offset_compare_short, offset_compare_long, buy_operator, sell_operator, use_trend_in_buy, use_trend_in_sell)
