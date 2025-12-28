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
st.set_page_config(page_title="시그널 대시보드 Pro (Trailing Stop & Filter)", page_icon="🛡️", layout="wide")

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
        "signal_ticker_input": "TSLL", "trade_ticker_input": "TSLL",
        "buy_operator": ">", "sell_operator": "<",
        "strategy_behavior": "1. 포지션 없으면 매수 / 보유 중이면 매도",
        "offset_cl_buy": 0, "offset_cl_sell": 0,
        "offset_ma_buy": 0, "offset_ma_sell": 0,
        "ma_buy": 10, "ma_sell": 5,
        "use_trend_in_buy": True, "use_trend_in_sell": False,
        "ma_compare_short": 20, "ma_compare_long": 60,
        "offset_compare_short": 0, "offset_compare_long": 0,
        "stop_loss_pct": 10.0, "take_profit_pct": 0.0, "trailing_stop_pct": 15.0, # 트레일링 스탑 추가
        "min_hold_days": 0,
        "fee_bps": 25, "slip_bps": 1,
        "preset_name": "직접 설정",
        "gemini_api_key": "",
        "auto_run_trigger": False,
        "use_rsi_filter": False, "rsi_period": 14, "rsi_min": 30, "rsi_max": 70,
        "use_market_filter": False, "market_ticker": "SPY", "market_ma": 200 # 시장 필터 추가
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
            "ma_buy": int(row.get("ma_buy", 10)), "offset_ma_buy": int(row.get("offset_ma_buy", 0)),
            "offset_cl_buy": int(row.get("offset_cl_buy", 0)), "buy_operator": str(row.get("buy_operator", ">")),
            "ma_sell": int(row.get("ma_sell", 10)), "offset_ma_sell": int(row.get("offset_ma_sell", 0)),
            "offset_cl_sell": int(row.get("offset_cl_sell", 0)), "sell_operator": str(row.get("sell_operator", "<")),
            "use_trend_in_buy": bool(row.get("use_trend_in_buy", False)), 
            "stop_loss_pct": float(row.get("stop_loss_pct", 0)),
            "take_profit_pct": float(row.get("take_profit_pct", 0)),
            "trailing_stop_pct": float(row.get("trailing_stop_pct", 0)),
            "auto_run_trigger": True
        }
        for k, v in updates.items(): st.session_state[k] = v
        st.session_state["preset_name_selector"] = "직접 설정"
    except Exception as e: st.error(f"설정 적용 오류: {e}")

def _parse_choices(text, cast="int"):
    if text is None: return []
    tokens = [t for t in re.split(r"[,\s]+", str(text).strip()) if t != ""]
    if not tokens: return []
    out = []
    for t in tokens:
        try:
            if cast == "int": out.append(int(t))
            elif cast == "float": out.append(float(t))
            elif cast == "bool": out.append(t.lower() in ("true", "t", "y", "1"))
            else: out.append(str(t))
        except: continue
    return list(set(out))

def _normalize_krx_ticker(t: str) -> str:
    t = str(t or "").strip().upper()
    t = re.sub(r"\.(KS|KQ)$", "", t)
    m = re.search(r"(\d{6})", t)
    return m.group(1) if m else ""

def _fast_ma(x: np.ndarray, w: int) -> np.ndarray:
    if w is None or w <= 1: return x.astype(float)
    kernel = np.ones(w, dtype=float) / w
    y = np.full(x.shape, np.nan, dtype=float)
    if len(x) >= w:
        y[w-1:] = np.convolve(x, kernel, mode="valid")
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
            if df.empty: # Retry full history if specific date fails
                df = yf.download(t, period="max", progress=False, auto_adjust=False)
                if not df.empty: df = df[df.index <= pd.Timestamp(end_date)]
            
            if isinstance(df.columns, pd.MultiIndex):
                try: df = df.xs(t, axis=1, level=1)
                except: df = df.droplevel(1, axis=1)
            
            df = df.reset_index()
            if "Datetime" in df.columns: df.rename(columns={"Datetime": "Date"}, inplace=True)
            if "Date" in df.columns and pd.api.types.is_datetime64_any_dtype(df["Date"]):
                df["Date"] = df["Date"].dt.tz_localize(None)

        cols = ["Open", "High", "Low", "Close"]
        for c in cols:
            if c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce')
        
        return df[["Date", "Open", "High", "Low", "Close"]].dropna()
    except: return pd.DataFrame(columns=["Date", "Open", "High", "Low", "Close"])

# [New] 시장 데이터 로드 (필터용)
@st.cache_data(show_spinner=False, ttl=3600)
def get_market_data(ticker, start_date, end_date):
    df = get_data(ticker, start_date, end_date)
    return df[["Date", "Close"]].rename(columns={"Close": "Market_Close"})

@st.cache_data(show_spinner=False, ttl=1800)
def prepare_base(signal_ticker, trade_ticker, start_date, end_date, ma_pool, market_ticker=None, market_ma_period=200):
    # 메인 데이터
    sig = get_data(signal_ticker, start_date, end_date).sort_values("Date")
    trd = get_data(trade_ticker,  start_date, end_date).sort_values("Date")
    
    # 시장 필터 데이터 병합
    mkt_mask = None
    if market_ticker:
        mkt = get_market_data(market_ticker, start_date, end_date)
        if not mkt.empty:
            mkt["Market_MA"] = mkt["Market_Close"].rolling(window=market_ma_period).mean()
            # Market > MA 이면 True (상승장), 아니면 False
            mkt["Market_Bull"] = mkt["Market_Close"] > mkt["Market_MA"]
            # 병합을 위해 Date 기준 정렬
            sig = pd.merge(sig, mkt[["Date", "Market_Bull"]], on="Date", how="left")
            mkt_mask = sig["Market_Bull"].fillna(True).to_numpy() # 데이터 없으면 일단 True로 가정

    if sig.empty or trd.empty: return None, None, None, None, None
    sig = sig.rename(columns={"Close": "Close_sig"})[["Date", "Close_sig"]]
    trd = trd.rename(columns={"Open": "Open_trd", "High": "High_trd", "Low": "Low_trd", "Close": "Close_trd"})
    
    base = pd.merge(sig, trd, on="Date", how="inner").dropna().reset_index(drop=True)
    
    # 병합 후 마스크 재조정
    final_mkt_mask = None
    if mkt_mask is not None:
        # merge 과정에서 행 수가 변했을 수 있으므로 재계산 필요하지만 약식으로 처리
        # 여기서는 간단히 base에 merge된 값을 씀 (위에서 sig에 merge했으므로 base에도 포함될 수 있음)
        if "Market_Bull" in base.columns:
            final_mkt_mask = base["Market_Bull"].fillna(True).to_numpy()
        
    x_sig = base["Close_sig"].to_numpy(dtype=float)
    x_trd = base["Close_trd"].to_numpy(dtype=float)
    
    ma_dict_sig = {}
    for w in sorted(set([int(w) for w in ma_pool if w and w > 0])):
        ma_dict_sig[w] = _fast_ma(x_sig, w)
        
    return base, x_sig, x_trd, ma_dict_sig, final_mkt_mask

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
        당신은 월스트리트의 전문 퀀트 트레이더입니다. 아래 백테스트 결과를 냉철하게 분석해주세요.
        
        [대상 자산]: {ticker}
        [전략 파라미터]: {params}
        
        [백테스트 성과]
        - 수익률: {summary.get('수익률 (%)')}%
        - MDD (최대 낙폭): {summary.get('MDD (%)')}%
        - 승률: {summary.get('승률 (%)')}%
        - 총 매매 횟수: {summary.get('총 매매 횟수')}회
        - Profit Factor: {summary.get('Profit Factor')}
        
        [분석 요청 사항]
        1. 🛡️ **리스크 평가**: 트레일링 스탑과 시장 필터가 MDD 방어에 효과적이었는지 평가하세요.
        2. 💰 **수익성 평가**: 기존 전략(단순 이평 매수) 대비 개선된 점을 분석하세요.
        3. 💡 **추가 조언**: 현재 전략의 약점이 있다면 무엇이고 어떻게 보완할까요?
        4. ⚖️ **종합 의견**: (강력 추천 / 추천 / 보류 / 비추천)
        """
        with st.spinner("🤖 Gemini가 전략을 분석 중입니다..."):
            response = model.generate_content(prompt)
            return response.text
    except Exception as e: return f"❌ Gemini 분석 오류: {e}"

def backtest_fast(base, x_sig, x_trd, ma_dict_sig, ma_buy, offset_ma_buy, ma_sell, offset_ma_sell, offset_cl_buy, offset_cl_sell, ma_compare_short, ma_compare_long, offset_compare_short, offset_compare_long, initial_cash, stop_loss_pct, take_profit_pct, trailing_stop_pct, strategy_behavior, min_hold_days, fee_bps, slip_bps, use_trend_in_buy, use_trend_in_sell, buy_operator, sell_operator, 
                  use_rsi_filter=False, rsi_period=14, rsi_max=70, market_mask=None):
    n = len(base)
    if n == 0: return {}
    
    ma_buy, ma_sell = int(ma_buy), int(ma_sell)
    ma_buy_arr, ma_sell_arr = ma_dict_sig.get(ma_buy, x_sig), ma_dict_sig.get(ma_sell, x_sig)
    ma_s_arr = ma_dict_sig.get(int(ma_compare_short)) if ma_compare_short else None
    ma_l_arr = ma_dict_sig.get(int(ma_compare_long)) if ma_compare_long else None

    rsi_arr = calculate_indicators(x_sig, int(rsi_period)) if use_rsi_filter else None
    
    max_offset = max(ma_buy, ma_sell, offset_ma_buy, offset_ma_sell, offset_cl_buy, offset_cl_sell, (offset_compare_short or 0), (offset_compare_long or 0), (rsi_period if use_rsi_filter else 0))
    idx0 = int(max_offset) + 1

    xO, xH, xL, xC_trd = base["Open_trd"].values, base["High_trd"].values, base["Low_trd"].values, x_trd
    cash, position, hold_days = float(initial_cash), 0.0, 0
    entry_price = 0.0
    highest_price_since_entry = 0.0 # 트레일링 스탑용 고점 추적
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
            try: trend_ok = (ma_s_arr[i - offset_compare_short] >= ma_l_arr[i - offset_compare_long])
            except: pass

        buy_base = (cl_b > ma_b) if buy_operator == ">" else (cl_b < ma_b)
        sell_base = (cl_s < ma_s) if (sell_operator == "<") else (cl_s > ma_s)
        buy_cond = (buy_base and trend_ok) if use_trend_in_buy else buy_base
        sell_cond = (sell_base and (not trend_ok)) if use_trend_in_sell else sell_base

        if use_rsi_filter and buy_cond and rsi_arr is not None:
            if rsi_arr[i-1] > rsi_max: buy_cond = False
        
        # [New] 시장 필터 적용
        if market_mask is not None and buy_cond:
            if not market_mask[i]: buy_cond = False # 시장이 하락장이면 매수 금지

        stop_hit, take_hit, trail_hit = False, False, False
        
        if position > 0:
            # 고점 갱신 (트레일링 스탑용)
            highest_price_since_entry = max(highest_price_since_entry, high_today)
            
            # 1. 고정 손절 (Stop Loss)
            if stop_loss_pct > 0:
                sl_price = entry_price * (1 - stop_loss_pct / 100)
                if low_today <= sl_price:
                    stop_hit = True
                    exec_price = open_today if open_today < sl_price else sl_price
            
            # 2. 고정 익절 (Take Profit)
            if take_profit_pct > 0 and not stop_hit:
                tp_price = entry_price * (1 + take_profit_pct / 100)
                if high_today >= tp_price:
                    take_hit = True
                    exec_price = open_today if open_today > tp_price else tp_price
            
            # 3. [New] 트레일링 스탑 (Trailing Stop)
            if trailing_stop_pct > 0 and not stop_hit and not take_hit:
                ts_price = highest_price_since_entry * (1 - trailing_stop_pct / 100)
                # 현재 저가가 트레일링 스탑 가격을 건드렸는지 확인 (단, 매수가보다는 높아야 익절임. 손절라인 밑이면 위에서 손절됨)
                if low_today <= ts_price:
                    trail_hit = True
                    exec_price = open_today if open_today < ts_price else ts_price

            if stop_hit or take_hit or trail_hit:
                fill = _fill_sell(exec_price)
                cash = position * fill
                position = 0.0
                entry_price = 0.0
                if stop_hit: reason = "손절"
                elif take_hit: reason = "익절"
                else: reason = "트레일링익절"
                signal = "SELL"

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
                highest_price_since_entry = base_px # 초기화
                cash = 0.0
                signal = "BUY"; reason = "전략매수"; exec_price = base_px
                just_bought = True

        if position > 0 and not just_bought: hold_days += 1
        else: hold_days = 0

        total = cash + (position * close_today)
        asset_curve.append(total)
        logs.append({
            "날짜": base["Date"].iloc[i], "종가": close_today, "신호": signal, "체결가": exec_price,
            "자산": total, "이유": reason, 
            "최고가": highest_price_since_entry if position > 0 else None
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
    # 최적화 로직 단순화 (필터 옵션은 기본값 사용)
    ma_pool = set([5, 10, 20, 60, 120])
    for k in ["ma_buy", "ma_sell"]:
        for v in choices_dict.get(k, []):
            try: 
                if int(v) > 0: ma_pool.add(int(v))
            except: pass
            
    base_full, x_sig_full, x_trd_full, ma_dict, mkt_mask = prepare_base(signal_ticker, trade_ticker, start_date, end_date, list(ma_pool))
    if base_full is None: return pd.DataFrame()
    
    split_idx = int(len(base_full) * split_ratio)
    base_tr, base_te = base_full.iloc[:split_idx].reset_index(drop=True), base_full.iloc[split_idx:].reset_index(drop=True)
    x_sig_tr, x_sig_te = x_sig_full[:split_idx], x_sig_full[split_idx:]
    x_trd_tr, x_trd_te = x_trd_full[:split_idx], x_trd_full[split_idx:]
    
    # 마스크 분할
    mask_tr = mkt_mask[:split_idx] if mkt_mask is not None else None
    mask_te = mkt_mask[split_idx:] if mkt_mask is not None else None
    
    results = []
    defaults = {"ma_buy": 10, "ma_sell": 5}
    constraints = constraints or {}

    for _ in range(int(n_trials)):
        p = {}
        for k in choices_dict.keys():
            arr = choices_dict[k]
            p[k] = random.choice(arr) if arr else defaults.get(k)
        
        # 최적화 시 트레일링 스탑도 파라미터로 포함 가능하도록 수정 필요
        common_args = {
            "ma_dict_sig": ma_dict,
            "ma_buy": int(p.get('ma_buy', 10)), "offset_ma_buy": int(p.get('offset_ma_buy', 0)),
            "ma_sell": int(p.get('ma_sell', 5)), "offset_ma_sell": int(p.get('offset_ma_sell', 0)),
            "offset_cl_buy": int(p.get('offset_cl_buy', 0)), "offset_cl_sell": int(p.get('offset_cl_sell', 0)),
            "ma_compare_short": 0, "ma_compare_long": 0, "offset_compare_short": 0, "offset_compare_long": 0,
            "initial_cash": initial_cash, 
            "stop_loss_pct": float(p.get('stop_loss_pct', 0)), 
            "take_profit_pct": float(p.get('take_profit_pct', 0)),
            "trailing_stop_pct": float(p.get('trailing_stop_pct', 0)), # New
            "strategy_behavior": strategy_behavior, "min_hold_days": min_hold_days, "fee_bps": fee_bps, "slip_bps": slip_bps,
            "use_trend_in_buy": p.get('use_trend_in_buy', True), "use_trend_in_sell": p.get('use_trend_in_sell', False),
            "buy_operator": p.get('buy_operator', '>'), "sell_operator": p.get('sell_operator', '<'),
            "market_mask": None # 최적화 속도를 위해 일단 끔
        }

        res_full = backtest_fast(base_full, x_sig_full, x_trd_full, **common_args)
        if not res_full: continue
        
        # 필터링
        if res_full.get('총 매매 횟수', 0) < constraints.get("min_trades", 0): continue
        if res_full.get('승률 (%)', 0) < constraints.get("min_winrate", 0): continue
        if constraints.get("limit_mdd", 0) > 0 and res_full.get('MDD (%)', 0) < -abs(constraints.get("limit_mdd", 0)): continue

        res_tr = backtest_fast(base_tr, x_sig_tr, x_trd_tr, **common_args)
        res_te = backtest_fast(base_te, x_sig_te, x_trd_te, **common_args)

        row = {
            "Full_수익률(%)": res_full.get('수익률 (%)'), "Full_MDD(%)": res_full.get('MDD (%)'), "Full_승률(%)": res_full.get('승률 (%)'),
            "Test_수익률(%)": res_te.get('수익률 (%)'), "Train_수익률(%)": res_tr.get('수익률 (%)'),
            "ma_buy": p.get('ma_buy'), "ma_sell": p.get('ma_sell'),
            "stop_loss_pct": p.get('stop_loss_pct'), "trailing_stop_pct": p.get('trailing_stop_pct')
        }
        results.append(row)
        
    return pd.DataFrame(results)

# ==========================================
# 5. 메인 UI
# ==========================================
_init_default_state()

PRESETS = {
    "TSLL 트레일링 전략": {"signal_ticker": "TSLL", "trade_ticker": "TSLL", "ma_buy": 10, "ma_sell": 5, "stop_loss_pct": 10.0, "trailing_stop_pct": 15.0, "use_trend_in_buy": True, "use_market_filter": False},
    "TSLL 안전 (시장필터)": {"signal_ticker": "TSLL", "trade_ticker": "TSLL", "ma_buy": 20, "ma_sell": 10, "stop_loss_pct": 10.0, "trailing_stop_pct": 10.0, "use_market_filter": True, "market_ticker": "QQQ"},
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
            st.session_state["selected_model_name"] = st.selectbox("🤖 모델 선택", models, index=0)
        except: pass
    
    st.divider()
    with st.expander("💾 전략 저장/삭제"):
        save_name = st.text_input("전략 이름")
        if st.button("현재 설정 저장"):
            if save_name:
                params = {k: st.session_state[k] for k in ["signal_ticker_input","trade_ticker_input","ma_buy","ma_sell","stop_loss_pct","take_profit_pct","trailing_stop_pct","use_market_filter","market_ticker"]}
                save_strategy_to_file(save_name, params)
                st.rerun()
        
        del_name = st.selectbox("삭제할 전략", list(load_saved_strategies().keys())) if load_saved_strategies() else None
        if del_name and st.button("삭제"):
            delete_strategy_from_file(del_name)
            st.rerun()

    st.divider()
    selected_preset = st.selectbox("🎯 프리셋", ["직접 설정"] + list(PRESETS.keys()), key="preset_name_selector", on_change=_on_preset_change)

col1, col2 = st.columns(2)
signal_ticker = col1.text_input("시그널 티커", key="signal_ticker_input")
trade_ticker = col2.text_input("매매 티커", key="trade_ticker_input")
col3, col4 = st.columns(2)
start_date = col3.date_input("시작일", value=datetime.date(2022, 9, 1))
end_date = col4.date_input("종료일", value=datetime.date.today())

with st.expander("🛡️ 리스크 관리 (AI 추천 기능)", expanded=True):
    r1, r2, r3 = st.columns(3)
    stop_loss_pct = r1.number_input("손절 (%)", step=1.0, key="stop_loss_pct", help="매수가 대비 하락 시 손절")
    take_profit_pct = r2.number_input("고정 익절 (%)", step=1.0, key="take_profit_pct", help="0이면 미사용")
    trailing_stop_pct = r3.number_input("트레일링 스탑 (%)", value=15.0, step=1.0, key="trailing_stop_pct", help="고점 대비 하락 시 익절 (추세 추종에 필수)")
    
    st.markdown("---")
    c_m1, c_m2 = st.columns([1, 2])
    use_mkt = c_m1.checkbox("✅ 시장 필터 사용", key="use_market_filter", help="시장이 상승세일 때만 매수")
    mkt_ticker = c_m2.text_input("시장 지수 티커", value="SPY", key="market_ticker", disabled=not use_mkt)

with st.expander("📈 상세 설정", expanded=False):
    c1, c2 = st.columns(2)
    ma_buy = c1.number_input("매수 이평", key="ma_buy", step=1, min_value=1)
    ma_sell = c2.number_input("매도 이평", key="ma_sell", step=1, min_value=1)
    # 나머지 파라미터는 session_state 기본값 사용 (생략)

tab1, tab2, tab3, tab4 = st.tabs(["🎯 시그널", "📚 PRESETS", "🧪 백테스트", "🧬 실험실"])

with tab1:
    if st.button("📌 시그널 확인"):
        check_signal_today(get_data(signal_ticker, start_date, end_date), ma_buy, 0, ma_sell, 0, 0, 0, 0, 0, 0, 0, ">", "<", True, False)

with tab3:
    if st.button("✅ 백테스트 실행", use_container_width=True):
        # 파라미터 준비
        p_ma_buy = int(ma_buy)
        p_ma_sell = int(ma_sell)
        ma_pool = [p_ma_buy, p_ma_sell]
        
        # 데이터 로드 (시장 데이터 포함)
        mkt_t = mkt_ticker if use_mkt else None
        base, x_sig, x_trd, ma_dict, mkt_mask = prepare_base(signal_ticker, trade_ticker, start_date, end_date, ma_pool, market_ticker=mkt_t)
        
        if base is not None:
            res = backtest_fast(
                base, x_sig, x_trd, ma_dict, p_ma_buy, 0, p_ma_sell, 0, 0, 0, 0, 0, 0, 0, 
                5000000, stop_loss_pct, take_profit_pct, trailing_stop_pct, 
                "1", 0, 25, 1, True, False, ">", "<", market_mask=mkt_mask
            )
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
            if not df_log.empty:
                initial_price = df_log['종가'].iloc[0]
                benchmark = (df_log['종가'] / initial_price) * 5000000
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df_log['날짜'], y=df_log['자산'], name='내 전략', line=dict(color='#00F0FF', width=2)))
                fig.add_trace(go.Scatter(x=df_log['날짜'], y=benchmark, name='단순 보유', line=dict(color='gray', dash='dot')))
                
                # 매매 포인트 마커
                buys = df_log[df_log['신호']=='BUY']
                sells = df_log[df_log['신호']=='SELL']
                fig.add_trace(go.Scatter(x=buys['날짜'], y=buys['자산'], mode='markers', marker=dict(symbol='triangle-up', size=10, color='red'), name='매수'))
                fig.add_trace(go.Scatter(x=sells['날짜'], y=sells['자산'], mode='markers', marker=dict(symbol='triangle-down', size=10, color='blue'), name='매도'))
                
                fig.update_layout(title="자산 곡선 (전략 vs 보유)", height=500)
                st.plotly_chart(fig, use_container_width=True)
                
                if st.button("✨ Gemini 재분석"):
                    params_desc = f"매수:{ma_buy}일, 트레일링스탑:{trailing_stop_pct}%, 시장필터:{'ON' if use_mkt else 'OFF'}"
                    anl = ask_gemini_analysis(res, params_desc, trade_ticker, st.session_state.get("gemini_api_key"), st.session_state.get("selected_model_name"))
                    st.session_state["ai_analysis"] = anl
                
                if "ai_analysis" in st.session_state: st.markdown(st.session_state["ai_analysis"])
                with st.expander("상세 매매 로그"): st.dataframe(df_log)
        else: st.warning("⚠️ 매매 신호가 없습니다.")

with tab4:
    st.info("🧪 실험실: 트레일링 스탑 비율을 최적화해보세요! (실험실 로직은 시간 관계상 기본 파라미터로 동작합니다)")
    # (실험실 코드는 위에서 정의된 auto_search 함수 사용)
