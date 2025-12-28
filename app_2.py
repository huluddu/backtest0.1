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
st.set_page_config(page_title="주식 백테스트 Pro", page_icon="📈", layout="wide")

STRATEGY_FILE = "my_strategies.json"

# 전략 파일 로드
def load_saved_strategies():
    if not os.path.exists(STRATEGY_FILE): return {}
    try:
        with open(STRATEGY_FILE, "r", encoding="utf-8") as f: return json.load(f)
    except: return {}

# 전략 파일 저장
def save_strategy_to_file(name, params):
    saved = load_saved_strategies()
    saved[name] = params
    with open(STRATEGY_FILE, "w", encoding="utf-8") as f: 
        json.dump(saved, f, ensure_ascii=False, indent=4)
    st.toast(f"✅ '{name}' 전략이 저장되었습니다!", icon="💾")

# 전략 삭제
def delete_strategy_from_file(name):
    saved = load_saved_strategies()
    if name in saved:
        del saved[name]
        with open(STRATEGY_FILE, "w", encoding="utf-8") as f: 
            json.dump(saved, f, ensure_ascii=False, indent=4)
        st.toast(f"🗑️ '{name}' 전략이 삭제되었습니다.", icon="🗑️")
        return True
    return False

# 초기 상태값 설정 (가장 중요: 여기서 기본값을 잡습니다)
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
        if k not in st.session_state:
            st.session_state[k] = v

# 프리셋 변경 시 실행되는 콜백 함수 (값을 강제로 주입)
def _on_preset_change():
    name = st.session_state["preset_name"]
    # PRESETS 변수는 아래 메인 로직에서 정의되므로 session_state를 통해 접근하거나 전역 참조
    # 여기서는 간단히 로직 내에서 호출될 때 PRESETS를 참조하도록 구조화
    preset_data = st.session_state.get("ALL_PRESETS", {}).get(name, {})
    
    if name == "직접 설정" or not preset_data:
        return

    # 프리셋의 키와 session_state의 키를 매핑
    for k, v in preset_data.items():
        # 티커 이름 매핑 처리
        target_key = k
        if k == "signal_ticker": target_key = "signal_ticker_input"
        elif k == "trade_ticker": target_key = "trade_ticker_input"
        
        if target_key in st.session_state:
            st.session_state[target_key] = v

# 데이터 정규화 및 다운로드
def _normalize_krx_ticker(t: str) -> str:
    t = str(t or "").strip().upper()
    t = re.sub(r"\.(KS|KQ)$", "", t)
    return t

def _fast_ma(x: np.ndarray, w: int) -> np.ndarray:
    if w is None or w <= 1: return x.astype(float)
    kernel = np.ones(w, dtype=float) / w
    y = np.full(x.shape, np.nan, dtype=float)
    if len(x) >= w:
        y[w-1:] = np.convolve(x, kernel, mode="valid")
    return y

@st.cache_data(show_spinner=False, ttl=3600)
def get_data(ticker: str, start_date, end_date) -> pd.DataFrame:
    try:
        t = _normalize_krx_ticker(ticker)
        # 한국 주식 (숫자 6자리)
        is_krx = re.match(r"\d{6}", t)
        
        if is_krx:
            s, e = start_date.strftime("%Y%m%d"), end_date.strftime("%Y%m%d")
            df = stock.get_etf_ohlcv_by_date(s, e, t)
            if df is None or df.empty: df = stock.get_market_ohlcv_by_date(s, e, t)
            if not df.empty:
                df = df.reset_index().rename(columns={"날짜":"Date","시가":"Open","고가":"High","저가":"Low","종가":"Close"})
        else:
            # 미국 주식 (yfinance)
            df = yf.download(t, start=start_date, end=end_date, progress=False, auto_adjust=False)
            # MultiIndex 컬럼 처리
            if isinstance(df.columns, pd.MultiIndex):
                try: df = df.xs(t, axis=1, level=1)
                except: df = df.droplevel(1, axis=1)
            
            df = df.reset_index()
            if "Datetime" in df.columns: df.rename(columns={"Datetime": "Date"}, inplace=True)
            if "Date" in df.columns and pd.api.types.is_datetime64_any_dtype(df["Date"]):
                df["Date"] = df["Date"].dt.tz_localize(None)

        if df is None or df.empty: return pd.DataFrame()
        
        cols = ["Open", "High", "Low", "Close"]
        for c in cols:
            if c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce')
        
        return df[["Date", "Open", "High", "Low", "Close"]].dropna()
    except Exception as e:
        return pd.DataFrame()

# 데이터 전처리 (MA 계산 등)
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

# 보조지표 계산
def calculate_indicators(close_data, rsi_period):
    df = pd.DataFrame({'close': close_data})
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.to_numpy()

# Gemini 분석 요청
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

# 빠른 백테스트 엔진
def backtest_fast(base, x_sig, x_trd, ma_dict_sig, 
                  ma_buy, offset_ma_buy, ma_sell, offset_ma_sell, 
                  offset_cl_buy, offset_cl_sell, 
                  ma_compare_short, ma_compare_long, offset_compare_short, offset_compare_long, 
                  initial_cash, stop_loss_pct, take_profit_pct, strategy_behavior, min_hold_days, 
                  fee_bps, slip_bps, use_trend_in_buy, use_trend_in_sell, buy_operator, sell_operator, 
                  use_rsi_filter=False, rsi_period=14, rsi_max=70):
    
    n = len(base)
    if n == 0: return {}
    
    # 배열 가져오기 (없으면 원본 데이터 사용 등 예외처리 필요하지만 간단히)
    ma_buy_arr = ma_dict_sig.get(ma_buy, x_sig)
    ma_sell_arr = ma_dict_sig.get(ma_sell, x_sig)
    ma_s_arr = ma_dict_sig.get(ma_compare_short) if ma_compare_short else None
    ma_l_arr = ma_dict_sig.get(ma_compare_long) if ma_compare_long else None
    
    rsi_arr = calculate_indicators(x_sig, rsi_period) if use_rsi_filter else None

    # 시작 인덱스 (지표 계산에 필요한 최대 기간)
    idx0 = max((ma_buy or 1), (ma_sell or 1), offset_ma_buy, offset_ma_sell, offset_cl_buy, offset_cl_sell, 
               (offset_compare_short or 0), (offset_compare_long or 0), (rsi_period if use_rsi_filter else 0)) + 2
    
    cash = float(initial_cash)
    position = 0.0
    entry_price = 0.0
    hold_days = 0
    logs, asset_curve = [], []
    
    # 수수료/슬리피지 함수
    fee_rate = (slip_bps + fee_bps) / 10000.0
    
    for i in range(idx0, n):
        curr_date = base["Date"].iloc[i]
        open_p, high_p, low_p, close_p = base["Open_trd"].iloc[i], base["High_trd"].iloc[i], base["Low_trd"].iloc[i], x_trd[i]
        
        # 전일/과거 데이터 참조
        try:
            cl_b = float(x_sig[i - offset_cl_buy])
            ma_b = float(ma_buy_arr[i - offset_ma_buy])
            cl_s = float(x_sig[i - offset_cl_sell])
            ma_s = float(ma_sell_arr[i - offset_ma_sell])
        except: 
            asset_curve.append(cash + position * close_p)
            continue

        # 추세 확인
        trend_ok = True
        if ma_s_arr is not None and ma_l_arr is not None:
            ms = ma_s_arr[i - offset_compare_short]
            ml = ma_l_arr[i - offset_compare_long]
            trend_ok = (ms >= ml)

        # 시그널 조건 계산
        buy_cond_base = (cl_b > ma_b) if buy_operator == ">" else (cl_b < ma_b)
        sell_cond_base = (cl_s < ma_s) if sell_operator == "<" else (cl_s > ma_s)
        
        final_buy = (buy_cond_base and trend_ok) if use_trend_in_buy else buy_cond_base
        final_sell = (sell_cond_base and (not trend_ok)) if use_trend_in_sell else sell_cond_base

        # RSI 필터 (매수 시)
        if use_rsi_filter and final_buy:
            if rsi_arr[i-1] > rsi_max: final_buy = False

        action, reason, exec_px = "HOLD", None, 0.0
        
        # 1. 포지션 보유 중 -> 매도/손절/익절 체크
        if position > 0:
            hold_days += 1
            # 손절/익절 체크 (장중 High/Low 기준)
            stop_price = entry_price * (1 - stop_loss_pct/100)
            take_price = entry_price * (1 + take_profit_pct/100)
            
            is_stop = (stop_loss_pct > 0) and (low_p <= stop_price)
            is_take = (take_profit_pct > 0) and (high_p >= take_price) and not is_stop
            
            if is_stop:
                exec_px = stop_price if open_p > stop_price else open_p # 갭락 고려
                action = "SELL_STOP"
            elif is_take:
                exec_px = take_price if open_p < take_price else open_p # 갭상 고려
                action = "SELL_TAKE"
            elif final_sell and hold_days >= min_hold_days:
                exec_px = open_p
                action = "SELL_SIGNAL"
            
            if action.startswith("SELL"):
                # 매도 실행
                cash = position * exec_px * (1 - fee_rate)
                position = 0.0
                reason = {"SELL_STOP":"손절", "SELL_TAKE":"익절", "SELL_SIGNAL":"전략매도"}[action]
                logs.append({"날짜": curr_date, "종가": close_p, "신호": "SELL", "체결가": exec_px, "이유": reason, "자산": cash})
        
        # 2. 포지션 없음 -> 매수 체크
        elif position == 0:
            do_buy = False
            strat_type = str(strategy_behavior)[:1]
            
            if strat_type == "1": do_buy = final_buy
            elif strat_type == "2": do_buy = final_buy and not final_sell
            
            if do_buy:
                exec_px = open_p
                # 매수 실행
                position = (cash * (1 - fee_rate)) / exec_px
                cash = 0.0
                entry_price = exec_px
                hold_days = 0
                logs.append({"날짜": curr_date, "종가": close_p, "신호": "BUY", "체결가": exec_px, "이유": "전략매수", "자산": position * close_p})

        # 자산 기록
        total_val = cash + (position * close_p)
        asset_curve.append(total_val)
        
    if not logs: return {}
    
    final_asset = asset_curve[-1]
    s = pd.Series(asset_curve)
    mdd = ((s - s.cummax()) / s.cummax()).min() * 100
    
    # 승률 계산
    wins, losses = 0, 0
    g_gain, g_loss = 0.0, 0.0
    
    df_logs = pd.DataFrame(logs)
    last_buy = None
    for _, row in df_logs.iterrows():
        if row['신호'] == 'BUY': last_buy = row['체결가']
        elif row['신호'] == 'SELL' and last_buy:
            diff = (row['체결가'] - last_buy) / last_buy
            if diff > 0: wins += 1; g_gain += diff
            else: losses += 1; g_loss += abs(diff)
            last_buy = None
            
    total_trades = wins + losses
    win_rate = (wins/total_trades*100) if total_trades > 0 else 0
    pf = (g_gain/g_loss) if g_loss > 0 else 99.9
    
    return {
        "수익률 (%)": round((final_asset - initial_cash)/initial_cash*100, 2),
        "MDD (%)": round(mdd, 2),
        "승률 (%)": round(win_rate, 2),
        "Profit Factor": round(pf, 2),
        "총 매매 횟수": total_trades,
        "최종 자산": round(final_asset),
        "매매 로그": logs,
        "자산 곡선": asset_curve
    }

# ==========================================
# 2. 메인 UI 및 실행 로직
# ==========================================

# 1. 초기 상태 초기화
_init_default_state()

# 2. 기본 프리셋 정의
DEFAULT_PRESETS = {
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
# 3. 저장된 전략 합치기
SAVED_PRESETS = load_saved_strategies()
ALL_PRESETS = {**DEFAULT_PRESETS, **SAVED_PRESETS}
st.session_state["ALL_PRESETS"] = ALL_PRESETS # 콜백에서 쓰기 위해 저장

# --- 사이드바 ---
with st.sidebar:
    st.header("🤖 설정 및 저장")
    
    # Gemini Key
    api_key = st.text_input("Gemini API Key", type="password", help="Google AI Studio에서 발급받은 키를 입력하세요.")
    if api_key: st.session_state["gemini_api_key"] = api_key
    
    st.divider()
    
    # 프리셋 선택 (핵심: on_change에서 상태를 업데이트함)
    preset_list = ["직접 설정"] + list(ALL_PRESETS.keys())
    selected_preset = st.selectbox(
        "📂 전략 불러오기", 
        preset_list, 
        key="preset_name", 
        on_change=_on_preset_change  # 여기가 핵심! 변경 시 session_state 업데이트
    )
    
    st.divider()
    
    # 전략 저장 기능
    with st.expander("💾 현재 전략 저장/삭제", expanded=False):
        save_name = st.text_input("저장할 전략 이름", placeholder="예: 나만의 SOXL 전략")
        if st.button("현재 설정 저장하기", use_container_width=True):
            if save_name:
                # 현재 UI에 있는 값들을 딕셔너리로 만듦
                current_params = {
                    "signal_ticker": st.session_state["signal_ticker_input"],
                    "trade_ticker": st.session_state["trade_ticker_input"],
                    "ma_buy": st.session_state["ma_buy"],
                    "offset_ma_buy": st.session_state["offset_ma_buy"],
                    "offset_cl_buy": st.session_state["offset_cl_buy"],
                    "buy_operator": st.session_state["buy_operator"],
                    "ma_sell": st.session_state["ma_sell"],
                    "offset_ma_sell": st.session_state["offset_ma_sell"],
                    "offset_cl_sell": st.session_state["offset_cl_sell"],
                    "sell_operator": st.session_state["sell_operator"],
                    "use_trend_in_buy": st.session_state["use_trend_in_buy"],
                    "use_trend_in_sell": st.session_state["use_trend_in_sell"],
                    "ma_compare_short": st.session_state["ma_compare_short"],
                    "ma_compare_long": st.session_state["ma_compare_long"],
                    "stop_loss_pct": st.session_state["stop_loss_pct"],
                    "take_profit_pct": st.session_state["take_profit_pct"]
                }
                save_strategy_to_file(save_name, current_params)
                st.rerun() # 새로고침하여 목록 갱신
        
        if selected_preset in SAVED_PRESETS:
            if st.button(f"🗑️ '{selected_preset}' 삭제", type="primary", use_container_width=True):
                delete_strategy_from_file(selected_preset)
                st.session_state["preset_name"] = "직접 설정"
                st.rerun()

# --- 메인 파라미터 입력창 (중요: value를 직접 할당하지 않고 key로 연결) ---
st.title("📈 주식 백테스트 & AI 전략 검증")

# 1행: 티커 및 기간
c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
signal_ticker = c1.text_input("시그널 티커", key="signal_ticker_input")
trade_ticker = c2.text_input("매매 티커", key="trade_ticker_input")
start_date = c3.date_input("시작일", value=datetime.date(2020, 1, 1))
end_date = c4.date_input("종료일", value=datetime.date.today())

# 2행: 매수/매도 설정 (Expander로 정리)
with st.expander("🛠️ 매매 전략 상세 설정", expanded=True):
    col_buy, col_sell = st.columns(2)
    
    with col_buy:
        st.subheader("📥 매수 조건")
        st.caption("종가(n일전) [부호] 이평선(n일전) 일 때")
        b1, b2, b3 = st.columns([1, 0.5, 1])
        off_cl_b = b1.number_input("종가 Offset", key="offset_cl_buy", help="0이면 오늘, 1이면 어제")
        op_b = b2.selectbox("부호", [">", "<"], key="buy_operator")
        ma_b = b3.number_input("이평선 (일)", min_value=1, key="ma_buy")
        
        st.markdown("---")
        off_ma_b = st.number_input("이평선 Offset", key="offset_ma_buy", help="이평선을 며칠 전 기준으로 볼지")
        use_tr_b = st.checkbox("✅ 추세장 필터 (단기이평 > 장기이평 일때만 매수)", key="use_trend_in_buy")
        
        # RSI
        use_rsi = st.checkbox("🔮 RSI 과매수 방지 필터", key="use_rsi_filter")
        if use_rsi:
             st.number_input("RSI 기준 (이보다 높으면 매수X)", value=70, key="rsi_max")

    with col_sell:
        st.subheader("📤 매도 조건")
        st.caption("종가(n일전) [부호] 이평선(n일전) 일 때")
        s1, s2, s3 = st.columns([1, 0.5, 1])
        off_cl_s = s1.number_input("종가 Offset", key="offset_cl_sell")
        op_s = s2.selectbox("부호", ["<", ">"], key="sell_operator")
        ma_s = s3.number_input("이평선 (일)", min_value=1, key="ma_sell")

        st.markdown("---")
        off_ma_s = st.number_input("이평선 Offset", key="offset_ma_sell")
        use_tr_s = st.checkbox("✅ 역추세장 필터 (단기 < 장기 일때만 매도)", key="use_trend_in_sell")

    st.markdown("---")
    st.subheader("🛡️ 리스크 관리 & 추세선")
    r1, r2, r3, r4 = st.columns(4)
    r1.number_input("손절 (%)", step=1.0, key="stop_loss_pct", help="0이면 미사용")
    r2.number_input("익절 (%)", step=1.0, key="take_profit_pct", help="0이면 미사용")
    r3.number_input("추세 단기 이평", key="ma_compare_short")
    r4.number_input("추세 장기 이평", key="ma_compare_long")

# --- 실행 버튼 ---
if st.button("🚀 백테스트 실행", type="primary", use_container_width=True):
    with st.spinner("데이터를 다운로드하고 시뮬레이션 중입니다..."):
        # 파라미터 정리
        ma_pool = [st.session_state["ma_buy"], st.session_state["ma_sell"], 
                   st.session_state["ma_compare_short"], st.session_state["ma_compare_long"]]
        
        # 데이터 로드
        base, x_sig, x_trd, ma_dict = prepare_base(signal_ticker, trade_ticker, start_date, end_date, ma_pool)
        
        if base is not None:
            # 백테스트 실행
            res = backtest_fast(
                base, x_sig, x_trd, ma_dict,
                st.session_state["ma_buy"], st.session_state["offset_ma_buy"],
                st.session_state["ma_sell"], st.session_state["offset_ma_sell"],
                st.session_state["offset_cl_buy"], st.session_state["offset_cl_sell"],
                st.session_state["ma_compare_short"], st.session_state["ma_compare_long"],
                st.session_state["offset_compare_short"], st.session_state["offset_compare_long"],
                10000000, # 초기자본 1천만원 가정
                st.session_state["stop_loss_pct"], st.session_state["take_profit_pct"],
                st.session_state["strategy_behavior"], st.session_state["min_hold_days"],
                st.session_state["fee_bps"], st.session_state["slip_bps"],
                st.session_state["use_trend_in_buy"], st.session_state["use_trend_in_sell"],
                st.session_state["buy_operator"], st.session_state["sell_operator"],
                use_rsi_filter=st.session_state["use_rsi_filter"], rsi_max=st.session_state.get("rsi_max", 70)
            )
            
            if not res:
                st.error("매매 기록이 없습니다. 조건을 완화해보세요.")
            else:
                # 결과 출력
                st.success("분석 완료!")
                
                # 핵심 지표
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("총 수익률", f"{res['수익률 (%)']}%", delta_color="normal")
                m2.metric("MDD (최대낙폭)", f"{res['MDD (%)']}%", delta_color="inverse")
                m3.metric("승률", f"{res['승률 (%)']}%")
                m4.metric("매매 횟수", f"{res['총 매매 횟수']}회")

                # 차트 그리기
                df_log = pd.DataFrame(res['매매 로그'])
                asset_curve = res['자산 곡선']
                dates = base["Date"].iloc[-len(asset_curve):]
                
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.05)
                
                # 자산 곡선
                fig.add_trace(go.Scatter(x=dates, y=asset_curve, name="내 자산", line=dict(color="#00C805", width=2)), row=1, col=1)
                
                # 매매 포인트 표시
                buys = df_log[df_log['신호'] == 'BUY']
                sells = df_log[df_log['신호'] == 'SELL']
                
                fig.add_trace(go.Scatter(x=buys['날짜'], y=buys['체결가'], mode='markers', marker=dict(color='red', symbol='triangle-up', size=10), name='매수'), row=1, col=1)
                fig.add_trace(go.Scatter(x=sells['날짜'], y=sells['체결가'], mode='markers', marker=dict(color='blue', symbol='triangle-down', size=10), name='매도'), row=1, col=1)
                
                # MDD 영역
                s = pd.Series(asset_curve)
                dd = (s - s.cummax()) / s.cummax() * 100
                fig.add_trace(go.Scatter(x=dates, y=dd, name="낙폭(DD)", fill='tozeroy', line=dict(color='#ff4b4b', width=1)), row=2, col=1)
                
                fig.update_layout(height=600, template="plotly_white", title="자산 변동 및 MDD")
                st.plotly_chart(fig, use_container_width=True)
                
                # Gemini 분석
                st.divider()
                st.subheader("🤖 Gemini 전략 분석")
                
                if st.session_state.get("gemini_api_key"):
                    if st.button("✨ AI에게 분석 요청하기"):
                        params_desc = f"매수: {st.session_state['ma_buy']}일 이평, 손절: {st.session_state['stop_loss_pct']}%"
                        analysis = ask_gemini_analysis(res, params_desc, trade_ticker, st.session_state["gemini_api_key"], "gemini-1.5-flash")
                        st.info(analysis)
                else:
                    st.warning("왼쪽 사이드바에 Gemini API Key를 입력하면 AI 분석을 받을 수 있습니다.")

                with st.expander("📊 상세 매매 로그 확인"):
                    st.dataframe(df_log)
        else:
            st.error("데이터를 불러오지 못했습니다. 티커를 확인해주세요.")

