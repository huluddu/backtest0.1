# === PATCHED BY GPT-5 Thinking (2025-10-16, Asia/Seoul) ===
# Applied fixes:
# 1) Today-index bug: i=0 -> i=len(df)-1 inside check_signal_today / summarize_signal_today.
# 2) Trend comparator: ms > ml -> ms >= ml within backtest_fast.
# 3) Hold-days guard: prevent increment on buy day using `just_bought` flag; heuristic insertion.
# 4) Removed duplicate trailing return blocks in backtest_fast (best-effort).
# Notes:
# - If you rely on execution_lag_days or realtime param passthrough, consider manual review near those sections.
# - Please run a quick sanity backtest on a small date range to verify expected behavior.
# =========================================================

import streamlit as st

# === Streamlit preset/session_state sync helpers (injected) ===
import streamlit as st

def _init_default_state():
    defaults = {
        "signal_ticker_input": "SOXL",
        "trade_ticker_input": "SOXL",
        "buy_operator": ">",
        "sell_operator": "<",
        "strategy_behavior": "1. 포지션 없으면 매수 / 보유 중이면 매도",
        "offset_cl_buy": 0, "offset_cl_sell": 0,
        "offset_ma_buy": 0, "offset_ma_sell": 0,
        "ma_buy": 50, "ma_sell": 10,
        "use_trend_in_buy": True, "use_trend_in_sell": False,
        "ma_compare_short": 20, "ma_compare_long": 50,
        "offset_compare_short": 0, "offset_compare_long": 0,
        "stop_loss_pct": 0.0, "take_profit_pct": 0.0, "min_hold_days": 0,
        "preset_name": "직접 설정",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

def _apply_preset_to_state(preset: dict | None):
    if not preset:
        return
    for k, v in preset.items():
        st.session_state[k if not k.endswith("_ticker") else (k.replace("_ticker", "_ticker_input"))] = v

def _on_preset_change(PRESETS: dict):
    name = st.session_state.get("preset_name", "직접 설정")
    preset = {} if name == "직접 설정" else PRESETS.get(name, {})
    _apply_preset_to_state(preset)
    st.rerun()
# === end injected helpers ===

import pandas as pd
import yfinance as yf
import datetime
import plotly.graph_objects as go
import random
from pykrx import stock
from functools import lru_cache
import numpy as np
import re

# ============== Page Setup & Header (UI only) ==============
st.set_page_config(page_title="시그널 대시보드", page_icon="📊", layout="wide")
colA, colB, colC, colD = st.columns([1.5,1,1,1])
with colA:
    st.markdown("## 📊 오늘 시그널 대시보드")
with colB:
    if st.button("🔄 캐시 초기화", use_container_width=True):
        st.cache_data.clear()
        st.toast("캐시 초기화 완료!", icon="✅")
with colC:
    pass  # 필요하면 토글/디버그 버튼 배치
with colD:
    st.caption(f"기준일: **{pd.Timestamp.today().strftime('%Y-%m-%d')}**")
st.divider()


def _normalize_krx_ticker(t: str) -> str:
    """'069500.KS' -> '069500', '371460' -> '371460'"""
    if not isinstance(t, str):
        t = str(t or "")
    t = t.strip().upper()
    t = re.sub(r"\.(KS|KQ)$", "", t)  # 접미사 제거
    m = re.search(r"(\d{6})", t)
    return m.group(1) if m else ""

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

##########################


@st.cache_data(show_spinner=False, ttl=3600)
def get_krx_data_cached(ticker: str, start_date, end_date):
    """KRX(숫자 6자리)용: OHLC 로딩 (ETF/일반 모두 커버, 빈DF 가드 포함)"""
    code = _normalize_krx_ticker(ticker)
    if not code:
        return pd.DataFrame(columns=["Date", "Open", "High", "Low", "Close"])

    s = start_date.strftime("%Y%m%d")
    e = end_date.strftime("%Y%m%d")

    df = pd.DataFrame()
    try:
        # 1) ETF 먼저 시도
        df = stock.get_etf_ohlcv_by_date(s, e, code)
        if df is None or df.empty:
            # 2) 일반 종목(주식/ETF 모두 커버)로 재시도
            df = stock.get_market_ohlcv_by_date(s, e, code)
    except Exception:
        df = pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame(columns=["Date", "Open", "High", "Low", "Close"])

    df = (
        df.reset_index()
          .rename(columns={"날짜": "Date", "시가": "Open", "고가": "High",
                           "저가": "Low", "종가": "Close"})
          .loc[:, ["Date", "Open", "High", "Low", "Close"]]
          .dropna()
    )
    return df

@st.cache_data(show_spinner=False, ttl=3600)
def get_yf_data_cached(ticker: str, start_date, end_date):
    """야후파이낸스용: OHLC 로딩 (빈DF/멀티컬럼 가드)"""
    df = yf.download(ticker, start=start_date, end=end_date, progress=False)

    # ✅ 빈 DF·컬럼 누락 가드
    if df is None or df.empty or (isinstance(df, pd.DataFrame) and len(df.columns) == 0):
        return pd.DataFrame(columns=["Date", "Open", "High", "Low", "Close"])

    if isinstance(df.columns, pd.MultiIndex):
        tu = str(ticker).upper()
        try:
            o = df[("Open",  tu)]
            h = df[("High",  tu)]
            l = df[("Low",   tu)]
            c = df[("Close", tu)]
            df = pd.concat([o, h, l, c], axis=1)
            df.columns = ["Open", "High", "Low", "Close"]
        except Exception:
            df = df.droplevel(1, axis=1)
            # 일부 케이스에 특정 컬럼이 빠질 수 있어, 부족하면 안전 반환
            if not {"Open","High","Low","Close"}.issubset(df.columns):
                return pd.DataFrame(columns=["Date", "Open", "High", "Low", "Close"])
            df = df[["Open", "High", "Low", "Close"]]
    else:
        # 단일 컬럼 구조에서도 누락 시 안전 반환
        if not {"Open","High","Low","Close"}.issubset(df.columns):
            return pd.DataFrame(columns=["Date", "Open", "High", "Low", "Close"])
        df = df[["Open", "High", "Low", "Close"]]

    df = df.reset_index()
    # 인덱스명이 Datetime/Date 등 다양해서 일괄 표준화
    if "Date" not in df.columns and "Datetime" in df.columns:
        df.rename(columns={"Datetime": "Date"}, inplace=True)

    # 최종 표준 스키마 보장
    if not {"Date","Open","High","Low","Close"}.issubset(df.columns):
        return pd.DataFrame(columns=["Date", "Open", "High", "Low", "Close"])

    df = df[["Date", "Open", "High", "Low", "Close"]].dropna()
    return df


def get_data(ticker: str, start_date, end_date) -> pd.DataFrame:
    """티커 타입에 따라 KRX/yf 로더 분기 ('.KS' '.KQ' 포함, 실패시 표준 빈 스키마)"""
    try:
        t = (ticker or "").strip()
        is_krx_like = t.isdigit() or t.lower().endswith(".ks") or t.lower().endswith(".kq")
        df = get_krx_data_cached(t, start_date, end_date) if is_krx_like \
             else get_yf_data_cached(t, start_date, end_date)
        if df is None or df.empty or not {"Date","Open","High","Low","Close"}.issubset(df.columns):
            return pd.DataFrame(columns=["Date", "Open", "High", "Low", "Close"])
        return df
    except Exception as e:
        st.error(f"❌ 데이터 로딩 실패: {e}")
        return pd.DataFrame(columns=["Date", "Open", "High", "Low", "Close"])



@st.cache_data(show_spinner=False, ttl=30)
def fetch_yf_near_realtime_close(ticker: str):
    """
    yfinance 1분봉의 가장 최근 종가를 반환.
    - 보통 수 분 지연(거래소 사정)일 수 있음
    - ttl=30초 캐시로 과호출 방지
    """
    try:
        df = yf.download(
            tickers=ticker,
            period="2d",          # 전일 포함해 공백 캔들 방지
            interval="1m",
            auto_adjust=False,
            progress=False,
        )
        if df.empty:
            return None
        # 멀티인덱스 방어
        if isinstance(df.columns, pd.MultiIndex):
            df = df.droplevel(1, axis=1)
        last = df.dropna().iloc[-1]
        return {
            "price": float(last["Close"]),
            "timestamp": last.name.to_pydatetime(),
            "source": "yfinance_1m"
        }
    except Exception:
        return None


def get_spot_price_fast_us_only(ticker: str):
    """
    미주 전용: yfinance 1분봉만 사용.
    """
    out = fetch_yf_near_realtime_close(ticker)
    return out  # 없으면 None



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

###### 1min yfinance 유틸 함수 추가 #########
@st.cache_data(show_spinner=False, ttl=30)
def get_yf_1m_grouped_close(ticker: str, tz: str, session_start: str, session_end: str):
    """
    yfinance 1분봉을 가져와 거래소 타임존(tz)으로 변환 후,
    세션(날짜)별 마지막 Close를 집계해 반환.
    Returns:
      - daily_close: pd.Series(index=date, value=close)
      - last_price: float or None
      - last_ts: pd.Timestamp or None (타임존 포함)
    """
    try:
        df = yf.download(
            tickers=ticker,
            period="5d",          # 최근 며칠만
            interval="1m",
            auto_adjust=False,
            progress=False,
        )
        if df.empty:
            return pd.Series(dtype=float), None, None

        if isinstance(df.columns, pd.MultiIndex):
            df = df.droplevel(1, axis=1)

        # yfinance 분봉 인덱스가 tz-naive인 경우가 많음 → UTC로 가정 후 변환
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        df = df.tz_convert(tz)

        # 정규장만 남김
        df = df.between_time(session_start, session_end).copy()
        if df.empty:
            return pd.Series(dtype=float), None, None

        df["session"] = df.index.date
        daily_close = df.groupby("session")["Close"].last()

        last_row = df.iloc[-1]
        last_price = float(last_row["Close"])
        last_ts = last_row.name  # tz-aware

        return daily_close, last_price, last_ts
    except Exception:
        return pd.Series(dtype=float), None, None


### 오늘의 시그널 (일반) ####
def check_signal_today(
    df,
    ma_buy, offset_ma_buy, ma_sell, offset_ma_sell,
    offset_cl_buy, offset_cl_sell,
    ma_compare_short=None, ma_compare_long=None,
    offset_compare_short=1, offset_compare_long=1,
    buy_operator=">", sell_operator="<",
    use_trend_in_buy=True, use_trend_in_sell=False,
    *,                       # ← 키워드 전용
    strategy_behavior="1. 포지션 없으면 매수 / 보유 중이면 매도",
    min_hold_days=0
    ):

    df = df.copy()
    df = df.sort_values("Date").reset_index(drop=True)
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df["MA_BUY"] = df["Close"].rolling(ma_buy).mean()
    df["MA_SELL"] = df["Close"].rolling(ma_sell).mean()

    if ma_compare_short and ma_compare_long:
        df["MA_SHORT"] = df["Close"].rolling(ma_compare_short).mean()
        df["MA_LONG"] = df["Close"].rolling(ma_compare_long).mean()

    i = 0
    try:
        cl_b = float(df["Close"].iloc[i - offset_cl_buy])
        ma_b = float(df["MA_BUY"].iloc[i - offset_ma_buy])
        cl_s = float(df["Close"].iloc[i - offset_cl_sell])
        ma_s = float(df["MA_SELL"].iloc[i - offset_ma_sell])
        ref_date = df["Date"].iloc[-1].strftime('%Y-%m-%d')
    except Exception as e:
        st.warning(f"❗오늘 시그널 판단에 필요한 데이터가 부족합니다: {e}")
        return

    st.subheader("📌 오늘 시그널 판단")
    st.write(f"📆 기준일 (데이터가 있는 마지막 날): {ref_date}")

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

    # ── 최근 조건 만족일 찾기: BUY / SELL / HOLD(둘 다 불만족) ──
    last_buy_date  = None
    last_sell_date = None
    last_hold_date = None

    # 뒤에서 앞으로 훑되, 오프셋 안정 구간부터
    safe_start = max(offset_cl_buy, offset_ma_buy, offset_cl_sell, offset_ma_sell,
                     (offset_compare_short or 0), (offset_compare_long or 0))
    for j in range(len(df)-1, safe_start-1, -1):
        try:
            cb = df["Close"].iloc[j - offset_cl_buy]
            mb = df["MA_BUY"].iloc[j - offset_ma_buy]
            cs = df["Close"].iloc[j - offset_cl_sell]
            ms = df["MA_SELL"].iloc[j - offset_ma_sell]

            # 추세필터(있는 경우만)
            trend_pass = True
            if (ma_compare_short and ma_compare_long
                and "MA_SHORT" in df.columns and "MA_LONG" in df.columns):
                ms_short = df["MA_SHORT"].iloc[j - offset_compare_short]
                ms_long  = df["MA_LONG"].iloc[j - offset_compare_long]
                trend_pass = (ms_short >= ms_long)

            _buy_base  = (cb > mb) if (buy_operator == ">") else (cb < mb)
            _sell_base = (cs < ms) if (sell_operator == "<") else (cs > ms)
            _buy_ok    = (_buy_base  and trend_pass)       if use_trend_in_buy  else _buy_base
            _sell_ok   = (_sell_base and (not trend_pass)) if use_trend_in_sell else _sell_base

            if last_sell_date is None and _sell_ok:
                last_sell_date = df["Date"].iloc[j]
            if last_buy_date is None and _buy_ok:
                last_buy_date = df["Date"].iloc[j]
            if last_hold_date is None and (not _buy_ok and not _sell_ok):
                last_hold_date = df["Date"].iloc[j]

            if last_sell_date and last_hold_date and last_buy_date:
                break
        except Exception:
            continue

    # 표시
    if last_buy_date:
        st.write(f"🗓 최근 **BUY 조건** 만족: {pd.to_datetime(last_buy_date).strftime('%Y-%m-%d')}")
    if last_sell_date:
        st.write(f"🗓 최근 **SELL 조건** 만족: {pd.to_datetime(last_sell_date).strftime('%Y-%m-%d')}")
    if last_hold_date:
        st.write(f"🗓 최근 **HOLD(양쪽 불만족)**: {pd.to_datetime(last_hold_date).strftime('%Y-%m-%d')}")
    if not (last_buy_date or last_sell_date or last_hold_date):
        st.warning("❗최근 조건에 부합하는 날을 찾지 못했습니다.")

    ### 오늘의 시그널 (실시간) ###

def check_signal_today_realtime(
    df_daily: pd.DataFrame,
    ticker: str,
    *,
    # 아래부터는 키워드 인자 
    tz: str = "America/New_York",
    session_start: str = "09:30",
    session_end: str = "16:00",
    ma_buy, offset_ma_buy, ma_sell, offset_ma_sell,
    offset_cl_buy, offset_cl_sell,
    ma_compare_short=None, ma_compare_long=None,
    offset_compare_short=1, offset_compare_long=1,
    buy_operator=">", sell_operator="<",
    use_trend_in_buy=True, use_trend_in_sell=False
):
    """
    일봉 df_daily를 기본으로 사용하되,
    '오늘'만 yfinance 1분봉의 최신가로 Close를 덮어쓴 뒤,
    원래 전달받은 오프셋들을 그대로 유지하여 check_signal_today 실행.
    """
    # 1) 1분봉 세션 집계 (최근 며칠 날짜별 마지막 Close) + 최신 바 시각
    daily_close_1m, last_price, last_ts = get_yf_1m_grouped_close(
        ticker, tz=tz, session_start=session_start, session_end=session_end
    )

    # 2) 작업용 일봉 복사
    df_rt = df_daily.copy().sort_values("Date").reset_index(drop=True)
    df_rt["Date"] = pd.to_datetime(df_rt["Date"])

    # 3) 오늘 세션 날짜 계산 (tz-aware 안전 처리)
    today_sess_date = None
    if last_ts is not None:
        ts = pd.Timestamp(last_ts)
        if ts.tz is None:
            ts = ts.tz_localize("UTC").tz_convert(tz)
        else:
            ts = ts.tz_convert(tz)
        today_sess_date = ts.date()

    # 4) 오늘만 분봉 최신가로 패치 (전일/과거일은 EOD 그대로 둠)
    patched = False
    if (daily_close_1m is not None) and (not daily_close_1m.empty) and (today_sess_date is not None):
        # 오늘 세션의 종가(=현재까지의 마지막 1분봉 close)
        today_close_1m = daily_close_1m.get(today_sess_date, None)

        df_rt["__date"] = df_rt["Date"].dt.date
        if today_close_1m is not None:
            if (df_rt["__date"] == today_sess_date).any():
                # 오늘 행이 이미 있으면 Close만 교체
                df_rt.loc[df_rt["__date"] == today_sess_date, "Close"] = float(today_close_1m)
                patched = True
            else:
                # 오늘 행이 없으면 오늘 행 추가 (OHLC가 없을 수 있으니 Close만 필수로 추가)
                df_rt = pd.concat([df_rt, pd.DataFrame([{
                    "Date": pd.Timestamp(today_sess_date),
                    "Close": float(today_close_1m),
                }])], ignore_index=True)
                df_rt = df_rt.sort_values("Date").reset_index(drop=True)
                patched = True

        df_rt = df_rt.drop(columns=["__date"], errors="ignore")

    # 5) 오프셋은 '유지'하여 기존 일봉 판정 함수 호출

    check_signal_today(
        df_rt,
        ma_buy=ma_buy, offset_ma_buy=offset_ma_buy,
        ma_sell=ma_sell, offset_ma_sell=offset_ma_sell,
        offset_cl_buy=offset_cl_buy, offset_cl_sell=offset_cl_sell,
        ma_compare_short=ma_compare_short if (ma_compare_short or 0) > 0 else None,
        ma_compare_long=ma_compare_long  if (ma_compare_long  or 0) > 0 else None,
        offset_compare_short=offset_compare_short, offset_compare_long=offset_compare_long,
        buy_operator=buy_operator, sell_operator=sell_operator,
        use_trend_in_buy=use_trend_in_buy, use_trend_in_sell=use_trend_in_sell
    )

    # 6) 디버그용 (선택)
    with st.expander("🐞 실시간 패치 디버그", expanded=False):
        st.write({
            "ticker": ticker,
            "patched_today": patched,
            "today_sess_date": str(today_sess_date) if today_sess_date else None,
            "last_price": float(last_price) if last_price is not None else None,
            "last_ts": str(last_ts) if last_ts is not None else None,
        })
        if (daily_close_1m is not None) and (not daily_close_1m.empty):
            st.write("최근 세션별 종가(1분봉 집계) tail(5):", daily_close_1m.tail(5))
        st.write("패치 후 일봉 tail(5):", df_rt.tail(5))



    
##################################


# ✅ 전략 프리셋 목록 정의
PRESETS = {

    "SOXL 도전 전략": {
        "signal_ticker": "SOXL", "trade_ticker": "SOXL",
        "offset_cl_buy": 1, "buy_operator": ">", "offset_ma_buy": 1, "ma_buy": 20,
        "offset_cl_sell": 1, "sell_operator": ">", "offset_ma_sell": 20, "ma_sell": 10,
        "use_trend_in_buy": True, "use_trend_in_sell": True,
        "offset_compare_short": 10, "ma_compare_short": 5,
        "offset_compare_long": 20, "ma_compare_long": 5,
        "stop_loss_pct": 0.0, "take_profit_pct": 0.0
    },
  
    "SOXL 안전 전략": {
        "signal_ticker": "SOXL", "trade_ticker": "SOXL",
        "offset_cl_buy": 20, "buy_operator": ">", "offset_ma_buy": 50, "ma_buy": 10,
        "offset_cl_sell": 50, "sell_operator": ">", "offset_ma_sell": 1, "ma_sell": 10,
        "use_trend_in_buy": True, "use_trend_in_sell": True,
        "offset_compare_short": 20, "ma_compare_short": 10,
        "offset_compare_long": 20, "ma_compare_long": 1,
        "stop_loss_pct": 35.0, "take_profit_pct": 15.0
    },
  
    "TSLL 안전 전략": {
        "signal_ticker": "TSLL", "trade_ticker": "TSLL",
        "offset_cl_buy": 20, "buy_operator": "<", "offset_ma_buy": 50, "ma_buy": 20,
        "offset_cl_sell": 50, "sell_operator": ">", "offset_ma_sell": 20, "ma_sell": 10,
        "use_trend_in_buy": True, "use_trend_in_sell": True,
        "offset_compare_short": 1, "ma_compare_short": 10,
        "offset_compare_long": 50, "ma_compare_long": 50,         
        "stop_loss_pct": 25.0, "take_profit_pct": 15.0
    },
    
    "TSLL 도전 전략": {
        "signal_ticker": "TSLL", "trade_ticker": "TSLL",
        "offset_cl_buy": 50, "buy_operator": ">", "offset_ma_buy": 50, "ma_buy": 10,
        "offset_cl_sell": 10, "sell_operator": "<", "offset_ma_sell": 10, "ma_sell": 20,
        "use_trend_in_buy": True, "use_trend_in_sell": True,
        "offset_compare_short": 50, "ma_compare_short": 50,
        "offset_compare_long": 50, "ma_compare_long": 5,         
        "stop_loss_pct": 0.0, "take_profit_pct": 0.0
    },

    "GGLL 전략": {
        "signal_ticker": "GGLL", "trade_ticker": "GGLL",
        "offset_cl_buy": 1, "buy_operator": "<", "offset_ma_buy": 1, "ma_buy": 20,
        "offset_cl_sell": 20, "sell_operator": "<", "offset_ma_sell": 20, "ma_sell": 50,
        "use_trend_in_buy": True, "use_trend_in_sell": True,
        "offset_compare_short": 20, "ma_compare_short": 1,
        "offset_compare_long": 50, "ma_compare_long": 1,         
        "stop_loss_pct": 15.0, "take_profit_pct": 0.0
    },

    "GGLL 안전 전략": {
        "signal_ticker": "GGLL", "trade_ticker": "GGLL",
        "offset_cl_buy": 10, "buy_operator": ">", "offset_ma_buy": 50, "ma_buy": 5,
        "offset_cl_sell": 10, "sell_operator": "<", "offset_ma_sell": 20, "ma_sell": 20,
        "use_trend_in_buy": True, "use_trend_in_sell": True,
        "offset_compare_short": 10, "ma_compare_short": 20,
        "offset_compare_long": 50, "ma_compare_long": 10,         
        "stop_loss_pct": 20.0, "take_profit_pct": 20.0
    },
    
    "GGLL 안전 전략2": {
        "signal_ticker": "GGLL", "trade_ticker": "GGLL",
        "offset_cl_buy": 50, "buy_operator": "<", "offset_ma_buy": 20, "ma_buy": 1,
        "offset_cl_sell": 20, "sell_operator": "<", "offset_ma_sell": 1, "ma_sell": 1,
        "use_trend_in_buy": True, "use_trend_in_sell": True,
        "offset_compare_short": 1, "ma_compare_short": 10,
        "offset_compare_long": 50, "ma_compare_long": 10,         
        "stop_loss_pct": 20.0, "take_profit_pct": 30.0
    },    
    
    "BITX 전략": {
        "signal_ticker": "BITX", "trade_ticker": "BITX",
        "offset_cl_buy": 16, "buy_operator": ">", "offset_ma_buy": 26, "ma_buy": 5,
        "offset_cl_sell": 26, "sell_operator": ">", "offset_ma_sell": 2, "ma_sell": 15,
        "use_trend_in_buy": True, "use_trend_in_sell": True,
        "offset_compare_short": 26, "ma_compare_short": 15,
        "offset_compare_long": 6, "ma_compare_long": 15,         
        "stop_loss_pct": 30.0, "take_profit_pct": 0.0
    },

    "TQQQ 도전 전략": {
        "signal_ticker": "TQQQ", "trade_ticker": "TQQQ",
        "offset_cl_buy": 50, "buy_operator": ">", "offset_ma_buy": 10, "ma_buy": 1,
        "offset_cl_sell": 50, "sell_operator": ">", "offset_ma_sell": 1, "ma_sell": 1, 
        "use_trend_in_buy": True, "use_trend_in_sell": True,
        "offset_compare_short": 1, "ma_compare_short": 50,
        "offset_compare_long": 10, "ma_compare_long": 1,
        "stop_loss_pct": 15.0, "take_profit_pct": 25.0
    },

    "TQQQ 안전 전략": {
        "signal_ticker": "TQQQ", "trade_ticker": "TQQQ",
        "offset_cl_buy": 10, "buy_operator": "<", "offset_ma_buy": 50, "ma_buy": 20,
        "offset_cl_sell": 50, "sell_operator": ">", "offset_ma_sell": 10, "ma_sell": 20, 
        "use_trend_in_buy": True, "use_trend_in_sell": True,
        "offset_compare_short": 10, "ma_compare_short": 50,
        "offset_compare_long": 20, "ma_compare_long": 20,
        "stop_loss_pct": 25.0, "take_profit_pct": 25.0
    },

    "BITX-TQQQ 안전": {
        "signal_ticker": "BITX", "trade_ticker": "TQQQ",
        "offset_cl_buy": 10, "buy_operator": ">", "offset_ma_buy": 10, "ma_buy": 20,
        "offset_cl_sell": 50, "sell_operator": ">", "offset_ma_sell": 1, "ma_sell": 5,
        "use_trend_in_buy": False, "use_trend_in_sell": True,
        "offset_compare_short": 50, "ma_compare_short": 5,
        "offset_compare_long": 1, "ma_compare_long": 50,
        "stop_loss_pct": 0.0, "take_profit_pct": 15.0
    },

    "BITX-TQQQ 도전": {
        "signal_ticker": "BITX", "trade_ticker": "TQQQ",
        "offset_cl_buy": 50, "buy_operator": "<", "offset_ma_buy": 20, "ma_buy": 50,
        "offset_cl_sell": 50, "sell_operator": ">", "offset_ma_sell": 10, "ma_sell": 5,
        "use_trend_in_buy": True, "use_trend_in_sell": True,
        "offset_compare_short": 10, "ma_compare_short": 1,
        "offset_compare_long": 20, "ma_compare_long": 1,
        "stop_loss_pct": 25.0, "take_profit_pct": 25.0
    },

    "BTC-TQQQ": {
        "signal_ticker": "BTC-USD", "trade_ticker": "TQQQ",
        "offset_cl_buy": 10, "buy_operator": ">", "offset_ma_buy": 1, "ma_buy": 20,
        "offset_cl_sell": 50, "sell_operator": "<", "offset_ma_sell": 1, "ma_sell": 10,
        "use_trend_in_buy": True, "use_trend_in_sell": True,
        "offset_compare_short": 1, "ma_compare_short": 5,
        "offset_compare_long": 10, "ma_compare_long": 5,
        "stop_loss_pct": 15.0, "take_profit_pct": 15.0
    },
    
    "BTC-TQQQ2": {
        "signal_ticker": "BTC-USD", "trade_ticker": "TQQQ",
        "offset_cl_buy": 50, "buy_operator": ">", "offset_ma_buy": 20, "ma_buy": 1,
        "offset_cl_sell": 50, "sell_operator": "<", "offset_ma_sell": 20, "ma_sell": 1,
        "use_trend_in_buy": True, "use_trend_in_sell": True,
        "offset_compare_short": 1, "ma_compare_short": 10,
        "offset_compare_long": 20, "ma_compare_long": 10,
        "stop_loss_pct": 35.0, "take_profit_pct": 25.0
    },
    
    
    "453850 ACE 미국30년국채 전략": {
        "signal_ticker": "453850", "trade_ticker": "453850",
        "offset_cl_buy": 16, "buy_operator": "<", "offset_ma_buy": 26, "ma_buy": 15,
        "offset_cl_sell": 26, "sell_operator": ">", "offset_ma_sell": 2, "ma_sell": 20,
        "use_trend_in_buy": True, "use_trend_in_sell": False,
        "offset_compare_short": 2, "ma_compare_short": 15,
        "offset_compare_long": 26, "ma_compare_long": 15,         
        "stop_loss_pct": 0.0, "take_profit_pct": 10.0
    },
          
    "465580 ACE미국빅테크TOP7PLUS": {
        "signal_ticker": "465580", "trade_ticker": "465580",
        "offset_cl_buy": 2, "buy_operator": ">", "offset_ma_buy": 2, "ma_buy": 5,
        "offset_cl_sell": 2, "sell_operator": "<", "offset_ma_sell": 2, "ma_sell": 25, 
        "use_trend_in_buy": False, "use_trend_in_sell": True,
        "offset_compare_short": 6, "ma_compare_short": 10,
        "offset_compare_long": 2, "ma_compare_long": 10,         
        "stop_loss_pct": 0.0, "take_profit_pct": 10.0
    },

    "390390 KODEX미국반도체": {
        "signal_ticker": "390390", "trade_ticker": "390390",
        "offset_cl_buy": 6, "buy_operator": "<", "offset_ma_buy": 2, "ma_buy": 5,
        "offset_cl_sell": 26, "sell_operator": ">", "offset_ma_sell": 2, "ma_sell": 20, 
        "use_trend_in_buy": False, "use_trend_in_sell": True,
        "offset_compare_short": 6, "ma_compare_short": 25,
        "offset_compare_long": 2, "ma_compare_long": 25,
        "stop_loss_pct": 0.0, "take_profit_pct": 10.0
    },

    "371460 TIGER차이나전기차SOLACTIVE": {
        "signal_ticker": "371460", "trade_ticker": "371460",
        "offset_cl_buy": 2, "buy_operator": ">", "offset_ma_buy": 6, "ma_buy": 10,
        "offset_cl_sell": 16, "sell_operator": ">", "offset_ma_sell": 2, "ma_sell": 5, 
        "use_trend_in_buy": True, "use_trend_in_sell": True,
        "offset_compare_short": 6, "ma_compare_short": 15,
        "offset_compare_long": 16, "ma_compare_long": 10,         
        "stop_loss_pct": 0.0, "take_profit_pct": 10.0
    },

    "483280 AITOP10커브드콜": {
        "signal_ticker": "483280", "trade_ticker": "483280",
        "offset_cl_buy": 26, "buy_operator": ">", "offset_ma_buy": 26, "ma_buy": 20,
        "offset_cl_sell": 26, "sell_operator": ">", "offset_ma_sell": 6, "ma_sell": 20, 
        "use_trend_in_buy": True, "use_trend_in_sell": True,
        "offset_compare_short": 2, "ma_compare_short": 20,
        "offset_compare_long": 16, "ma_compare_long": 5,         
        "stop_loss_pct": 0.0, "take_profit_pct": 0.0
    },
}



# === PRESETS 전체 오늘 시그널 일괄 체크 (check_signal_today와 동일한 판정 규칙) ===
def summarize_signal_today(df, p):
    """
    check_signal_today와 '완전히 동일한' 규칙으로
    (1) 오늘 시그널 라벨과
    (2) 최근 BUY/SELL/HOLD 날짜
    를 계산해 반환합니다.

    - 인덱싱 방식: i=0에서 음수 인덱싱 사용 (끝에서 offset만큼 뒤)
    - 추세: ms >= ml
    - offset_compare_short/long 기본 1 (check_signal_today와 동일)
    - 프리셋 p의 오프셋/부호/추세 플래그를 그대로 사용
    """
    import pandas as pd

    if df is None or df.empty:
        return {"label": "데이터없음", "last_buy": None, "last_sell": None, "last_hold": None}

    # 프리셋 파라미터 (기본값도 check_signal_today에 맞춤)
    ma_buy  = int(p.get("ma_buy", 50))
    ma_sell = int(p.get("ma_sell", 10))
    offset_ma_buy  = int(p.get("offset_ma_buy", 50))
    offset_ma_sell = int(p.get("offset_ma_sell", 50))
    offset_cl_buy  = int(p.get("offset_cl_buy", 1))
    offset_cl_sell = int(p.get("offset_cl_sell", 50))

    buy_operator  = p.get("buy_operator",  ">")
    sell_operator = p.get("sell_operator", "<")

    use_trend_in_buy  = bool(p.get("use_trend_in_buy",  True))
    use_trend_in_sell = bool(p.get("use_trend_in_sell", False))

    ma_compare_short = p.get("ma_compare_short", None)
    ma_compare_long  = p.get("ma_compare_long",  None)
    if ma_compare_short is not None:
        ma_compare_short = int(ma_compare_short)
        if ma_compare_short <= 0:
            ma_compare_short = None
    if ma_compare_long is not None:
        ma_compare_long = int(ma_compare_long)
        if ma_compare_long <= 0:
            ma_compare_long = None

    offset_compare_short = int(p.get("offset_compare_short", 1))
    offset_compare_long  = int(p.get("offset_compare_long",  1))

    # 데이터 정규화
    df = df.copy().sort_values("Date").reset_index(drop=True)
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")

    # 이평
    df["MA_BUY"]  = df["Close"].rolling(ma_buy).mean()
    df["MA_SELL"] = df["Close"].rolling(ma_sell).mean()
    if ma_compare_short and ma_compare_long:
        df["MA_SHORT"] = df["Close"].rolling(ma_compare_short).mean()
        df["MA_LONG"]  = df["Close"].rolling(ma_compare_long).mean()

    # ← check_signal_today와 동일: i=0, 음수 인덱싱으로 "끝에서 offset만큼"
    i = 0
    try:
        cl_b = float(df["Close"].iloc[i - offset_cl_buy])
        ma_b = float(df["MA_BUY"].iloc[i - offset_ma_buy])
        cl_s = float(df["Close"].iloc[i - offset_cl_sell])
        ma_s = float(df["MA_SELL"].iloc[i - offset_ma_sell])
    except Exception:
        return {"label": "데이터부족", "last_buy": None, "last_sell": None, "last_hold": None}

    # 추세(있으면) — check_signal_today와 동일: ms >= ml
    trend_ok = True
    if (ma_compare_short and ma_compare_long and
        ("MA_SHORT" in df.columns) and ("MA_LONG" in df.columns)):
        try:
            ms = float(df["MA_SHORT"].iloc[i - offset_compare_short])
            ml = float(df["MA_LONG"].iloc[i - offset_compare_long])
            trend_ok = (ms >= ml)
        except Exception:
            # 데이터 부족이면 추세필터를 '못씀'으로 간주(=기본 True 유지)
            pass

    buy_base  = (cl_b > ma_b) if (buy_operator == ">") else (cl_b < ma_b)
    sell_base = (cl_s < ma_s) if (sell_operator == "<") else (cl_s > ma_s)
    buy_ok  = (buy_base  and trend_ok)        if use_trend_in_buy  else buy_base
    sell_ok = (sell_base and (not trend_ok))  if use_trend_in_sell else sell_base

    if buy_ok and sell_ok: label = "BUY & SELL"
    elif buy_ok:           label = "BUY"
    elif sell_ok:          label = "SELL"
    else:                  label = "HOLD"

    # 최근 BUY/SELL/HOLD 탐색 (check_signal_today와 동일 인덱싱 규칙)
    last_buy = last_sell = last_hold = None
    safe_start = max(
        offset_cl_buy, offset_ma_buy,
        offset_cl_sell, offset_ma_sell,
        (offset_compare_short or 0), (offset_compare_long or 0)
    )
    for j in range(len(df)-1, safe_start-1, -1):
        try:
            cb = df["Close"].iloc[j - offset_cl_buy]
            mb = df["MA_BUY"].iloc[j - offset_ma_buy]
            cs = df["Close"].iloc[j - offset_cl_sell]
            ms = df["MA_SELL"].iloc[j - offset_ma_sell]

            trend_pass = True
            if (ma_compare_short and ma_compare_long and
                "MA_SHORT" in df.columns and "MA_LONG" in df.columns):
                ms_short = df["MA_SHORT"].iloc[j - offset_compare_short]
                ms_long  = df["MA_LONG"].iloc[j - offset_compare_long]
                trend_pass = (ms_short >= ms_long)

            _buy_base  = (cb > mb) if (buy_operator == ">") else (cb < mb)
            _sell_base = (cs < ms) if (sell_operator == "<") else (cs > ms)
            _buy_ok    = (_buy_base  and trend_pass)        if use_trend_in_buy  else _buy_base
            _sell_ok   = (_sell_base and (not trend_pass))  if use_trend_in_sell else _sell_base

            if last_buy is None and _buy_ok:
                last_buy  = pd.to_datetime(df["Date"].iloc[j]).strftime("%Y-%m-%d")
            if last_sell is None and _sell_ok:
                last_sell = pd.to_datetime(df["Date"].iloc[j]).strftime("%Y-%m-%d")
            if last_hold is None and (not _buy_ok and not _sell_ok):
                last_hold = pd.to_datetime(df["Date"].iloc[j]).strftime("%Y-%m-%d")

            if last_buy and last_sell and last_hold:
                break
        except Exception:
            continue

    return {"label": label, "last_buy": last_buy, "last_sell": last_sell, "last_hold": last_hold}


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
    execution_lag_days=0,
    execution_price_mode="same_close"
):
    import numpy as np
    import pandas as pd

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
    xC_sig = x_sig
    xC_trd = x_trd

    if execution_price_mode not in ("same_open", "same_close"):
        execution_price_mode = "same_close"

    def _fill_buy(px: float) -> float:
        return px * (1 + (slip_bps + fee_bps) / 10000.0)

    def _fill_sell(px: float) -> float:
        return px * (1 - (slip_bps + fee_bps) / 10000.0)

    def _check_intraday_exit(buy_px, o, h, l):
        if buy_px is None:
            return False, False, None
        stop_trigger = take_trigger = False
        stop_px = buy_px * (1 - stop_loss_pct / 100.0) if stop_loss_pct > 0 else None
        take_px = buy_px * (1 + take_profit_pct / 100.0) if take_profit_pct > 0 else None
        if stop_px is not None and o <= stop_px:
            return True, False, stop_px
        if take_px is not None and o >= take_px:
            return False, True, take_px
        if stop_px is not None and l <= stop_px <= h:
            stop_trigger = True
        if take_px is not None and l <= take_px <= h:
            take_trigger = True
        if stop_trigger and take_trigger:
            return True, False, stop_px
        if stop_trigger:
            return True, False, stop_px
        if take_trigger:
            return False, True, take_px
        return False, False, None

    cash = float(initial_cash)
    position = 0.0
    buy_price = None
    hold_days = 0
    logs = []
    asset_curve = []
    sb = (strategy_behavior or "1")[:1]  # "1","2","3"

    for i in range(idx0, n):
        just_bought = False
        exec_price = None
        signal = "HOLD"
        reason = None

        open_today  = xO[i]
        high_today  = xH[i]
        low_today   = xL[i]
        close_today = xC_trd[i]

        # 조건 계산 (오프셋 기준)
        try:
            cl_b = float(xC_sig[i - offset_cl_buy])   # BUY 판단용 종가
            ma_b = float(ma_buy_arr[i - offset_ma_buy])
            cl_s = float(xC_sig[i - offset_cl_sell])  # SELL 판단용 종가
            ma_s = float(ma_sell_arr[i - offset_ma_sell])
        except Exception:
            total = cash + (position * close_today if position > 0.0 else 0.0)
            asset_curve.append(total)
            logs.append({
                "날짜": pd.to_datetime(base["Date"].iloc[i]).strftime("%Y-%m-%d"),
                "종가": round(close_today, 2),
                "체결가": None,
                "신호": "HOLD",
                "포지션": round(position, 6),
                "자산": round(total),
                "매수시그널": None,
                "매도시그널": None,
                "손절발동": False,
                "익절발동": False,
                "추세만족": None,
                "보유일": hold_days,
                "양시그널": False,
                "이유": "데이터부족"
            })
            continue

        trend_ok = True
        if (ma_s_arr is not None) and (ma_l_arr is not None):
            ms = ma_s_arr[i - offset_compare_short] if i - offset_compare_short >= 0 else np.nan
            ml = ma_l_arr[i - offset_compare_long]  if i - offset_compare_long  >= 0 else np.nan
            trend_ok = (np.isfinite(ms) and np.isfinite(ml) and ms > ml)

        buy_base  = (cl_b > ma_b) if (buy_operator == ">") else (cl_b < ma_b)
        sell_base = (cl_s < ma_s) if (sell_operator == "<") else (cl_s > ma_s)

        buy_condition  = (buy_base and trend_ok) if use_trend_in_buy  else buy_base
        sell_condition = (sell_base and (not trend_ok)) if use_trend_in_sell else sell_base
        both_condition = bool(buy_condition and sell_condition)

        # 보유 중 장중 stop/take 우선
        stop_hit = take_hit = False
        intraday_px = None
        if position > 0.0 and (stop_loss_pct > 0 or take_profit_pct > 0):
            stop_hit, take_hit, intraday_px = _check_intraday_exit(buy_price, open_today, high_today, low_today)

        if position > 0.0 and (stop_hit or take_hit):
            px = intraday_px if intraday_px is not None else close_today
            fill = _fill_sell(px)
            cash = position * fill
            position = 0.0
            buy_price = None
            signal = "SELL"
            exec_price = fill
            reason = f"{'손절' if stop_hit else '익절'}(px={px:.2f})"

        # 즉시 매도(규칙)
        if position > 0.0 and signal == "HOLD":
            can_sell = (hold_days >= int(min_hold_days))
            if sell_condition and can_sell:
                base_px = open_today if execution_price_mode == "same_open" else close_today
                fill = _fill_sell(base_px)
                cash = position * fill
                position = 0.0
                buy_price = None
                signal = "SELL"
                exec_price = fill
                cmp_op = "<" if sell_operator == "<" else ">"
                trend_txt = "역추세통과" if (use_trend_in_sell and not trend_ok) else ("조건충족" if sell_condition else "조건미충족")
                reason = f"규칙매도: 종가({cl_s:.2f}){cmp_op}MA({ma_s:.2f}) ({trend_txt})"
            elif sell_condition and not can_sell:
                reason = f"보유일미달({hold_days} < {int(min_hold_days)})"

        # 즉시 매수(규칙)
        if position == 0.0 and signal == "HOLD":
            do_buy = False
            if sb == "1":
                do_buy = buy_condition
            elif sb in ("2", "3"):
                do_buy = buy_condition and not sell_condition
                if sb == "3":
                    do_buy = False

            if do_buy:
                base_px = open_today if execution_price_mode == "same_open" else close_today
                fill = _fill_buy(base_px)
                position = cash / fill
                cash = 0.0
                buy_price = fill
                signal = "BUY"
                exec_price = fill
                cmp_op = ">" if buy_operator == ">" else "<"
                trend_txt = "추세통과" if (use_trend_in_buy and trend_ok) else ("조건충족" if buy_condition else "조건미충족")
                both_txt = " & 양시그널" if both_condition else ""
                reason = f"규칙매수: 종가({cl_b:.2f}){cmp_op}MA({ma_b:.2f}) ({trend_txt}){both_txt}"

        # 보유일 갱신
        if position > 0.0:
            if not just_bought:
                hold_days += 1
        else:
            hold_days = 0

        # HOLD 사유 보강(조건미충족 시 현재 비교값 표시)
        if signal == "HOLD" and reason is None:
            buy_ok_txt  = f"종가({cl_b:.2f}){'>' if buy_operator == '>' else '<'}MA({ma_b:.2f})"
            sell_ok_txt = f"종가({cl_s:.2f}){'<' if sell_operator == '<' else '>'}MA({ma_s:.2f})"
            reason = f"조건미충족(매수:{buy_ok_txt}={bool(buy_base)}, 매도:{sell_ok_txt}={bool(sell_base)})" if position==0.0 else "보유중"

        total = cash + (position * close_today if position > 0.0 else 0.0)
        asset_curve.append(total)

        logs.append({
            "날짜": pd.to_datetime(base["Date"].iloc[i]).strftime("%Y-%m-%d"),
            "종가": round(close_today, 2),
            "체결가": round(exec_price, 4) if exec_price is not None else None,
            "신호": signal,
            "포지션": round(position, 6),
            "자산": round(total),
            "매수시그널": bool(buy_condition),
            "매도시그널": bool(sell_condition),
            "손절발동": bool(stop_hit),
            "익절발동": bool(take_hit),
            "추세만족": bool(trend_ok),
            "보유일": hold_days,
            "양시그널": both_condition,
            "이유": reason
        })

    if not asset_curve:
        return {}

    mdd_series = pd.Series(asset_curve)
    peak = mdd_series.cummax()
    drawdown = mdd_series / peak - 1.0
    mdd = float(drawdown.min() * 100)

    mdd_pos = int(np.argmin(drawdown.values))
    df_dates = base["Date"].iloc[-len(mdd_series):].reset_index(drop=True)
    mdd_date = pd.to_datetime(df_dates.iloc[mdd_pos])

    recovery_date = None
    peak_at_mdd = peak.iloc[mdd_pos]
    for j in range(mdd_pos, len(mdd_series)):
        if mdd_series.iloc[j] >= peak_at_mdd:
            recovery_date = pd.to_datetime(df_dates.iloc[j])
            break

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
        pb = b.get("체결가") or b.get("종가")
        ps = s.get("체결가") or s.get("종가")
        if (pb is None) or (ps is None):
            continue
        r = (ps - pb) / pb
        trade_returns.append(r)
        if r >= 0:
            wins += 1; gross_profit += r
        else:
            gross_loss += (-r)

    total_trades = len(trade_returns)
    win_rate = round((wins / total_trades) * 100, 2) if total_trades else 0.0
    import numpy as _np
    avg_trade_return_pct = round((_np.mean(trade_returns) * 100), 2) if trade_returns else 0.0
    median_trade_return_pct = round((_np.median(trade_returns) * 100), 2) if trade_returns else 0.0
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



    # ===== 루프 종료 후: 요약 계산 =====
    if not asset_curve:
        return {}

    mdd_series = pd.Series(asset_curve)
    peak = mdd_series.cummax()
    drawdown = mdd_series / peak - 1.0
    mdd = float(drawdown.min() * 100)

    mdd_pos = int(np.argmin(drawdown.values))
    df_dates = base["Date"].iloc[-len(mdd_series):].reset_index(drop=True)
    mdd_date = pd.to_datetime(df_dates.iloc[mdd_pos])

    recovery_date = None
    peak_at_mdd = peak.iloc[mdd_pos]
    for j in range(mdd_pos, len(mdd_series)):
        if mdd_series.iloc[j] >= peak_at_mdd:
            recovery_date = pd.to_datetime(df_dates.iloc[j])
            break

    # 트레이드 페어 성과
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
        pb = b.get("체결가") or b.get("종가")
        ps = s.get("체결가") or s.get("종가")
        if (pb is None) or (ps is None):
            continue
        r = (ps - pb) / pb
        trade_returns.append(r)
        if r >= 0:
            wins += 1; gross_profit += r
        else:
            gross_loss += (-r)

    total_trades = len(trade_returns)
    win_rate = round((wins / total_trades) * 100, 2) if total_trades else 0.0
    import numpy as _np
    avg_trade_return_pct = round((_np.mean(trade_returns) * 100), 2) if trade_returns else 0.0
    median_trade_return_pct = round((_np.median(trade_returns) * 100), 2) if trade_returns else 0.0
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



# ===== Auto Optimizer# ===== Auto Optimizer (Train/Test) =====
def _score_from_summary(summary: dict, metric: str, mode: str = "max"):
    """
    summary: backtest_fast() 결과 요약 dict (매매 로그 제외)
    metric: "수익률 (%)", "샤프", "Profit Factor", "MDD (%)" 등
    mode: "max" 또는 "min"
    """
    val = summary.get(metric, None)
    if val is None:
        return None
    # MDD(%)는 작을수록 좋으니 보통 min, 나머지는 max 권장
    return val if mode == "max" else (-val)

def _prepare_base_for_range(signal_ticker, trade_ticker, start_date, end_date, ma_pool):
    """기간을 나눠 재계산 (lookahead 방지)."""
    return prepare_base(signal_ticker, trade_ticker, start_date, end_date, ma_pool)

def _try_backtest_once(params, base_pack, fees_pack, exec_pack):
    """주어진 파라미터로 1회 백테스트."""
    base, x_sig, x_trd, ma_dict_sig = base_pack
    fee_bps, slip_bps = fees_pack
    execution_lag_days, execution_price_mode = exec_pack

    # 필요한 MA 즉석 보충
    need_windows = [params["ma_buy"], params["ma_sell"]]
    if params.get("ma_compare_short"): need_windows.append(params["ma_compare_short"])
    if params.get("ma_compare_long"):  need_windows.append(params["ma_compare_long"])
    for w in need_windows:
        if w and w not in ma_dict_sig:
            ma_dict_sig[w] = _fast_ma(x_sig, w)

    r = backtest_fast(
        base, x_sig, x_trd, ma_dict_sig,
        params["ma_buy"], params["offset_ma_buy"],
        params["ma_sell"], params["offset_ma_sell"],
        params["offset_cl_buy"], params["offset_cl_sell"],
        params.get("ma_compare_short"), params.get("ma_compare_long"),
        params["offset_compare_short"], params["offset_compare_long"],
        initial_cash=params["initial_cash"],
        stop_loss_pct=params["stop_loss_pct"], take_profit_pct=params["take_profit_pct"],
        strategy_behavior=params["strategy_behavior"],
        min_hold_days=params["min_hold_days"],
        fee_bps=fee_bps, slip_bps=slip_bps,
        use_trend_in_buy=params["use_trend_in_buy"],
        use_trend_in_sell=params["use_trend_in_sell"],
        buy_operator=params["buy_operator"], sell_operator=params["sell_operator"],
        execution_lag_days=execution_lag_days,
        execution_price_mode=execution_price_mode
    )
    return r

def _sample_params(choices_dict, defaults):
    """choices_dict에서 랜덤 샘플 1개 뽑아 파라미터 dict 구성."""
    def pick(key, fallback):
        arr = choices_dict.get(key, [])
        return random.choice(arr) if arr else fallback

    ma_compare_short = pick("ma_compare_short", defaults["ma_compare_short"])
    mcl_raw          = pick("ma_compare_long", defaults["ma_compare_long"])
    if mcl_raw == "same":
        ma_compare_long = ma_compare_short
    else:
        ma_compare_long = mcl_raw

    return {
        "ma_buy":             pick("ma_buy", defaults["ma_buy"]),
        "offset_ma_buy":      pick("offset_ma_buy", defaults["offset_ma_buy"]),
        "offset_cl_buy":      pick("offset_cl_buy", defaults["offset_cl_buy"]),
        "buy_operator":       pick("buy_operator", defaults["buy_operator"]),

        "ma_sell":            pick("ma_sell", defaults["ma_sell"]),
        "offset_ma_sell":     pick("offset_ma_sell", defaults["offset_ma_sell"]),
        "offset_cl_sell":     pick("offset_cl_sell", defaults["offset_cl_sell"]),
        "sell_operator":      pick("sell_operator", defaults["sell_operator"]),

        "use_trend_in_buy":   pick("use_trend_in_buy", defaults["use_trend_in_buy"]),
        "use_trend_in_sell":  pick("use_trend_in_sell", defaults["use_trend_in_sell"]),
        "ma_compare_short":   ma_compare_short,
        "ma_compare_long":    ma_compare_long,
        "offset_compare_short": pick("offset_compare_short", defaults["offset_compare_short"]),
        "offset_compare_long":  pick("offset_compare_long", defaults["offset_compare_long"]),

        "stop_loss_pct":      pick("stop_loss_pct", defaults["stop_loss_pct"]),
        "take_profit_pct":    pick("take_profit_pct", defaults["take_profit_pct"]),

        "initial_cash":       defaults["initial_cash"],
        "strategy_behavior":  defaults["strategy_behavior"],
        "min_hold_days":      defaults["min_hold_days"],
    }

def auto_search_train_test(
    signal_ticker, trade_ticker,
    start_date, end_date,
    split_ratio,                    # 예: 0.7 → 앞 70% train, 뒤 30% test
    choices_dict,
    n_trials=200,
    objective_metric="수익률 (%)",
    objective_mode="max",           # "max" 또는 "min"
    initial_cash=5_000_000,
    fee_bps=0, slip_bps=0,
    strategy_behavior="1. 포지션 없으면 매수 / 보유 중이면 매도",
    min_hold_days=0,
    execution_lag_days=1,
    execution_price_mode="same_close",
    constraints=None,               # {"min_trades": 5, "min_winrate": 0.0, "max_mdd": None}
):
    """랜덤 탐색 기반 자동 최적화 + Train/Test + Full(전체) 성과 포함."""
    import pandas as pd
    import numpy as np
    import random

    constraints = constraints or {}
    min_trades  = constraints.get("min_trades", 0)
    min_winrate = constraints.get("min_winrate", 0.0)
    max_mdd     = constraints.get("max_mdd", None)

    # 기본 값
    defaults = dict(
        ma_buy=50, offset_ma_buy=50, offset_cl_buy=1, buy_operator=">",
        ma_sell=10, offset_ma_sell=50, offset_cl_sell=50, sell_operator="<",
        use_trend_in_buy=True, use_trend_in_sell=False,
        offset_compare_short=20, ma_compare_short=50, offset_compare_long=50, ma_compare_long=50,
        stop_loss_pct=15.0, take_profit_pct=20.0,
        initial_cash=initial_cash, strategy_behavior=strategy_behavior, min_hold_days=min_hold_days
    )

    # 후보 MA 윈도우 풀(최소화): 속도 위해 집합으로 모아 계산
    ma_pool = set()
    for key in ("ma_buy", "ma_sell", "ma_compare_short", "ma_compare_long"):
        for v in choices_dict.get(key, []):
            if v == "same":
                continue
            if isinstance(v, int) and v > 0:
                ma_pool.add(v)
    if not ma_pool:
        ma_pool = {5, 10, 15, 25, 50}

    # 전체 base (split용 날짜 시퀀스 얻기)
    base_full, x_sig_full, x_trd_full, _ = prepare_base(signal_ticker, trade_ticker, start_date, end_date, list(ma_pool))
    n_all = len(base_full)
    if n_all < 50:
        return pd.DataFrame()

    split_idx = max(1, min(n_all-1, int(n_all * split_ratio)))
    date_train_end = pd.to_datetime(base_full["Date"].iloc[split_idx - 1]).date()

    # Train/Test/FULL base packs
    base_tr, x_sig_tr, x_trd_tr, ma_tr     = prepare_base(signal_ticker, trade_ticker, start_date,     date_train_end, list(ma_pool))
    base_te, x_sig_te, x_trd_te, ma_te     = prepare_base(signal_ticker, trade_ticker, date_train_end, end_date,       list(ma_pool))
    base_fu, x_sig_fu, x_trd_fu, ma_fu     = prepare_base(signal_ticker, trade_ticker, start_date,     end_date,       list(ma_pool))

    base_pack_tr = (base_tr, x_sig_tr, x_trd_tr, ma_tr)
    base_pack_te = (base_te, x_sig_te, x_trd_te, ma_te)
    base_pack_fu = (base_fu, x_sig_fu, x_trd_fu, ma_fu)

    fees_pack    = (fee_bps, slip_bps)
    exec_pack    = (execution_lag_days, execution_price_mode)

    results = []
    seen = set()

    for _ in range(int(n_trials)):
        params = _sample_params(choices_dict, defaults)

        # 중복 파라미터 skip (간단 직렬화)
        sig_key = tuple(sorted((k, str(v)) for k, v in params.items()))
        if sig_key in seen:
            continue
        seen.add(sig_key)

        # Train 실행
        r_tr = _try_backtest_once(params, base_pack_tr, fees_pack, exec_pack)
        if not r_tr:
            continue

        # 제약조건 필터
        trades  = r_tr.get("총 매매 횟수", 0)
        wr      = r_tr.get("승률 (%)", 0.0)
        mdd_val = r_tr.get("MDD (%)", 0.0)
        if trades < min_trades: 
            continue
        if wr < min_winrate:
            continue
        if (max_mdd is not None) and (mdd_val > max_mdd):
            continue

        # Test 실행
        r_te = _try_backtest_once(params, base_pack_te, fees_pack, exec_pack)
        if not r_te:
            continue

        # Test MDD constraint (magnitude)
        _mdd_te = r_te.get("MDD (%)", 0.0)
        _mdd_te_mag = (-_mdd_te) if (_mdd_te < 0) else _mdd_te
        if (max_mdd is not None) and (_mdd_te_mag > max_mdd):
            continue

        # Full 실행 (전체 기간)
        r_fu = _try_backtest_once(params, base_pack_fu, fees_pack, exec_pack)
        if not r_fu:
            continue


        # Full MDD constraint (magnitude)
        _mdd_fu = r_fu.get("MDD (%)", 0.0)
        _mdd_fu_mag = (-_mdd_fu) if (_mdd_fu < 0) else _mdd_fu
        if (max_mdd is not None) and (_mdd_fu_mag > max_mdd):
            continue

        # === 결과 행 ===
        row = {
            # Full (Train+Test 전체)
            f"Full_{objective_metric}": r_fu.get(objective_metric, None),
            "Full_수익률 (%)": r_fu.get("수익률 (%)", None),
            "Full_승률(%)": r_fu.get("승률 (%)", None),
            "Full_MDD(%)": r_fu.get("MDD (%)", None),
            "Full_ProfitFactor": r_fu.get("Profit Factor", None),
            "Full_총매매": r_fu.get("총 매매 횟수", None),
            
            # Train
            f"Train_{objective_metric}": r_tr.get(objective_metric, None),
            "Train_수익률(%)": r_tr.get("수익률 (%)", None),
            "Train_승률(%)": r_tr.get("승률 (%)", None),
            "Train_MDD(%)": r_tr.get("MDD (%)", None),
            "Train_ProfitFactor": r_tr.get("Profit Factor", None),
            "Train_총매매": r_tr.get("총 매매 횟수", None),

            # Test
            f"Test_{objective_metric}": r_te.get(objective_metric, None),
            "Test_수익률(%)": r_te.get("수익률 (%)", None),
            "Test_승률(%)": r_te.get("승률 (%)", None),
            "Test_MDD(%)": r_te.get("MDD (%)", None),
            "Test_ProfitFactor": r_te.get("Profit Factor", None),
            "Test_총매매": r_te.get("총 매매 횟수", None),
        }

        # 파라미터 기록
        row.update({
            "offset_cl_buy": params["offset_cl_buy"], "buy_operator": params["buy_operator"], "offset_ma_buy": params["offset_ma_buy"], "ma_buy": params["ma_buy"],
            "offset_cl_sell": params["offset_cl_sell"], "sell_operator": params["sell_operator"], "offset_ma_sell": params["offset_ma_sell"], "ma_sell": params["ma_sell"],
            "use_trend_in_buy": params["use_trend_in_buy"], "use_trend_in_sell": params["use_trend_in_sell"],
            "offset_compare_short": params["offset_compare_short"], "ma_compare_short": params["ma_compare_short"],
            "offset_compare_long": params["offset_compare_long"], "ma_compare_long": params["ma_compare_long"],
            "stop_loss_pct": params["stop_loss_pct"], "take_profit_pct": params["take_profit_pct"],
            "min_hold_days": params["min_hold_days"],
        })
        results.append(row)

    import pandas as pd
    df = pd.DataFrame(results)
    if df.empty:
        return df

    # Full 별칭 컬럼(보기용)
    if "Full_수익률 (%)" in df.columns and "FULL_수익률" not in df.columns:
        df["FULL_수익률"] = df["Full_수익률 (%)"]
    if "Full_MDD(%)" in df.columns and "FULL_MDD" not in df.columns:
        df["FULL_MDD"] = df["Full_MDD(%)"]
    elif "Full_MDD (%)" in df.columns and "FULL_MDD" not in df.columns:
        df["FULL_MDD"] = df["Full_MDD (%)"]
    if "Full_ProfitFactor" in df.columns and "FULL_ProfitFactor" not in df.columns:
        df["FULL_ProfitFactor"] = df["Full_ProfitFactor"]

    # 정렬: Full 기준
    _full_col_map = {
        "수익률 (%)": "Full_수익률 (%)",
        "MDD (%)": "Full_MDD(%)" if "Full_MDD(%)" in df.columns else "Full_MDD (%)",
        "Profit Factor": "Full_ProfitFactor",
        "승률": "Full_승률(%)",
        "샤프": "Full_샤프"  # 있을 경우
    }
    _col = _full_col_map.get(objective_metric, "Full_수익률 (%)")
    _asc = (objective_mode == "min")
    if _col in df.columns:
        df = df.sort_values(by=_col, ascending=_asc, na_position="last").reset_index(drop=True)

    # Full 관련 컬럼 앞으로
    _prefer = [
        "FULL_수익률", "FULL_MDD", "FULL_ProfitFactor",
        "Full_수익률 (%)", "Full_MDD(%)", "Full_MDD (%)", "Full_ProfitFactor",
        "Test_수익률(%)", "Test_MDD(%)", "Test_ProfitFactor",
        "Train_수익률(%)", "Train_MDD(%)", "Train_ProfitFactor",
    ]
    _front = [c for c in _prefer if c in df.columns]
    _rest = [c for c in df.columns if c not in _front]
    df = df[_front + _rest]

    return df

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
            execution_price_mode="same_close"
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



#########################################################
# ✅ UI 구성 (UI-only; 로직 함수는 기존 그대로 사용)

# 페이지/헤더 (상단 set_page_config만 사용)
st.title("📊 전략 백테스트 웹앱")


# initialize defaults and apply preset once per run
_init_default_state()
if "preset_name" in st.session_state:
    _apply_preset_to_state({} if st.session_state["preset_name"] == "직접 설정" else PRESETS.get(st.session_state["preset_name"], {}))
st.markdown("모든 매매는 종가 매매이나, 손절,익절은 장중 시가. n일전 데이터 기반으로 금일 종가 매매를 한다.")
st.markdown("KODEX미국반도체 390390, KODEX200 069500 KDOEX인버스 114800, KODEX미국나스닥100 379810, ACEKRX금현물 411060, KODEX은선물 114800, ACE미국30년국채액티브(H) 453850, ACE미국빅테크TOP7Plus 465580")

# 📌 프리셋 선택
selected_preset = st.selectbox("🎯 전략 프리셋 선택", ["직접 설정"] + list(PRESETS.keys()), key="preset_name", on_change=_on_preset_change, args=(PRESETS,))
preset_values = {} if selected_preset == "직접 설정" else PRESETS[selected_preset]

# 기본 입력
col1, col2 = st.columns(2)
with col1:
    signal_ticker = st.text_input(
        "시그널 판단용 티커",
        value=preset_values.get("signal_ticker", "SOXL"),
        key="signal_ticker_input"
    )
with col2:
    trade_ticker = st.text_input(
        "실제 매매 티커",
        value=preset_values.get("trade_ticker", "SOXL"),
        key="trade_ticker_input"
    )

col3, col4 = st.columns(2)
with col3:
    start_date = st.date_input(
        "시작일",
        value=datetime.date(2015, 1, 1),
        min_value=datetime.date(1990, 1, 1),
        max_value=datetime.date.today()
    )
with col4:
    end_date = st.date_input(
        "종료일",
        value=datetime.date.today(),
        min_value=start_date,
        max_value=datetime.date.today()
    )

# 전략 조건 설정
with st.expander("📈 전략 조건 설정", expanded=False):
    ops = [">", "<"]

    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown("**📥 매수 조건**")
        offset_cl_buy = st.number_input("□일 전 종가", key="offset_cl_buy", value=preset_values.get("offset_cl_buy", 1))
        buy_operator = st.selectbox("매수 조건 부호", ops, index=ops.index(st.session_state.get("buy_operator", ">")), key="buy_operator")
        offset_ma_buy = st.number_input("□일 전", key="offset_ma_buy", value=preset_values.get("offset_ma_buy", 50))
        ma_buy = st.number_input("□일 이동평균선", key="ma_buy", value=preset_values.get("ma_buy", 50))

        st.markdown("---")
        use_trend_in_buy = st.checkbox("매수에 추세필터 적용", value=preset_values.get("use_trend_in_buy", True))
        offset_compare_short = st.number_input("□일 전", key="offset_compare_short", value=preset_values.get("offset_compare_short", 20))
        ma_compare_short = st.number_input("□일 이동평균선이 (short)", key="ma_compare_short", value=preset_values.get("ma_compare_short", 50))
        offset_compare_long = st.number_input("□일 전", key="offset_compare_long", value=preset_values.get("offset_compare_long", 50))
        ma_compare_long = st.number_input("□일 이동평균선 (long)보다 커야 **매수**", key="ma_compare_long", value=preset_values.get("ma_compare_long", 50))

    with col_right:
        st.markdown("**📤 매도 조건**")
        offset_cl_sell = st.number_input("□일 전 종가", key="offset_cl_sell", value=preset_values.get("offset_cl_sell", 50))
        sell_operator = st.selectbox("매도 조건 부호", ops, index=ops.index(st.session_state.get("sell_operator", "<")), key="sell_operator")
        offset_ma_sell = st.number_input("□일 전", key="offset_ma_sell", value=preset_values.get("offset_ma_sell", 50))
        ma_sell = st.number_input("□일 이동평균선", key="ma_sell", value=preset_values.get("ma_sell", 10))

        stop_loss_pct = st.number_input("손절 기준 (%)", key="stop_loss_pct", value=preset_values.get("stop_loss_pct", 15.0), step=0.5)
        take_profit_pct = st.number_input("익절 기준 (%)", key="take_profit_pct", value=preset_values.get("take_profit_pct", 20.0), step=0.5)
        min_hold_days = st.number_input("매수 후 최소 보유일", key="min_hold_days", value=0, min_value=0, step=1)

        st.markdown("---")
        use_trend_in_sell = st.checkbox("매도는 역추세만(추세 불통과일 때만)", value=preset_values.get("use_trend_in_sell", True))

    strategy_behavior = st.selectbox("⚙️ 매수/매도 조건 동시 발생 시 행동", options=[
        "1. 포지션 없으면 매수 / 보유 중이면 매도",
        "2. 포지션 없으면 매수 / 보유 중이면 HOLD",
        "3. 포지션 없으면 HOLD / 보유 중이면 매도"
    ], key="strategy_behavior")

# 체결/비용 & 기타
with st.expander("⚙️ 체결/비용 & 기타 설정", expanded=False):
    initial_cash_ui = st.number_input("초기 자본", value=5_000_000, step=100_000)
    fee_bps = st.number_input("거래수수료 (bps)", value=25, step=1)
    slip_bps = st.number_input("슬리피지 (bps)", value=0, step=1)
    seed = st.number_input("랜덤 시뮬 Seed (재현성)", value=0, step=1)
    if seed:
        random.seed(int(seed))

# ================== 탭 ==================
tab1, tab2, tab3 = st.tabs(["🎯 오늘 시그널", "📚 PRESETS 일괄", "🧪 백테스트"])

# ───────────────────────────────────────
# TAB1: 오늘 시그널
# ───────────────────────────────────────
with tab1:
    c1, c2 = st.columns(2)
    with c1:
        if st.button("📌 오늘 시그널 체크", use_container_width=True):
            df_today = get_data(signal_ticker, start_date, end_date)
            if df_today.empty:
                st.warning("데이터가 없습니다.")
            else:
                check_signal_today(
                    df_today,
                    ma_buy=ma_buy, offset_ma_buy=offset_ma_buy,
                    ma_sell=ma_sell, offset_ma_sell=offset_ma_sell,
                    offset_cl_buy=offset_cl_buy, offset_cl_sell=offset_cl_sell,
                    ma_compare_short=ma_compare_short if (ma_compare_short or 0) > 0 else None,
                    ma_compare_long=ma_compare_long  if (ma_compare_long  or 0) > 0 else None,
                    offset_compare_short=offset_compare_short,
                    offset_compare_long=offset_compare_long,
                    buy_operator=buy_operator,
                    sell_operator=sell_operator,
                    use_trend_in_buy=use_trend_in_buy,
                    use_trend_in_sell=use_trend_in_sell,
                    #strategy_behavior=strategy_behavior,
                    #min_hold_days=int(min_hold_days)
                )
    with c2:
        if st.button("⚡ 오늘 시그널 체크 (실시간)", use_container_width=True):
            df_today = get_data(signal_ticker, start_date, end_date)
            if df_today.empty:
                st.error("기본 데이터 로딩 실패")
            else:
                is_krx = (signal_ticker.isdigit() or signal_ticker.lower().endswith(".ks") or signal_ticker.lower().endswith(".kq"))
                if is_krx:
                    st.warning("국내 티커는 일봉 데이터로 판정합니다.")
                    check_signal_today(
                        df_today,
                        ma_buy=ma_buy, offset_ma_buy=offset_ma_buy,
                        ma_sell=ma_sell, offset_ma_sell=offset_ma_sell,
                        offset_cl_buy=offset_cl_buy, offset_cl_sell=offset_cl_sell,
                        ma_compare_short=ma_compare_short if (ma_compare_short or 0) > 0 else None,
                        ma_compare_long=ma_compare_long  if (ma_compare_long  or 0) > 0 else None,
                        offset_compare_short=offset_compare_short, offset_compare_long=offset_compare_long,
                        buy_operator=buy_operator, sell_operator=sell_operator,
                        use_trend_in_buy=use_trend_in_buy, use_trend_in_sell=use_trend_in_sell,
                        #strategy_behavior=strategy_behavior,
                        #min_hold_days=int(min_hold_days)
                    )
                else:
                    check_signal_today_realtime(
                        df_today, signal_ticker,
                        tz="America/New_York", session_start="09:30", session_end="16:00",
                        ma_buy=ma_buy, offset_ma_buy=offset_ma_buy,
                        ma_sell=ma_sell, offset_ma_sell=offset_ma_sell,
                        offset_cl_buy=offset_cl_buy, offset_cl_sell=offset_cl_sell,
                        ma_compare_short=ma_compare_short, ma_compare_long=ma_compare_long,
                        offset_compare_short=offset_compare_short, offset_compare_long=offset_compare_long,
                        buy_operator=buy_operator, sell_operator=sell_operator,
                        use_trend_in_buy=use_trend_in_buy, use_trend_in_sell=use_trend_in_sell,
                        #strategy_behavior=strategy_behavior,
                        #min_hold_days=int(min_hold_days)
                    )

# ───────────────────────────────────────
# TAB2: PRESETS 일괄
# ───────────────────────────────────────
with tab2:
    st.markdown("#### 🧭 PRESETS 오늘 시그널 요약")

    # EOD 기준
    if st.button("📚 PRESETS 전체 오늘 시그널 보기", use_container_width=True):
        rows = []
        for name, p in PRESETS.items():
            sig_tic = p.get("signal_ticker", p.get("trade_ticker"))
            df = get_data(sig_tic, start_date, end_date)
            res = summarize_signal_today(df, p) if not df.empty else {
                "label": "데이터없음", "last_buy": None, "last_sell": None, "last_hold": None
            }
            rows.append({
                "전략명": name,
                "티커": sig_tic,
                "시그널": res["label"],
                "최근 BUY": res["last_buy"] or "-",
                "최근 SELL": res["last_sell"] or "-",
                "최근 HOLD": res["last_hold"] or "-",
                "비고": "-"
            })
        df_view = pd.DataFrame(rows)
        st.dataframe(df_view, use_container_width=True)
        st.download_button(
            "⬇️ CSV 다운로드 (EOD)",
            data=df_view.to_csv(index=False).encode("utf-8-sig"),
            file_name="presets_signal_eod.csv",
            mime="text/csv"
        )

    # 실시간(US 1분봉 집계 반영)
    if st.button("📚 PRESETS 전체 오늘 시그널 (실시간)", use_container_width=True):
        rows = []
        for name, p in PRESETS.items():
            sig_tic = p.get("signal_ticker", p.get("trade_ticker"))
            tz = "America/New_York"
            session_start, session_end = "09:30", "16:00"

            df0 = get_data(sig_tic, start_date, end_date)
            src = "EOD"
            df_rt = df0.copy()

            if not df0.empty and not (sig_tic.isdigit() or sig_tic.lower().endswith(".ks") or sig_tic.lower().endswith(".kq")):
                daily_close_1m, last_px, last_ts = get_yf_1m_grouped_close(
                    sig_tic, tz=tz, session_start=session_start, session_end=session_end
                )
                if not daily_close_1m.empty and last_ts is not None:
                    df_rt = df_rt.sort_values("Date").drop_duplicates(subset=["Date"]).reset_index(drop=True)
                    df_rt["Date"] = pd.to_datetime(df_rt["Date"])
                    df_rt["Date_only"] = df_rt["Date"].dt.date

                    ts = pd.Timestamp(last_ts)
                    if ts.tz is None:
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

                    df_rt = df_rt.sort_values("Date").drop_duplicates(subset=["Date"]).drop(columns=["Date_only"], errors="ignore").reset_index(drop=True)
                    src = "yfinance_1m_grouped"

            if not df_rt.empty:
                p_rt = dict(p)
                p_rt.update({
                    "offset_cl_buy": 0, "offset_ma_buy": 0,
                    "offset_cl_sell": 0, "offset_ma_sell": 0,
                    "offset_compare_short": 0, "offset_compare_long": 0,
                })
                res = summarize_signal_today(df_rt, p_rt)
            else:
                res = {"label": "데이터없음", "last_buy": None, "last_sell": None, "last_hold": None}

            rows.append({
                "전략명": name,
                "티커": sig_tic,
                "시그널": res["label"],
                "최근 BUY":  res.get("last_buy")  or "-",
                "최근 SELL": res.get("last_sell") or "-",
                "최근 HOLD": res.get("last_hold") or "-",
                "가격소스": src,
            })

        df_rt_view = pd.DataFrame(rows)
        st.dataframe(df_rt_view, use_container_width=True)
        st.download_button(
            "⬇️ CSV 다운로드 (실시간 요약)",
            data=df_rt_view.to_csv(index=False).encode("utf-8-sig"),
            file_name="presets_signal_realtime.csv",
            mime="text/csv"
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
            execution_price_mode="same_close"
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
            st.dataframe((df_log[[c for c in ["FULL_수익률","FULL_MDD","FULL_ProfitFactor,Full_수익률 (%)","Full_MDD (%)","Full_ProfitFactor"] if c in df_log.columns]] if any(c in df_log.columns for c in ["Full_수익률 (%)","Full_MDD (%)","Full_ProfitFactor","FULL_수익률","FULL_MDD","FULL_ProfitFactor"]) else df_log))
        csv = df_log.reset_index().to_csv(index=False).encode("utf-8-sig")
        st.download_button("⬇️ 백테스트 결과 다운로드 (CSV)", data=csv, file_name="backtest_result.csv", mime="text/csv")


    # ── 랜덤 시뮬 / 자동 탐색 (선택) ──
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
            txt_offset_cl_buy     = st.text_input("offset_cl_buy 후보",     "1,10,20,50")
            txt_buy_op            = st.text_input("buy_operator 후보",      ">,<")
            txt_offset_ma_buy     = st.text_input("offset_ma_buy 후보",     "1,10,20,50")
            txt_ma_buy            = st.text_input("ma_buy 후보",            "1,5,10,20,50")

            txt_offset_cl_sell    = st.text_input("offset_cl_sell 후보",    "1,10,20,50")
            txt_sell_op           = st.text_input("sell_operator 후보",     "<,>")
            txt_offset_ma_sell    = st.text_input("offset_ma_sell 후보",    "1,10,20,50")
            txt_ma_sell           = st.text_input("ma_sell 후보",           "1,5,10,20,50")

        with colR:
            txt_off_cmp_s         = st.text_input("offset_compare_short 후보", "1,10,20,50")
            txt_ma_cmp_s          = st.text_input("ma_compare_short 후보",     "1,5,10,20,50")
            txt_off_cmp_l         = st.text_input("offset_compare_long 후보",  "1,10,20,50")
            txt_ma_cmp_l          = st.text_input("ma_compare_long 후보",      "same")

            txt_use_trend_buy     = st.text_input("use_trend_in_buy 후보(True/False)",  "True,False")
            txt_use_trend_sell    = st.text_input("use_trend_in_sell 후보(True/False)", "True")
            txt_stop_loss         = st.text_input("stop_loss_pct 후보(%)",  "0")
            txt_take_profit       = st.text_input("take_profit_pct 후보(%)","0,15,25")

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
                execution_price_mode="same_close",
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
                        "offset_cl_buy","buy_operator","offset_ma_buy","ma_buy",
                        "offset_cl_sell","sell_operator","offset_ma_sell","ma_sell",
                        "use_trend_in_buy","use_trend_in_sell",
                        "offset_compare_short","ma_compare_short","offset_compare_long","ma_compare_long",                        
                        "stop_loss_pct","take_profit_pct","min_hold_days"
                    ]})












