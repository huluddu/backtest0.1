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

# ===== utils: parse list inputs =====
def _parse_list(text: str, cast_fn=int):
    """
    "5, 15, 25" -> [5, 15, 25]
    빈 문자열/None 이면 빈 리스트 반환.
    cast_fn: int/float/str 중 하나.
    """
    if not text:
        return []
    return [cast_fn(x.strip()) for x in str(text).split(",") if x.strip() != ""]


# ===== utils: build MA dict =====
def build_ma_dict_sig(close_series, ma_periods: set[int]):
    """
    close_series: pd.Series of close
    ma_periods: {5, 15, 25, ...} 0 또는 None은 제외
    return: {period: np.ndarray}
    """
    import numpy as np
    ma_dict = {}
    for p in sorted({int(p) for p in ma_periods if p and int(p) > 0}):
        ma_dict[p] = close_series.rolling(p).mean().to_numpy()
    return ma_dict


# ===== random simulator core =====
import random
import pandas as pd

def run_random_simulations(
    base, x_sig, x_trd, ma_dict_sig,
    n_runs: int,
    param_lists: dict,
    seed: int | None = None
):
    """
    n_runs: 반복 횟수 (예: 100)
    param_lists 예:
        {
            "ma_buy": [5,15,25], "offset_ma_buy": [1,5,25],
            "ma_sell": [5,15,25], "offset_ma_sell": [1],
            "offset_cl_buy": [5,15,25], "offset_cl_sell": [1,5],
            "ma_compare_short": [0,5,15,25], "ma_compare_long": [0,25],
            "offset_compare_short": [1,5,25], "offset_compare_long": [1,5,25],
            "stop_loss_pct": [0.0], "take_profit_pct": [0.0, 25, 50],
            "strategy_behavior": ["1. 포지션 없으면 매수 / 보유 중이면 매도"],
            "min_hold_days": [0, 3],
            "fee_bps": [0], "slip_bps": [0],
            "initial_cash": [5_000_000]
        }
    """
    if seed is not None:
        random.seed(seed)

    rows = []
    for i in range(1, n_runs + 1):
        # 1) 이번 회차 파라미터 샘플링
        picked = {}
        for k, v in param_lists.items():
            picked[k] = (random.choice(v) if isinstance(v, list) and len(v) > 0 else None)

        # 0 -> None (비교 MA 비활성화 용도) 치환
        for k in ("ma_compare_short", "ma_compare_long"):
            if picked.get(k) == 0:
                picked[k] = None

        # 2) backtest_fast 호출
        res = backtest_fast(
            base=base,
            x_sig=x_sig, x_trd=x_trd,
            ma_dict_sig=ma_dict_sig,
            ma_buy=picked.get("ma_buy"),
            offset_ma_buy=picked.get("offset_ma_buy"),
            ma_sell=picked.get("ma_sell"),
            offset_ma_sell=picked.get("offset_ma_sell"),
            offset_cl_buy=picked.get("offset_cl_buy"),
            offset_cl_sell=picked.get("offset_cl_sell"),
            ma_compare_short=picked.get("ma_compare_short"),
            ma_compare_long=picked.get("ma_compare_long"),
            offset_compare_short=picked.get("offset_compare_short", 1),
            offset_compare_long=picked.get("offset_compare_long", 1),
            initial_cash=picked.get("initial_cash", 5_000_000),
            stop_loss_pct=picked.get("stop_loss_pct", 0.0),
            take_profit_pct=picked.get("take_profit_pct", 0.0),
            strategy_behavior=picked.get("strategy_behavior", "1. 포지션 없으면 매수 / 보유 중이면 매도"),
            min_hold_days=picked.get("min_hold_days", 0),
            fee_bps=picked.get("fee_bps", 0),
            slip_bps=picked.get("slip_bps", 0),
        )

        # 3) 결과 지표 정리 (네 코드 키 이름 맞춰서)
        row = {
            "run": i,
            **picked,
            "최종자산": res.get("최종자산"),
            "총수익률(%)": res.get("총수익률(%)"),
            "연율화 수익률(%)": res.get("연율화 수익률(%)"),
            "연율화 변동성(%)": res.get("연율화 변동성(%)"),
            "샤프": res.get("샤프"),
            "MDD(%)": res.get("MDD(%)") or res.get("최대낙폭(%)"),
            "승률(%)": res.get("승률(%)"),
            "거래횟수": res.get("거래횟수"),
            "기간": res.get("기간"),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    sort_cols = [c for c in ["샤프", "연율화 수익률(%)", "총수익률(%)"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(by=sort_cols, ascending=[False] * len(sort_cols), na_position="last")
    best = df.iloc[0].to_dict() if len(df) else {}
    return df, best



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

@st.cache_data(show_spinner=False, ttl=3600)
def get_krx_data_cached(ticker: str, start_date, end_date):
    """KRX(숫자티커)용: pykrx에서 종가만 가져와 정리"""
    df = stock.get_etf_ohlcv_by_date(start_date.strftime("%Y%m%d"),
                                     end_date.strftime("%Y%m%d"),
                                     ticker)
    df = df[["종가"]].reset_index().rename(columns={"날짜": "Date", "종가": "Close"})
    return df

@st.cache_data(show_spinner=False, ttl=3600)
def get_yf_data_cached(ticker: str, start_date, end_date):
    """야후파이낸스용: Close만 단일 컬럼으로 정리"""
    df = yf.download(ticker, start=start_date, end=end_date)
    if isinstance(df.columns, pd.MultiIndex):
        # 티커 멀티컬럼 보정
        if ("Close", ticker.upper()) in df.columns:
            df = df[("Close", ticker.upper())]
        elif "Close" in df.columns.get_level_values(0):
            df = df["Close"]
        df = df.to_frame(name="Close")
    elif isinstance(df, pd.Series):
        df = df.to_frame(name="Close")
    df = df[["Close"]].dropna().reset_index()
    df.columns = ["Date", "Close"]
    return df

def get_data(ticker: str, start_date, end_date) -> pd.DataFrame:
    """티커 타입에 따라 KRX/yf 로더 분기"""
    try:
        if ticker.lower().endswith(".ks") or ticker.isdigit():
            return get_krx_data_cached(ticker, start_date, end_date)
        return get_yf_data_cached(ticker, start_date, end_date)
    except Exception as e:
        st.error(f"❌ 데이터 로딩 실패: {e}")
        return pd.DataFrame()



# ===== Base prepare =====
@st.cache_data(show_spinner=False, ttl=1800)
def prepare_base(signal_ticker, trade_ticker, start_date, end_date, ma_pool):
    """한 번에 머지 + 필요한 모든 MA(신호용) 미리 계산"""
    sig = get_data(signal_ticker, start_date, end_date).sort_values("Date")
    trd = get_data(trade_ticker, start_date, end_date).sort_values("Date")
    base = pd.merge(sig, trd, on="Date", suffixes=("_sig", "_trd"), how="inner").dropna().reset_index(drop=True)

    x_sig = base["Close_sig"].to_numpy(dtype=float)
    x_trd = base["Close_trd"].to_numpy(dtype=float)

    ma_dict_sig = {}
    for w in sorted(set([w for w in ma_pool if w and w > 0])):
        ma_dict_sig[w] = _fast_ma(x_sig, w)

    return base, x_sig, x_trd, ma_dict_sig


# ===== Prepare ma_dict_sig for Random Simulator =====
def _prepare_ma_dict_for_random(base, x_sig, ma_period_candidates):
    """
    base: 로그/시각화용 DF에 "종가" 열이 있다고 가정
    x_sig: 시그널 종가(Series 또는 ndarray) — base에 종가가 없으면 여기서 Series를 써야 함
    """
    import pandas as pd

    # 후보들 모아서 필요한 MA 기간 set 만들기 (0/None 제외)
    need_ma_periods = {int(p) for p in ma_period_candidates if p and int(p) > 0}

    # 종가 Series 확보
    if isinstance(base, pd.DataFrame) and ("종가" in base.columns):
        close_series = base["종가"]
    else:
        # x_sig가 pandas Series가 아니라면 Series로 변환
        if hasattr(x_sig, "rolling"):
            close_series = x_sig
        else:
            close_series = pd.Series(x_sig, name="Close")

    return build_ma_dict_sig(close_series, need_ma_periods)




def get_mdd(asset_curve):
    peak = asset_curve.cummax()
    drawdown = (asset_curve - peak) / peak
    return drawdown.min() * 100


def check_signal_today(df, ma_buy, offset_ma_buy, ma_sell, offset_ma_sell,
                       offset_cl_buy, offset_cl_sell,
                       ma_compare_short=None, ma_compare_long=None,
                       offset_compare_short=1, offset_compare_long=1):

    df = df.copy()
    df = df.sort_values("Date").reset_index(drop=True)
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df["MA_BUY"] = df["Close"].rolling(ma_buy).mean()
    df["MA_SELL"] = df["Close"].rolling(ma_sell).mean()

    if ma_compare_short and ma_compare_long:
        df["MA_SHORT"] = df["Close"].rolling(ma_compare_short).mean()
        df["MA_LONG"] = df["Close"].rolling(ma_compare_long).mean()

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
    if ma_compare_short and ma_compare_long:
        try:
            ma_short = float(df["MA_SHORT"].iloc[i - offset_compare_short])
            ma_long = float(df["MA_LONG"].iloc[i - offset_compare_long])
            trend_ok = ma_short >= ma_long
            trend_msg = f"{ma_short:.2f} vs {ma_long:.2f} → {'통과 ✅' if trend_ok else '미통과 ❌'}"
        except:
            trend_msg = "❗데이터 부족"
            trend_ok = False

    st.write(f"📈 추세 조건: {trend_msg}")

    buy_ok = cl_b > ma_b and trend_ok
    sell_ok = cl_s < ma_s

    st.write(f"💡 매수판단: 종가({cl_b:.2f}) {'>' if cl_b > ma_b else '<='} MA({ma_b:.2f}) → {'매수조건 ✅' if buy_ok else '조건부족 ❌'}")
    st.write(f"💡 매도판단: 종가({cl_s:.2f}) {'<' if cl_s < ma_s else '>='} MA({ma_s:.2f}) → {'매도조건 ✅' if sell_ok else '조건부족 ❌'}")

    if buy_ok:
        st.success("📈 오늘은 매수 시그널입니다!")
    elif sell_ok:
        st.error("📉 오늘은 매도 시그널입니다!")
    else:
        st.info("⏸ 매수/매도 조건 모두 만족하지 않음")

    last_buy_date = None
    last_sell_date = None

    for j in range(len(df) - max(offset_cl_buy, offset_ma_buy), 0, -1):
        try:
            cb = df["Close"].iloc[j - offset_cl_buy]
            mb = df["MA_BUY"].iloc[j - offset_ma_buy]
            cs = df["Close"].iloc[j - offset_cl_sell]
            ms = df["MA_SELL"].iloc[j - offset_ma_sell]

            trend_pass = True
            if ma_compare_short and ma_compare_long:
                ms_short = df["MA_SHORT"].iloc[j - offset_compare_short]
                ms_long = df["MA_LONG"].iloc[j - offset_compare_long]
                trend_pass = ms_short >= ms_long

            if last_buy_date is None and cb > mb and trend_pass:
                last_buy_date = df["Date"].iloc[j]
            if last_sell_date is None and cs < ms:
                last_sell_date = df["Date"].iloc[j]

            if last_buy_date and last_sell_date:
                break
        except:
            continue

    if last_buy_date:
        st.write(f"🗓 마지막 매수 조건 만족: {last_buy_date.strftime('%Y-%m-%d')}")
    if last_sell_date:
        st.write(f"🗓 마지막 매도 조건 만족: {last_sell_date.strftime('%Y-%m-%d')}")
    if not last_buy_date and not last_sell_date:
        st.warning("❗최근 매수/매도 조건에 부합한 날이 없습니다.")


# ✅ 전략 프리셋 목록 정의
PRESETS = {
    "SOXL 매수/매도 추세 포함 전략": {
        "ma_buy": 15, "offset_ma_buy": 15, "offset_cl_buy": 5,
        "ma_sell": 25, "offset_ma_sell": 1, "offset_cl_sell": 5,
        "ma_compare_short": 5, "ma_compare_long": 5,
        "offset_compare_short": 25, "offset_compare_long": 1,
        "buy_operator": "<", "sell_operator": "<",
        "use_trend_in_buy": True, "use_trend_in_sell": True,
        "stop_loss_pct": 0.0, "take_profit_pct": 0.0
    },

    "SOXL 익절 포함 최고 전략": {
        "ma_buy": 25, "offset_ma_buy": 5, "offset_cl_buy": 25,
        "ma_sell": 25, "offset_ma_sell": 1, "offset_cl_sell": 1,
        "ma_compare_short": 25, "ma_compare_long": 25,
        "offset_compare_short": 25, "offset_compare_long": 1,
        "buy_operator": ">", "sell_operator": "<",
        "use_trend_in_buy": True, "use_trend_in_sell": False,
        "stop_loss_pct": 0.0, "take_profit_pct": 50.0
    }
}

# ✅ UI 구성
st.set_page_config(page_title="전략 백테스트", layout="wide")
st.title("📊 전략 백테스트 웹앱")

st.markdown("KODEX미국반도체 390390, KODEX미국나스닥100 379810, ACEKRX금현물 411060, ACE미국30년국채액티브(H) 453850, ACE미국빅테크TOP7Plus 465580")

col1, col2 = st.columns(2)
with col1:
    signal_ticker = st.text_input("시그널 판단용 티커", value="SOXL")
with col2:
    trade_ticker = st.text_input("실제 매매 티커", value="SOXL")

col3, col4 = st.columns(2)
with col3:
    start_date = st.date_input("시작일", value=datetime.date(2021, 1, 1))
with col4:
    end_date = st.date_input("종료일", value=datetime.date.today())

with st.expander("📈 전략 조건 설정"):
    # 📌 프리셋 선택 UI
    selected_preset = st.selectbox("🎯 전략 프리셋 선택", ["직접 설정"] + list(PRESETS.keys()))
    preset_values = {} if selected_preset == "직접 설정" else PRESETS[selected_preset]

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
        ma_compare_short = st.number_input("□일 이동평균선보다", key="ma_compare_short", value=preset_values.get("ma_compare_short", 25))
        offset_compare_long = st.number_input("□일 전", key="offset_compare_long", value=preset_values.get("offset_compare_long", 1))
        ma_compare_long = st.number_input("□일 이동평균선이 커야 **매수**", key="ma_compare_long", value=preset_values.get("ma_compare_long", 25))

    with col_right:
        st.markdown("**📤 매도 조건**")
        offset_cl_sell = st.number_input("□일 전 종가", key="offset_cl_sell", value=preset_values.get("offset_cl_sell", 1))
        sell_operator = st.selectbox("매도 조건 부호", ops, index=ops.index(preset_values.get("sell_operator", "<")))
        offset_ma_sell = st.number_input("□일 전", key="offset_ma_sell", value=preset_values.get("offset_ma_sell", 1))
        ma_sell = st.number_input("□일 이동평균선", key="ma_sell", value=preset_values.get("ma_sell", 25))
        stop_loss_pct = st.number_input("손절 기준 (%)", key="stop_loss_pct", value=preset_values.get("stop_loss_pct", 0.0), step=0.5)
        take_profit_pct = st.number_input("익절 기준 (%)", key="take_profit_pct", value=preset_values.get("take_profit_pct", 0.0), step=0.5)
        min_hold_days = st.number_input("매수 후 최소 보유일", key="min_hold_days", value=0, min_value=0, step=1)
        st.markdown("---")
        use_trend_in_sell = st.checkbox("매도는 역추세만(추세 불통과일 때만)", value=preset_values.get("use_trend_in_sell", False))
        

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
    slip_bps = st.number_input("슬리피지 (bps)", value=0, step=1)
    seed = st.number_input("랜덤 시뮬 Seed (재현성)", value=0, step=1)
    if seed:
        random.seed(int(seed))

# ✅ 시그널 체크
if st.button("📌 오늘 시그널 체크"):
    df_today = get_data(signal_ticker, start_date, end_date)
    if not df_today.empty:
        check_signal_today(df_today,
            ma_buy=ma_buy,
            offset_ma_buy=offset_ma_buy,
            ma_sell=ma_sell,
            offset_ma_sell=offset_ma_sell,
            offset_cl_buy=offset_cl_buy,
            offset_cl_sell=offset_cl_sell,
            ma_compare_short=ma_compare_short if ma_compare_short > 0 else None,
            ma_compare_long=ma_compare_long if ma_compare_long > 0 else None,
            offset_compare_short=offset_compare_short,
            offset_compare_long=offset_compare_long
        )


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
    buy_operator=">", sell_operator="<"
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

    # ===== 변수 =====
    cash = float(initial_cash)
    position = 0.0
    buy_price = None
    asset_curve, logs = [], []
    sb = strategy_behavior[:1]
    hold_days = 0

    def _fill_buy(px: float) -> float:
        return px * (1 + (slip_bps + fee_bps) / 10000.0)

    def _fill_sell(px: float) -> float:
        return px * (1 - (slip_bps + fee_bps) / 10000.0)

    for i in range(idx0, n):
        just_bought = False

        # 값 가져오기 (iloc 금지, 배열 인덱싱)
        try:
            cl_b = float(x_sig[i - offset_cl_buy])
            ma_b = float(ma_buy_arr[i - offset_ma_buy])
            cl_s = float(x_sig[i - offset_cl_sell])
            ma_s = float(ma_sell_arr[i - offset_ma_sell])
        except Exception:
            asset_curve.append(cash + position * x_trd[i] if position else cash)
            continue

        trend_ok = True
        if (ma_s_arr is not None) and (ma_l_arr is not None):
            ms = ma_s_arr[i - offset_compare_short] if i - offset_compare_short >= 0 else np.nan
            ml = ma_l_arr[i - offset_compare_long]  if i - offset_compare_long  >= 0 else np.nan
            trend_ok = (np.isfinite(ms) and np.isfinite(ml) and ms >= ml)

        close_today = x_trd[i]
        profit_pct = ((close_today - buy_price) / buy_price * 100) if buy_price else 0.0

        # ===== 조건 계산 =====
        signal = "HOLD"
        
        if buy_operator == ">":
            buy_base = (cl_b > ma_b)
        else:
            buy_base = (cl_b < ma_b)
            
        if use_trend_in_buy:
            buy_condition = buy_base and trend_ok
        else:
            buy_condition = buy_base
            
        if sell_operator == "<":
            sell_base = (cl_s < ma_s)
        else:
            sell_base = (cl_s > ma_s)
            
        if use_trend_in_sell:
            sell_condition = sell_base and not trend_ok
        else:
            sell_condition = sell_base
            
        
        stop_hit = (stop_loss_pct > 0 and profit_pct <= -stop_loss_pct)
        take_hit = (take_profit_pct > 0 and profit_pct >= take_profit_pct)

        base_sell = (sell_condition or stop_hit or take_hit)
        can_sell = (position > 0.0) and base_sell and (hold_days >= min_hold_days)
        if stop_hit or take_hit:
            can_sell = True

        if sb == "1":
            if buy_condition and sell_condition:
                if position == 0.0:
                    fill = _fill_buy(close_today)
                    position = cash / fill; cash = 0.0
                    signal = "BUY"; buy_price = fill
                    hold_days = 0; just_bought = True
                else:
                    if hold_days >= min_hold_days:
                        fill = _fill_sell(close_today)
                        cash = position * fill; position = 0.0
                        signal = "SELL"; buy_price = None
                    else:
                        signal = "HOLD"

            elif position == 0.0 and buy_condition:
                fill = _fill_buy(close_today)
                position = cash / fill; cash = 0.0
                signal = "BUY"; buy_price = fill
                hold_days = 0; just_bought = True

            elif can_sell:
                fill = _fill_sell(close_today)
                cash = position * fill; position = 0.0
                signal = "SELL"; buy_price = None

        elif sb == "2":
            if buy_condition and sell_condition:
                if position == 0.0:
                    fill = _fill_buy(close_today)
                    position = cash / fill; cash = 0.0
                    signal = "BUY"; buy_price = fill
                    hold_days = 0; just_bought = True
                else:
                    signal = "HOLD"
            elif position == 0.0 and buy_condition:
                fill = _fill_buy(close_today)
                position = cash / fill; cash = 0.0
                signal = "BUY"; buy_price = fill
                hold_days = 0; just_bought = True
            elif can_sell:
                fill = _fill_sell(close_today)
                cash = position * fill; position = 0.0
                signal = "SELL"; buy_price = None

        else:  # '3'
            if buy_condition and sell_condition:
                if position == 0.0:
                    signal = "HOLD"
                else:
                    if hold_days >= min_hold_days:
                        fill = _fill_sell(close_today)
                        cash = position * fill; position = 0.0
                        signal = "SELL"; buy_price = None
                    else:
                        signal = "HOLD"
            elif buy_condition and position == 0.0:
                fill = _fill_buy(close_today)
                position = cash / fill; cash = 0.0
                signal = "BUY"; buy_price = fill
                hold_days = 0; just_bought = True
            elif can_sell:
                fill = _fill_sell(close_today)
                cash = position * fill; position = 0.0
                signal = "SELL"; buy_price = None

        # ✅ 체결 후 카운터 업데이트 (이중 증가 방지)
        if position > 0.0:
            if not just_bought:
                hold_days += 1
        else:
            hold_days = 0

        total = cash + (position * close_today if position > 0.0 else 0.0)
        asset_curve.append(total)

        logs.append({
            "날짜": pd.to_datetime(base["Date"].iloc[i]).strftime("%Y-%m-%d"),
            "종가": round(close_today, 2),
            "신호": signal,
            "자산": round(total),
            "매수시그널": buy_condition,
            "매도시그널": sell_condition,
            "손절발동": bool(stop_hit),
            "익절발동": bool(take_hit),
            "추세만족": bool(trend_ok),
            "매수가격비교": round(cl_b - ma_b, 6),   # (+면 종가>MA)
            "매도가격비교": round(cl_s - ma_s, 6),   # (-면 종가<MA)
            "매수이유": (f"종가({cl_b:.2f}) > MA_BUY({ma_b:.2f})" + (" + 추세필터 통과" if trend_ok else " + 추세필터 불통과")) if buy_condition else "",
            "매도이유": (f"종가({cl_s:.2f}) < MA_SELL({ma_s:.2f})") if sell_condition else "",
            "양시그널": buy_condition and sell_condition,
            "보유일": hold_days
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

    # 승률
    trade_pairs, cache_buy = [], None
    for log in logs:
        if log["신호"] == "BUY":
            cache_buy = log
        elif log["신호"] == "SELL" and cache_buy:
            trade_pairs.append((cache_buy, log))
            cache_buy = None
    wins = sum(1 for b, s in trade_pairs if s["종가"] > b["종가"])
    total_trades = len(trade_pairs)
    win_rate = round((wins / total_trades) * 100, 2) if total_trades else 0.0

    initial_cash_val = float(initial_cash)
    final_asset = float(asset_curve[-1])

        # ---- 거래당 수익률/Profit Factor 계산 추가 ----
    trade_returns = []   # 각 거래의 수익률(소수, 예: 0.0123 = 1.23%)
    gross_profit = 0.0   # 수익 거래들의 수익률 합
    gross_loss = 0.0     # 손실 거래들의 손실률 합(양수로 누적)

    for b, s in trade_pairs:
        r = (s["종가"] - b["종가"]) / b["종가"]
        trade_returns.append(r)
        if r >= 0:
            gross_profit += r
        else:
            gross_loss += (-r)

    avg_trade_return_pct = round((np.mean(trade_returns) * 100), 2) if trade_returns else 0.0
    median_trade_return_pct = round((np.median(trade_returns) * 100), 2) if trade_returns else 0.0
    profit_factor = round((gross_profit / gross_loss), 2) if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)
    
    return {
        "평균 거래당 수익률 (%)": avg_trade_return_pct,
        "수익률 (%)": round((final_asset - initial_cash_val) / initial_cash_val * 100, 2),
        "승률 (%)": win_rate,
        "MDD (%)": round(mdd, 2),
        "중앙값 거래당 수익률 (%)": median_trade_return_pct,
        "Profit Factor": profit_factor,
        "MDD 발생일": mdd_date.strftime("%Y-%m-%d"),
        "MDD 회복일": recovery_date.strftime("%Y-%m-%d") if recovery_date is not None else "미회복",
        "회복 기간 (일)": (recovery_date - mdd_date).days if recovery_date is not None else None,
        "총 매매 횟수": total_trades,
        "매매 로그": logs,
        "최종 자산": round(final_asset)
    }


# ===== Fast Random Sims =====
def run_random_simulations_fast(
    n_simulations, base, x_sig, x_trd, ma_dict_sig,
    initial_cash=5_000_000, fee_bps=25, slip_bps=0,
    randomize_sell_operator=False  # 필요 시 True로
):
    results = []
    for _ in range(n_simulations):
        ma_buy = random.choice([1, 5, 10, 15, 25])
        offset_ma_buy = random.choice([1, 5, 15, 25])
        offset_cl_buy = random.choice([1, 5, 15, 25])
        buy_operator = random.choice([">", "<"])

        ma_sell = random.choice([1, 5, 10, 15, 25])
        offset_ma_sell = random.choice([1, 5, 15, 25])
        offset_cl_sell = random.choice([1, 5, 15, 25])
        sell_operator = random.choice(["<", ">"]) 
        use_trend_in_buy = random.choice([True, False])
        use_trend_in_sell = random.choice([True, False])
        ma_compare_short = random.choice([1, 5, 15, 25])
        ma_compare_long  = ma_compare_short
        offset_compare_short = random.choice([1, 15, 25])
        offset_compare_long  = random.choice([1])

        stop_loss_pct = 0
        take_profit_pct = random.choice([0, 25, 50])

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
            buy_operator=buy_operator, sell_operator=sell_operator
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


# ✅ UI 버튼 및 시각화
if st.button("✅ 백테스트 실행"):
    # 1) 이번 실행에 필요한 MA 윈도우 풀 구성
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
        sell_operator=sell_operator
    )

    if result:
        st.subheader("📊 백테스트 결과 요약")
        summary = {k: v for k, v in result.items() if k != "매매 로그"}
        st.json(summary)

        df_log = pd.DataFrame(result["매매 로그"])
        df_log["날짜"] = pd.to_datetime(df_log["날짜"])
        df_log.set_index("날짜", inplace=True)

        # ===== 성과지표 보강 (연율화/샤프/벤치마크)
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

        # ===== 그래프 그리기 =====
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
                pnl = (r["종가"] - buy_cache["종가"]) / buy_cache["종가"] * 100
                pairs.append({
                    "진입일": buy_cache["날짜"],
                    "청산일": r["날짜"],
                    "진입가": buy_cache["종가"],
                    "청산가": r["종가"],
                    "보유일": r["보유일"],
                    "수익률(%)": round(pnl, 2),
                    "청산이유": "손절" if r["손절발동"] else ("익절" if r["익절발동"] else "규칙매도")
                })
                buy_cache = None

        if pairs:
            st.subheader("🧾 트레이드 요약")
            st.dataframe(pd.DataFrame(pairs))

        # 다운로드 버튼 (로그)
        with st.expander("🧾 매매 로그"):
            st.dataframe(df_log)
        csv = df_log.reset_index().to_csv(index=False).encode("utf-8-sig")
        st.download_button("⬇️ 백테스트 결과 다운로드 (CSV)", data=csv, file_name="backtest_result.csv", mime="text/csv")


if st.button("🧪 랜덤 전략 시뮬레이션 (100회 실행)"):
    # 랜덤 가능성 있는 MA 윈도우 풀
    ma_pool = [5, 10, 15, 25, 50]
    base, x_sig, x_trd, ma_dict_sig = prepare_base(
        signal_ticker, trade_ticker, start_date, end_date, ma_pool
    )
    if seed:
        random.seed(int(seed))
    df_sim = run_random_simulations_fast(100, base, x_sig, x_trd, ma_dict_sig)
    st.subheader("📈 랜덤 전략 시뮬레이션 결과")
    st.dataframe(df_sim.sort_values(by="수익률 (%)", ascending=False).reset_index(drop=True))

##########################랜덤ㄲ##############33

# ===== Streamlit: Random Simulator (list input) =====
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# 내부 헬퍼: 리스트 파서 (안전)
def _rs_parse_list(text: str, cast_fn=int):
    if not text:
        return []
    return [cast_fn(x.strip()) for x in str(text).split(",") if x.strip() != ""]

with st.expander("🎲 랜덤 시뮬레이터 (리스트 입력 → N회 무작위 샘플링)", expanded=False):
    st.caption("콤마로 구분해서 입력하세요. 예) 5, 15, 25")

    # --- 0) 데이터 준비 섹션: base/x_sig/x_trd 없으면 여기서 즉시 생성 ---
    import yfinance as yf

    have_all = all(name in globals() for name in ["base", "x_sig", "x_trd"])
    with st.container(border=True):
        st.markdown("**데이터 준비** (필수 객체가 없으면 여기서 생성합니다)")
        c1, c2, c3 = st.columns([1,1,1])
        with c1:
            sig_ticker = st.text_input("Signal Ticker", value="SOXL", key="rs_sig_ticker")
        with c2:
            trd_ticker = st.text_input("Trade Ticker(미입력 시 Signal과 동일)", value="", key="rs_trd_ticker")
        with c3:
            period = st.selectbox("기간", ["max","10y","5y","2y","1y","6mo","3mo","1mo"], index=2, key="rs_period")

        prep_btn = st.button("📦 데이터 준비/갱신", key="rs_prep_btn")
        if prep_btn or not have_all:
            try:
                s = yf.download(sig_ticker, period=period)
                if isinstance(s.columns, pd.MultiIndex):
                    s = s["Close"]
                else:
                    s = s[["Close"]] if "Close" in s.columns else s
                s = s.squeeze().dropna()

                if trd_ticker.strip():
                    t = yf.download(trd_ticker.strip(), period=period)
                    if isinstance(t.columns, pd.MultiIndex):
                        t = t["Close"]
                    else:
                        t = t[["Close"]] if "Close" in t.columns else t
                    t = t.squeeze().dropna()
                    # 인덱스 맞추기
                    t = t.reindex(s.index).ffill().bfill()
                else:
                    t = s.copy()

                # base / x_sig / x_trd 전역에 주입
                globals()["base"] = pd.DataFrame({"종가": s.astype(float)})
                globals()["x_sig"] = globals()["base"]["종가"]
                globals()["x_trd"] = t.astype(float).reindex(globals()["base"].index).ffill().bfill()

                st.success(f"데이터 준비 완료: Signal={sig_ticker}, Trade={trd_ticker or sig_ticker}, rows={len(globals()['base'])}")
                have_all = True
            except Exception as e:
                st.exception(e)
                have_all = False

    if not have_all:
        st.stop()

    # --- 1) 파라미터 입력 ---
    col1, col2 = st.columns(2)
    with col1:
        inp_ma_buy           = st.text_input("ma_buy 후보", "5, 15, 25", key="rs_ma_buy")
        inp_offset_ma_buy    = st.text_input("offset_ma_buy 후보", "1, 5, 25", key="rs_off_ma_buy")
        inp_ma_sell          = st.text_input("ma_sell 후보", "5, 15, 25", key="rs_ma_sell")
        inp_offset_ma_sell   = st.text_input("offset_ma_sell 후보", "1", key="rs_off_ma_sell")
        inp_offset_cl_buy    = st.text_input("offset_cl_buy 후보", "5, 15, 25", key="rs_off_cl_buy")
        inp_offset_cl_sell   = st.text_input("offset_cl_sell 후보", "1, 5", key="rs_off_cl_sell")

    with col2:
        inp_ma_cmp_s         = st.text_input("ma_compare_short 후보 (0=비활성)", "0, 5, 15, 25", key="rs_cmp_s")
        inp_ma_cmp_l         = st.text_input("ma_compare_long 후보 (0=비활성)", "0, 25", key="rs_cmp_l")
        inp_off_cmp_s        = st.text_input("offset_compare_short 후보", "1, 5, 25", key="rs_off_cmp_s")
        inp_off_cmp_l        = st.text_input("offset_compare_long 후보", "1, 5, 25", key="rs_off_cmp_l")
        inp_stop             = st.text_input("손절%(stop_loss_pct) 후보", "0.0, 5.0", key="rs_stop")
        inp_take             = st.text_input("익절%(take_profit_pct) 후보", "0.0, 10.0", key="rs_take")

    col3, col4 = st.columns(2)
    with col3:
        inp_min_hold         = st.text_input("min_hold_days 후보", "0, 3, 5", key="rs_min_hold")
        inp_fee              = st.text_input("수수료 bps 후보", "0, 5, 10", key="rs_fee")
        inp_slip             = st.text_input("슬리피지 bps 후보", "0, 5, 10", key="rs_slip")
        inp_cash             = st.text_input("초기자금 후보", "5000000", key="rs_cash")
    with col4:
        inp_behavior         = st.text_input("strategy_behavior 후보(문자열 콤마 분리)",
                                             "1. 포지션 없으면 매수 / 보유 중이면 매도",
                                             key="rs_behavior")
        n_runs               = st.number_input("반복 횟수", min_value=10, max_value=5000, value=100, step=10, key="rs_runs")
        seed_val             = st.number_input("Random Seed (선택)", value=0, step=1, key="rs_seed")
        run_btn              = st.button("🚀 랜덤 시뮬레이션 실행", key="rs_run_btn")

    # --- 2) 실행 ---
    if run_btn:
        try:
            # 후보 파싱
            ma_buy_list           = _rs_parse_list(inp_ma_buy, int)
            offset_ma_buy_list    = _rs_parse_list(inp_offset_ma_buy, int)
            ma_sell_list          = _rs_parse_list(inp_ma_sell, int)
            offset_ma_sell_list   = _rs_parse_list(inp_offset_ma_sell, int)
            offset_cl_buy_list    = _rs_parse_list(inp_offset_cl_buy, int)
            offset_cl_sell_list   = _rs_parse_list(inp_offset_cl_sell, int)

            ma_cmp_s_list         = _rs_parse_list(inp_ma_cmp_s, int)   # 0 허용
            ma_cmp_l_list         = _rs_parse_list(inp_ma_cmp_l, int)   # 0 허용
            off_cmp_s_list        = _rs_parse_list(inp_off_cmp_s, int)
            off_cmp_l_list        = _rs_parse_list(inp_off_cmp_l, int)

            stop_list             = _rs_parse_list(inp_stop, float)
            take_list             = _rs_parse_list(inp_take, float)
            min_hold_list         = _rs_parse_list(inp_min_hold, int)
            fee_list              = _rs_parse_list(inp_fee, int)
            slip_list             = _rs_parse_list(inp_slip, int)
            cash_list             = _rs_parse_list(inp_cash, int)
            behavior_list         = [s.strip() for s in inp_behavior.split(",") if s.strip()]

            # 빈 후보 방지
            required_lists = {
                "ma_buy": ma_buy_list, "offset_ma_buy": offset_ma_buy_list,
                "ma_sell": ma_sell_list, "offset_ma_sell": offset_ma_sell_list,
                "offset_cl_buy": offset_cl_buy_list, "offset_cl_sell": offset_cl_sell_list,
                "ma_compare_short": ma_cmp_s_list, "ma_compare_long": ma_cmp_l_list,
                "offset_compare_short": off_cmp_s_list, "offset_compare_long": off_cmp_l_list,
                "stop_loss_pct": stop_list, "take_profit_pct": take_list,
                "min_hold_days": min_hold_list, "fee_bps": fee_list, "slip_bps": slip_list,
                "initial_cash": cash_list, "strategy_behavior": behavior_list,
            }
            empties = [k for k, v in required_lists.items() if isinstance(v, list) and len(v) == 0]
            if empties:
                st.error(f"아래 후보 리스트가 비어 있습니다: {', '.join(empties)}")
                st.stop()

            # MA dict 1회 준비 (SECTION B의 빌더 사용 가능)
            ma_period_candidates = set(ma_buy_list + ma_sell_list + ma_cmp_s_list + ma_cmp_l_list)
            if "build_ma_dict_sig" not in globals():
                # 안전망: 간단 구현
                def build_ma_dict_sig(close_series, ma_periods: set[int]):
                    import numpy as np
                    ma_dict = {}
                    for p in sorted({int(p) for p in ma_periods if p and int(p) > 0}):
                        ma_dict[p] = close_series.rolling(p).mean().to_numpy()
                    return ma_dict

            if "종가" in base.columns:
                close_series = base["종가"]
            else:
                close_series = x_sig if hasattr(x_sig, "rolling") else pd.Series(x_sig, name="Close")
            ma_dict_sig = build_ma_dict_sig(close_series, ma_period_candidates)

            # 파라미터 dict
            param_lists = required_lists

            # 진행률 표시 + 실행 (SECTION A의 run_random_simulations 써도 OK)
            progress = st.progress(0)
            logs = st.empty()

            import random
            if seed_val:
                random.seed(int(seed_val))
            rows = []
            total = int(n_runs)

            for i in range(1, total + 1):
                picked = {k: (random.choice(v) if isinstance(v, list) and len(v) > 0 else None)
                          for k, v in param_lists.items()}
                for k in ("ma_compare_short", "ma_compare_long"):
                    if picked.get(k) == 0:
                        picked[k] = None

                res = backtest_fast(
                    base=base, x_sig=x_sig, x_trd=x_trd, ma_dict_sig=ma_dict_sig,
                    ma_buy=picked.get("ma_buy"),
                    offset_ma_buy=picked.get("offset_ma_buy"),
                    ma_sell=picked.get("ma_sell"),
                    offset_ma_sell=picked.get("offset_ma_sell"),
                    offset_cl_buy=picked.get("offset_cl_buy"),
                    offset_cl_sell=picked.get("offset_cl_sell"),
                    ma_compare_short=picked.get("ma_compare_short"),
                    ma_compare_long=picked.get("ma_compare_long"),
                    offset_compare_short=picked.get("offset_compare_short", 1),
                    offset_compare_long=picked.get("offset_compare_long", 1),
                    initial_cash=picked.get("initial_cash", 5_000_000),
                    stop_loss_pct=picked.get("stop_loss_pct", 0.0),
                    take_profit_pct=picked.get("take_profit_pct", 0.0),
                    strategy_behavior=picked.get("strategy_behavior", "1. 포지션 없으면 매수 / 보유 중이면 매도"),
                    min_hold_days=picked.get("min_hold_days", 0),
                    fee_bps=picked.get("fee_bps", 0),
                    slip_bps=picked.get("slip_bps", 0),
                )

                rows.append({
                    "run": i, **picked,
                    "최종자산": res.get("최종자산"),
                    "총수익률(%)": res.get("총수익률(%)"),
                    "연율화 수익률(%)": res.get("연율화 수익률(%)"),
                    "연율화 변동성(%)": res.get("연율화 변동성(%)"),
                    "샤프": res.get("샤프"),
                    "MDD(%)": res.get("MDD(%)") or res.get("최대낙폭(%)"),
                    "승률(%)": res.get("승률(%)"),
                    "거래횟수": res.get("거래횟수"),
                    "기간": res.get("기간"),
                })

                if i % 5 == 0 or i == total:
                    progress.progress(i/total)
                    logs.write(f"진행률: {i}/{total} 회")

            df_rand = pd.DataFrame(rows)
            sort_cols = [c for c in ["샤프", "연율화 수익률(%)", "총수익률(%)"] if c in df_rand.columns]
            if sort_cols:
                df_rand = df_rand.sort_values(by=sort_cols, ascending=[False]*len(sort_cols), na_position="last")
            best = df_rand.iloc[0].to_dict() if len(df_rand) else {}

            if len(df_rand) == 0:
                st.warning("결과가 비어 있습니다. 입력 후보를 확인하세요.")
            else:
                st.success("완료!")
                st.dataframe(df_rand.reset_index(drop=True))

                st.subheader("🏆 Top-1 결과")
                st.json(best)

                plot_col = None
                for c in ["샤프", "연율화 수익률(%)", "총수익률(%)"]:
                    if c in df_rand.columns and pd.api.types.is_numeric_dtype(df_rand[c]):
                        plot_col = c
                        break
                if plot_col:
                    fig, ax = plt.subplots()
                    ax.hist(df_rand[plot_col].dropna().to_numpy(), bins=30)
                    ax.set_title(f"{plot_col} 분포")
                    st.pyplot(fig, clear_figure=True)

        except Exception as e:
            st.exception(e)
