
# -*- coding: utf-8 -*-
"""
preset_ui_patch.py

Drop-in helper for Streamlit apps:
- Fixes preset->widget sync by driving all widget values via st.session_state.
- Provides a single entry function: `mount_preset_ui(PRESETS)`
    * Renders a "🎯 전략 프리셋 선택" selectbox.
    * Applies selected preset into st.session_state (with safe defaults).
    * Renders all strategy controls keyed to st.session_state.
    * Returns a params dict you can feed to your backtest.

USAGE (in your Streamlit app):

    from preset_ui_patch import mount_preset_ui

    params = mount_preset_ui(PRESETS)
    # ... use `params` for your backtest / charts

Notes:
- Make sure you do NOT set conflicting `value=` on widgets outside this module.
- If you had cached functions depending on these params, include the relevant keys in cache keys.
"""

from typing import Dict, Any
import streamlit as st

# ------------------------------------------------------------
# 1) Defaults & keys
# ------------------------------------------------------------
DEFAULTS: Dict[str, Any] = {
    # tickers
    "signal_ticker_input": "SOXL",
    "trade_ticker_input": "SOXL",

    # buy condition
    "ma_buy": 50,
    "offset_ma_buy": 0,
    "offset_cl_buy": 0,
    "buy_operator": ">",  # ">" or "<"

    # sell condition
    "ma_sell": 10,
    "offset_ma_sell": 0,
    "offset_cl_sell": 0,
    "sell_operator": "<",  # ">" or "<"

    # trend compare
    "use_trend_in_buy": True,
    "use_trend_in_sell": False,
    "ma_compare_short": 20,
    "offset_compare_short": 0,
    "ma_compare_long": 50,
    "offset_compare_long": 0,

    # risk controls
    "stop_loss_pct": 0.0,
    "take_profit_pct": 0.0,
    "min_hold_days": 0,

    # conflict policy
    "strategy_behavior": "1. 포지션 없으면 매수 / 보유 중이면 매도",
}

STRATEGY_BEHAVIOR_OPTIONS = [
    "1. 포지션 없으면 매수 / 보유 중이면 매도",
    "2. 포지션 없으면 매수 / 보유 중이면 HOLD",
    "3. 포지션 없으면 HOLD / 보유 중이면 매도",
]


def _ensure_defaults():
    """Initialize missing keys in st.session_state with DEFAULTS once."""
    for k, v in DEFAULTS.items():
        if k not in st.session_state:
            st.session_state[k] = v
    if "preset_name" not in st.session_state:
        st.session_state["preset_name"] = "직접 설정"


def _apply_preset_to_state(preset: Dict[str, Any] | None):
    """Write preset values into st.session_state (non-destructive for missing keys)."""
    if not preset:
        return
    for k, default_v in DEFAULTS.items():
        if k in preset:
            st.session_state[k] = preset[k]


def _on_preset_change(PRESETS: Dict[str, Dict[str, Any]]):
    name = st.session_state.get("preset_name", "직접 설정")
    preset = {} if name == "직접 설정" else PRESETS.get(name, {})
    _apply_preset_to_state(preset)
    # Force re-render so widgets immediately reflect the new state
    st.rerun()


def _read_params_from_state() -> Dict[str, Any]:
    """Collect current params from st.session_state into a plain dict."""
    params = {k: st.session_state.get(k, DEFAULTS[k]) for k in DEFAULTS.keys()}
    # normalize some types
    params["ma_buy"] = int(params["ma_buy"])
    params["ma_sell"] = int(params["ma_sell"])
    params["ma_compare_short"] = int(params["ma_compare_short"])
    params["ma_compare_long"] = int(params["ma_compare_long"])
    params["offset_ma_buy"] = int(params["offset_ma_buy"])
    params["offset_ma_sell"] = int(params["offset_ma_sell"])
    params["offset_cl_buy"] = int(params["offset_cl_buy"])
    params["offset_cl_sell"] = int(params["offset_cl_sell"])
    params["offset_compare_short"] = int(params["offset_compare_short"])
    params["offset_compare_long"] = int(params["offset_compare_long"])
    params["stop_loss_pct"] = float(params["stop_loss_pct"])
    params["take_profit_pct"] = float(params["take_profit_pct"])
    params["min_hold_days"] = int(params["min_hold_days"])
    return params


# ------------------------------------------------------------
# 2) Public entry point
# ------------------------------------------------------------
def mount_preset_ui(PRESETS: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Render the whole Preset + Controls UI and return params dict.
    Call this ONCE in your Streamlit script where you build the sidebar/body.
    """
    _ensure_defaults()

    # --- Preset selection
    colA, colB = st.columns([2, 1])
    with colA:
        st.selectbox(
            "🎯 전략 프리셋 선택",
            ["직접 설정"] + list(PRESETS.keys()),
            key="preset_name",
            on_change=_on_preset_change,
            args=(PRESETS,),
        )
    with colB:
        if st.button("프리셋 불러오기", use_container_width=True):
            _on_preset_change(PRESETS)

    # --- Tickers
    st.markdown("#### 📈 티커 설정")
    c1, c2 = st.columns(2)
    with c1:
        st.text_input("Signal Ticker", key="signal_ticker_input")
    with c2:
        st.text_input("Trade Ticker", key="trade_ticker_input")

    # --- Buy & Sell
    st.markdown("#### 🟢 매수 조건 / 🔴 매도 조건")
    c1, c2 = st.columns(2)
    with c1:
        st.number_input("MA (매수)", min_value=1, max_value=1000, step=1, key="ma_buy")
        st.number_input("MA 오프셋 (매수)", min_value=-365, max_value=365, step=1, key="offset_ma_buy")
        st.number_input("종가 오프셋 (매수)", min_value=-365, max_value=365, step=1, key="offset_cl_buy")
        st.selectbox("부호 (매수)", [">", "<"], key="buy_operator")
    with c2:
        st.number_input("MA (매도)", min_value=1, max_value=1000, step=1, key="ma_sell")
        st.number_input("MA 오프셋 (매도)", min_value=-365, max_value=365, step=1, key="offset_ma_sell")
        st.number_input("종가 오프셋 (매도)", min_value=-365, max_value=365, step=1, key="offset_cl_sell")
        st.selectbox("부호 (매도)", [">", "<"], key="sell_operator")

    # --- Trend compare
    st.markdown("#### 📐 추세 비교 (선택)")
    c1, c2 = st.columns(2)
    with c1:
        st.checkbox("매수에 추세 사용", key="use_trend_in_buy")
        st.number_input("단기 MA", min_value=1, max_value=1000, step=1, key="ma_compare_short")
        st.number_input("단기 오프셋", min_value=-365, max_value=365, step=1, key="offset_compare_short")
    with c2:
        st.checkbox("매도에 추세 사용", key="use_trend_in_sell")
        st.number_input("장기 MA", min_value=1, max_value=1000, step=1, key="ma_compare_long")
        st.number_input("장기 오프셋", min_value=-365, max_value=365, step=1, key="offset_compare_long")

    # --- Risk controls & behavior
    st.markdown("#### 🧯 리스크 / 동시 신호 처리")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.number_input("손절 (%)", min_value=0.0, max_value=100.0, step=0.1, key="stop_loss_pct")
    with c2:
        st.number_input("익절 (%)", min_value=0.0, max_value=100.0, step=0.1, key="take_profit_pct")
    with c3:
        st.number_input("최소 보유일", min_value=0, max_value=365, step=1, key="min_hold_days")

    st.selectbox("⚙️ 매수/매도 조건 동시 발생 시 행동", STRATEGY_BEHAVIOR_OPTIONS, key="strategy_behavior")

    # Read out params
    return _read_params_from_state()


# Optional: local smoke test
if __name__ == "__main__":
    st.set_page_config(page_title="Preset UI Patch – Demo", layout="wide")
    st.title("Preset UI Patch – Demo")

    # Tiny sample presets for testing
    PRESETS = {
        "무매A(예시)": {
            "signal_ticker_input": "TSLL",
            "trade_ticker_input": "TSLL",
            "ma_buy": 50,
            "offset_ma_buy": 0,
            "offset_cl_buy": 0,
            "buy_operator": ">",
            "ma_sell": 20,
            "offset_ma_sell": 0,
            "offset_cl_sell": 0,
            "sell_operator": "<",
            "use_trend_in_buy": True,
            "use_trend_in_sell": False,
            "ma_compare_short": 10,
            "offset_compare_short": 0,
            "ma_compare_long": 60,
            "offset_compare_long": 0,
            "stop_loss_pct": 10.0,
            "take_profit_pct": 30.0,
            "min_hold_days": 2,
            "strategy_behavior": "1. 포지션 없으면 매수 / 보유 중이면 매도",
        },
        "주추(예시)": {
            "signal_ticker_input": "SOXL",
            "trade_ticker_input": "SOXL",
            "ma_buy": 5, "offset_ma_buy": 0, "offset_cl_buy": 0, "buy_operator": ">",
            "ma_sell": 5, "offset_ma_sell": 0, "offset_cl_sell": 0, "sell_operator": "<",
            "use_trend_in_buy": False, "use_trend_in_sell": False,
            "ma_compare_short": 20, "offset_compare_short": 0,
            "ma_compare_long": 50, "offset_compare_long": 0,
            "stop_loss_pct": 0.0, "take_profit_pct": 0.0, "min_hold_days": 0,
            "strategy_behavior": "2. 포지션 없으면 매수 / 보유 중이면 HOLD",
        },
    }

    params = mount_preset_ui(PRESETS)
    st.divider()
    st.write("현재 파라미터:", params)
