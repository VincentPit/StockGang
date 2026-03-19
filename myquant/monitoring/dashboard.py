"""
Streamlit Dashboard — live monitoring of portfolio, positions, and signals.

Run with:
    streamlit run monitoring/dashboard.py
"""
from __future__ import annotations

import json
import time
from datetime import datetime

import pandas as pd
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MyQuant Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _load_snapshot() -> dict:
    """Load latest portfolio snapshot from Redis or a local JSON file (dev)."""
    try:
        import redis
        r = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)
        raw = r.get("state:portfolio_snapshot")
        if raw:
            return json.loads(raw)
    except Exception:
        pass
    # Fallback: demo data for local dev
    return {
        "nav": 1_050_230.50,
        "cash": 400_000.00,
        "market_value": 650_230.50,
        "total_pnl": 50_230.50,
        "total_pnl_pct": "+5.02%",
        "unrealized_pnl": 28_000.00,
        "realized_pnl": 22_230.50,
        "current_drawdown": "-1.20%",
        "max_drawdown": "-3.50%",
        "total_trades": 47,
        "open_positions": 5,
        "positions": {
            "hk00700": {"qty": 200, "cost": 320.00, "price": 342.00, "upnl": 4400.00, "pct": "+6.88%"},
            "sh600036": {"qty": 1000, "cost": 42.50, "price": 44.10, "upnl": 1600.00, "pct": "+3.76%"},
            "usTSLA": {"qty": 50, "cost": 195.00, "price": 204.50, "upnl": 475.00, "pct": "+4.87%"},
            "sh600519": {"qty": 100, "cost": 1650.00, "price": 1720.00, "upnl": 7000.00, "pct": "+4.24%"},
            "hk09988": {"qty": 300, "cost": 76.00, "price": 82.50, "upnl": 1950.00, "pct": "+8.55%"},
        },
        "timestamp": datetime.now().isoformat(),
    }


def main() -> None:
    st.title("📈 MyQuant — Automated Trading Dashboard")
    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    snapshot = _load_snapshot()

    # ── Top KPI Row ───────────────────────────────────────────
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("NAV", f"¥{snapshot['nav']:,.0f}", snapshot['total_pnl_pct'])
    col2.metric("Cash", f"¥{snapshot['cash']:,.0f}")
    col3.metric("Unrealized PnL", f"¥{snapshot['unrealized_pnl']:+,.0f}")
    col4.metric("Realized PnL", f"¥{snapshot['realized_pnl']:+,.0f}")
    col5.metric("Drawdown", snapshot['current_drawdown'])

    st.divider()

    # ── Positions Table ───────────────────────────────────────
    st.subheader("📊 Open Positions")
    positions = snapshot.get("positions", {})
    if positions:
        rows = []
        for sym, p in positions.items():
            rows.append({
                "Symbol": sym,
                "Qty": p["qty"],
                "Avg Cost": f"{p['cost']:.4f}",
                "Last Price": f"{p['price']:.4f}",
                "Unrealized PnL": f"¥{p['upnl']:+,.2f}",
                "Return": p["pct"],
            })
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No open positions.")

    st.divider()

    # ── Stats Row ─────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Trades", snapshot.get("total_trades", 0))
    col2.metric("Max Drawdown", snapshot.get("max_drawdown", "0%"))
    col3.metric("Open Positions", snapshot.get("open_positions", 0))

    # ── Auto-refresh every 5 seconds ─────────────────────────
    st.markdown("---")
    if st.button("🔄 Refresh"):
        st.rerun()

    # Auto-refresh
    time.sleep(5)
    st.rerun()


if __name__ == "__main__":
    main()
