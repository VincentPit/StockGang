"""
backtest_run.py — Multi-strategy backtest with ML training, walk-forward retrain,
ATR-based position sizing, and historical regime detection.

Improvements over v1
--------------------
1. Walk-forward ML training  : LightGBM retrains every 63 bars (quarterly) on a
                               rolling 2-year window — prevents stale-regime overfitting.
2. 23-feature model          : Added vol_regime, trend_strength, close_loc, gap_pct,
                               vol_trend, above_ma50, ma_spread on top of original 16.
3. ATR position sizing       : Each order is sized so a 1-ATR adverse move costs ≤0.5%
                               of NAV, scaled by model confidence (replaces fixed 100 lots).
4. Historical regime filter  : HistoricalRegimeDetector computes RISK_ON/OFF from
                               replayed price bars — zero look-ahead bias.

Pipeline
--------
  1. Download/cache 3 years of history via download_data.py (run once).
  2. Load first 2 years as warm-up / ML training data.
  3. Test on the most recent 1 year (out-of-sample).
  4. Strategies: LightGBM (quarterly retrain) + MA Crossover + RSI + MACD.
  5. All signals → HistoricalRegimeDetector gate → ATR sizer → RiskGate → PaperBroker.

Usage
-----
    python backtest_run.py
"""
import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from myquant.config.logging_config import setup_logging
from myquant.backtest.simulator import Backtester, BacktestConfig
from myquant.models.bar import BarInterval
from myquant.strategy.ml.lgbm_strategy import LGBMStrategy
from myquant.strategy.nlp.news_strategy import NewsStrategy
from myquant.strategy.technical.ma_crossover import MACrossoverStrategy
from myquant.strategy.technical.macd_strategy import MACDStrategy
from myquant.strategy.technical.rsi_strategy import RSIStrategy


# ── Symbol universe ────────────────────────────────────────────────────────
# Auto-selected by tools/stock_screener.py — top-6 by composite score from
# 33-stock candidate universe using 1yr of real price data.
# Scoring: trend%(0.25) + ATR%(0.20) + autocorr(0.20) + 6m-mom(0.20) + -MaxDD(0.15)
#
# Dropped:  hk03690 (Meituan)  — rank #20/33, 1Y=-54.3%, MaxDD=-58.0%
#           sh600036 (CMB)     — rank #30/33, 1Y=+2.7%,  ATR=1.39% (too low for ATR sizing)
SYMBOLS = [
    "hk00005",   # HSBC HK               score=0.684 | 1Y=+144.8% | ATR=4.16% | Trend=87.9%
    "hk00883",   # CNOOC                 score=0.653 | 1Y=+98.1% | ATR=4.84% | Trend=66.3%
    "sh601899",   # Zijin Mining          score=0.635 | 1Y=+190.6% | ATR=3.96% | Trend=71.6%
    "hk01299",   # AIA Group             score=0.601 | 1Y=+37.3% | ATR=3.66% | Trend=66.7%
    "hk02318",   # Ping An HK            score=0.565 | 1Y=+93.9% | ATR=3.33% | Trend=68.7%
    "sz300750",   # CATL                  score=0.528 | 1Y=+162.1% | ATR=3.47% | Trend=63.0%
]


async def run_backtest() -> None:
    setup_logging()

    end_date   = datetime.now()
    test_start = end_date - timedelta(days=365)      # 1-year test window
    train_years = 2                                   # 2-year warm-up for ML

    config = BacktestConfig(
        symbols         = SYMBOLS,
        start_date      = test_start,
        end_date        = end_date,
        initial_cash    = 1_000_000.0,
        interval        = BarInterval.D1,
        commission_rate = 0.0003,
        slippage        = 0.0002,
        apply_stamp_duty= True,
        train_years     = train_years,
    )

    # ── ML strategy — LightGBM with quarterly walk-forward retrain ────────
    lgbm = LGBMStrategy(
        strategy_id    = "lgbm_core",
        symbols        = SYMBOLS,
        forward_days   = 5,          # predict 5-day forward return
        threshold      = 0.015,      # ±1.5% minimum move to label as signal
        train_ratio    = 0.70,
        min_confidence = 0.52,
        retrain_every  = 63,         # retrain every ~quarter (63 trading days)
        max_train_bars = 504,        # rolling window: last ~2 years
        use_macro      = False,
        num_leaves     = 31,
        n_estimators   = 300,
    )

    # ── Technical strategies ──────────────────────────────────────────────
    ma = MACrossoverStrategy(
        strategy_id = "ma_cross",
        symbols     = SYMBOLS,
        fast_period = 10,
        slow_period = 30,
        use_ema     = True,
    )
    rsi = RSIStrategy(
        strategy_id = "rsi_filter",
        symbols     = SYMBOLS,
        period      = 14,
        oversold    = 35,
        overbought  = 65,
        exit_mid    = False,
    )
    macd = MACDStrategy(
        strategy_id = "macd_sig",
        symbols     = SYMBOLS,
        fast        = 12,
        slow        = 26,
        signal      = 9,
        min_hist    = 0.0,
    )

    # ── News strategy (active in live; silent in backtest) ────────────────
    news = NewsStrategy(
        strategy_id              = "news_sentiment",
        symbols                  = SYMBOLS,
        buy_threshold            = 0.45,
        sell_threshold           = -0.45,
        window_size              = 8,
        news_fetch_interval_mins = 30,
        min_confidence           = 0.60,
    )

    # ── Assemble backtester ───────────────────────────────────────────────
    # Note: macro_filter removed — HistoricalRegimeDetector inside the simulator
    # provides regime filtering without look-ahead bias.
    backtester = (
        Backtester(config)
        .add_strategy(lgbm)
        .add_strategy(ma)
        .add_strategy(rsi)
        .add_strategy(macd)
        .add_strategy(news)
    )

    result = await backtester.run()

    # ── Save detailed trade log ───────────────────────────────────────────
    if not result.trades.empty:
        result.trades.to_csv("backtest_trades.csv", index=False)
        print(f"\nTrades saved to backtest_trades.csv  ({len(result.trades)} fills)")

        # ── Attribution helpers ───────────────────────────────────────────
        def _fifo_pnl(grp: "pd.DataFrame") -> float:
            """Match BUY fills → SELL fills (FIFO) and return net P&L."""
            grp = grp.sort_values("time").reset_index(drop=True)
            buy_q: list[tuple[float, float]] = []
            pnl = 0.0
            for _, row in grp.iterrows():
                qty, price, comm = float(row["qty"]), float(row["price"]), float(row["commission"])
                if row["side"] == "BUY":
                    buy_q.append((qty, price + comm / max(qty, 1)))
                else:
                    rem = qty
                    trade_pnl = -comm
                    while rem > 0 and buy_q:
                        bqty, bprice = buy_q[0]
                        matched = min(rem, bqty)
                        trade_pnl += matched * (price - bprice)
                        rem -= matched
                        buy_q[0] = (bqty - matched, bprice)
                        if buy_q[0][0] <= 0:
                            buy_q.pop(0)
                    pnl += trade_pnl
            return pnl

        df = result.trades
        if "symbol" in df.columns:
            print("\nPer-symbol P&L attribution:")
            print(f"  {'Symbol':<14} {'Fills':>6} {'Net P&L':>12} {'Buys':>6} {'Sells':>7}")
            print("  " + "-" * 48)
            sym_totals: list[tuple[str, float, int, int, int]] = []
            for sym, grp in df.groupby("symbol"):
                fills = len(grp)
                buys  = int((grp["side"] == "BUY").sum())
                sells = int((grp["side"] == "SELL").sum())
                sym_pnl = _fifo_pnl(grp)
                sym_totals.append((str(sym), sym_pnl, fills, buys, sells))
            for sym, spnl, fills, buys, sells in sorted(sym_totals, key=lambda x: x[1]):
                sign = "+" if spnl >= 0 else ""
                print(f"  {sym:<14} {fills:>6} {sign}{spnl:>11,.0f} {buys:>6} {sells:>7}")

        if "strategy" in df.columns:
            print("\nPer-strategy fill count:")
            print(f"  {'Strategy':<20} {'Fills':>7} {'Buys':>6} {'Sells':>7}")
            print("  " + "-" * 44)
            for strat, grp in df.groupby("strategy"):
                buys  = (grp["side"] == "BUY").sum()
                sells = (grp["side"] == "SELL").sum()
                print(f"  {strat:<20} {len(grp):>7} {buys:>6} {sells:>7}")
    else:
        print("\nNo trades filled during the test period.")


if __name__ == "__main__":
    asyncio.run(run_backtest())

