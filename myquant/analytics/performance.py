"""
Performance Analyzer — institutional-grade risk and return metrics.

Computes every metric a fund manager or allocator would expect:
  - Annualised return, volatility, Sharpe, Sortino, Calmar
  - Rolling Sharpe, rolling drawdown
  - Max drawdown duration (days)
  - Monthly / yearly return heatmap data
  - Win/loss streaks, expectancy, payoff ratio
  - Tail risk: CVaR (Expected Shortfall), skewness, kurtosis
  - Ulcer Index, Gain-to-Pain ratio
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class DrawdownInfo:
    """Describes a single drawdown episode."""
    peak_date: datetime
    trough_date: datetime
    recovery_date: Optional[datetime]
    max_dd_pct: float
    duration_days: int          # peak to recovery (or to end if unrecovered)
    underwater_days: int        # peak to trough


@dataclass
class TradeStats:
    """Round-trip trade analysis."""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    avg_hold_bars: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0
    payoff_ratio: float = 0.0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    avg_win_hold: float = 0.0
    avg_loss_hold: float = 0.0


@dataclass
class PerformanceReport:
    """Full performance tear-sheet."""
    # ── Return metrics ──
    total_return: float = 0.0
    annualised_return: float = 0.0
    annualised_volatility: float = 0.0
    downside_deviation: float = 0.0

    # ── Risk-adjusted ──
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    omega_ratio: float = 0.0
    gain_to_pain: float = 0.0
    ulcer_index: float = 0.0

    # ── Drawdown ──
    max_drawdown: float = 0.0
    max_drawdown_duration_days: int = 0
    avg_drawdown: float = 0.0
    drawdown_episodes: list[DrawdownInfo] = field(default_factory=list)
    current_drawdown: float = 0.0

    # ── Tail risk ──
    var_95: float = 0.0         # Value at Risk (95%)
    var_99: float = 0.0         # Value at Risk (99%)
    cvar_95: float = 0.0        # Conditional VaR / Expected Shortfall (95%)
    skewness: float = 0.0
    kurtosis: float = 0.0

    # ── Rolling metrics ──
    rolling_sharpe_60d: list[tuple[datetime, float]] = field(default_factory=list)
    rolling_sharpe_252d: list[tuple[datetime, float]] = field(default_factory=list)
    rolling_volatility_60d: list[tuple[datetime, float]] = field(default_factory=list)

    # ── Calendar returns ──
    monthly_returns: dict[str, float] = field(default_factory=dict)   # "2024-01" → 0.023
    yearly_returns: dict[int, float] = field(default_factory=dict)    # 2024 → 0.15

    # ── Trade analysis ──
    trade_stats: TradeStats = field(default_factory=TradeStats)

    # ── Benchmark comparison ──
    alpha: float = 0.0
    beta: float = 0.0
    information_ratio: float = 0.0
    tracking_error: float = 0.0

    def summary_dict(self) -> dict:
        """Return a JSON-serialisable summary for the API."""
        return {
            "total_return": round(self.total_return, 4),
            "annualised_return": round(self.annualised_return, 4),
            "annualised_volatility": round(self.annualised_volatility, 4),
            "sharpe_ratio": round(self.sharpe_ratio, 3),
            "sortino_ratio": round(self.sortino_ratio, 3),
            "calmar_ratio": round(self.calmar_ratio, 3),
            "omega_ratio": round(self.omega_ratio, 3),
            "max_drawdown": round(self.max_drawdown, 4),
            "max_drawdown_duration_days": self.max_drawdown_duration_days,
            "current_drawdown": round(self.current_drawdown, 4),
            "var_95": round(self.var_95, 4),
            "cvar_95": round(self.cvar_95, 4),
            "skewness": round(self.skewness, 3),
            "kurtosis": round(self.kurtosis, 3),
            "ulcer_index": round(self.ulcer_index, 4),
            "gain_to_pain": round(self.gain_to_pain, 3),
            "alpha": round(self.alpha, 4),
            "beta": round(self.beta, 3),
            "information_ratio": round(self.information_ratio, 3),
            "tracking_error": round(self.tracking_error, 4),
            "trade_stats": {
                "total_trades": self.trade_stats.total_trades,
                "win_rate": round(self.trade_stats.win_rate, 3),
                "profit_factor": round(self.trade_stats.profit_factor, 3),
                "expectancy": round(self.trade_stats.expectancy, 2),
                "payoff_ratio": round(self.trade_stats.payoff_ratio, 3),
                "max_consecutive_wins": self.trade_stats.max_consecutive_wins,
                "max_consecutive_losses": self.trade_stats.max_consecutive_losses,
                "avg_hold_bars": round(self.trade_stats.avg_hold_bars, 1),
            },
            "monthly_returns": {k: round(v, 4) for k, v in self.monthly_returns.items()},
            "yearly_returns": {k: round(v, 4) for k, v in self.yearly_returns.items()},
        }


class PerformanceAnalyzer:
    """
    Computes a full performance report from a NAV time-series and optional trade log.

    Usage:
        analyzer = PerformanceAnalyzer(risk_free_rate=0.02)
        report = analyzer.analyze(nav_series, trades_df, benchmark_series)
    """

    def __init__(self, risk_free_rate: float = 0.02, periods_per_year: int = 252) -> None:
        self._rf = risk_free_rate
        self._ann = periods_per_year

    def analyze(
        self,
        nav_series: pd.Series,
        trades_df: Optional[pd.DataFrame] = None,
        benchmark_series: Optional[pd.Series] = None,
    ) -> PerformanceReport:
        """
        Generate a full performance report.

        Args:
            nav_series: DatetimeIndex → NAV values.
            trades_df: Optional trade log with columns: time, symbol, side, qty, price, commission
            benchmark_series: Optional benchmark NAV for alpha/beta calculation.
        """
        report = PerformanceReport()

        if nav_series is None or len(nav_series) < 2:
            return report

        # Ensure sorted
        nav = nav_series.sort_index()
        returns = nav.pct_change().dropna()

        if len(returns) < 1:
            return report

        # ── Core return metrics ──
        report.total_return = (nav.iloc[-1] / nav.iloc[0]) - 1.0
        n_years = len(returns) / self._ann
        if n_years > 0:
            report.annualised_return = (1 + report.total_return) ** (1 / n_years) - 1
        report.annualised_volatility = float(returns.std() * np.sqrt(self._ann))

        # ── Downside deviation (MAR = risk-free rate) ──
        daily_rf = self._rf / self._ann
        downside = returns[returns < daily_rf] - daily_rf
        if len(downside) > 0:
            report.downside_deviation = float(np.sqrt(np.mean(downside ** 2)) * np.sqrt(self._ann))

        # ── Risk-adjusted ratios ──
        if report.annualised_volatility > 0:
            report.sharpe_ratio = (report.annualised_return - self._rf) / report.annualised_volatility

        if report.downside_deviation > 0:
            report.sortino_ratio = (report.annualised_return - self._rf) / report.downside_deviation

        # ── Drawdown analysis ──
        dd_info = self._compute_drawdowns(nav)
        report.max_drawdown = dd_info["max_dd"]
        report.max_drawdown_duration_days = dd_info["max_dd_duration"]
        report.avg_drawdown = dd_info["avg_dd"]
        report.drawdown_episodes = dd_info["episodes"]
        report.current_drawdown = dd_info["current_dd"]

        if report.max_drawdown < 0:
            report.calmar_ratio = report.annualised_return / abs(report.max_drawdown)

        # ── Omega ratio (threshold = 0) ──
        excess = returns - daily_rf
        gains = excess[excess > 0].sum()
        losses = abs(excess[excess < 0].sum())
        report.omega_ratio = (gains / losses) if losses > 0 else float("inf")

        # ── Gain-to-Pain ──
        total_gains = returns[returns > 0].sum()
        total_losses = abs(returns[returns < 0].sum())
        report.gain_to_pain = (total_gains / total_losses) if total_losses > 0 else float("inf")

        # ── Ulcer Index ──
        report.ulcer_index = self._ulcer_index(nav)

        # ── Tail risk ──
        returns_arr = returns.values
        report.var_95 = float(np.percentile(returns_arr, 5))
        report.var_99 = float(np.percentile(returns_arr, 1))
        tail_returns = returns_arr[returns_arr <= np.percentile(returns_arr, 5)]
        report.cvar_95 = float(np.mean(tail_returns)) if len(tail_returns) > 0 else report.var_95
        report.skewness = float(pd.Series(returns_arr).skew())
        report.kurtosis = float(pd.Series(returns_arr).kurtosis())

        # ── Rolling metrics ──
        report.rolling_sharpe_60d = self._rolling_sharpe(returns, window=60)
        report.rolling_sharpe_252d = self._rolling_sharpe(returns, window=252)
        report.rolling_volatility_60d = [
            (dt, float(v))
            for dt, v in (returns.rolling(60).std() * np.sqrt(self._ann)).dropna().items()
        ]

        # ── Calendar returns ──
        report.monthly_returns = self._monthly_returns(nav)
        report.yearly_returns = self._yearly_returns(nav)

        # ── Trade analysis ──
        if trades_df is not None and not trades_df.empty:
            report.trade_stats = self._analyze_trades(trades_df)

        # ── Benchmark comparison ──
        if benchmark_series is not None and len(benchmark_series) >= 2:
            bench = benchmark_series.sort_index()
            bench_rets = bench.pct_change().dropna()
            # Align dates
            common = returns.index.intersection(bench_rets.index)
            if len(common) > 10:
                r = returns.loc[common].values
                b = bench_rets.loc[common].values
                cov_matrix = np.cov(r, b)
                if cov_matrix[1, 1] > 0:
                    report.beta = cov_matrix[0, 1] / cov_matrix[1, 1]
                    report.alpha = (report.annualised_return -
                                    self._rf - report.beta * (
                                        float(np.mean(b)) * self._ann - self._rf
                                    ))
                active_rets = r - b
                te = float(np.std(active_rets) * np.sqrt(self._ann))
                report.tracking_error = te
                if te > 0:
                    report.information_ratio = float(np.mean(active_rets) * self._ann / te)

        return report

    # ── Private helpers ──

    def _compute_drawdowns(self, nav: pd.Series) -> dict:
        """Compute all drawdown metrics."""
        running_max = nav.cummax()
        dd_series = (nav - running_max) / running_max

        max_dd = float(dd_series.min())
        current_dd = float(dd_series.iloc[-1])

        # Find drawdown episodes
        episodes: list[DrawdownInfo] = []
        in_drawdown = False
        peak_date = nav.index[0]
        trough_date = peak_date
        trough_val = 0.0

        for i in range(len(nav)):
            dt = nav.index[i]
            dd_val = dd_series.iloc[i]

            if dd_val < 0 and not in_drawdown:
                in_drawdown = True
                peak_date = nav.index[max(0, i - 1)]
                trough_date = dt
                trough_val = dd_val
            elif dd_val < trough_val and in_drawdown:
                trough_date = dt
                trough_val = dd_val
            elif dd_val >= 0 and in_drawdown:
                in_drawdown = False
                recovery_date = dt
                duration = (recovery_date - peak_date).days
                underwater = (trough_date - peak_date).days
                episodes.append(DrawdownInfo(
                    peak_date=peak_date,
                    trough_date=trough_date,
                    recovery_date=recovery_date,
                    max_dd_pct=trough_val,
                    duration_days=duration,
                    underwater_days=underwater,
                ))

        # Handle ongoing drawdown
        if in_drawdown:
            episodes.append(DrawdownInfo(
                peak_date=peak_date,
                trough_date=trough_date,
                recovery_date=None,
                max_dd_pct=trough_val,
                duration_days=(nav.index[-1] - peak_date).days,
                underwater_days=(trough_date - peak_date).days,
            ))

        max_dd_duration = max((e.duration_days for e in episodes), default=0)
        avg_dd = float(np.mean([e.max_dd_pct for e in episodes])) if episodes else 0.0

        return {
            "max_dd": max_dd,
            "max_dd_duration": max_dd_duration,
            "avg_dd": avg_dd,
            "current_dd": current_dd,
            "episodes": sorted(episodes, key=lambda e: e.max_dd_pct)[:10],  # top 10 worst
        }

    def _ulcer_index(self, nav: pd.Series) -> float:
        """Ulcer Index: RMS of percentage drawdown — penalises depth and duration."""
        running_max = nav.cummax()
        dd_pct = ((nav - running_max) / running_max) * 100
        return float(np.sqrt(np.mean(dd_pct ** 2)))

    def _rolling_sharpe(
        self, returns: pd.Series, window: int
    ) -> list[tuple[datetime, float]]:
        """Rolling annualised Sharpe ratio."""
        daily_rf = self._rf / self._ann
        excess = returns - daily_rf
        rolling_mean = excess.rolling(window).mean()
        rolling_std = returns.rolling(window).std()
        rolling = (rolling_mean / rolling_std) * np.sqrt(self._ann)
        return [(dt, float(v)) for dt, v in rolling.dropna().items()]

    def _monthly_returns(self, nav: pd.Series) -> dict[str, float]:
        """Monthly return breakdown."""
        monthly = nav.resample("M").last()
        rets = monthly.pct_change().dropna()
        return {dt.strftime("%Y-%m"): float(v) for dt, v in rets.items()}

    def _yearly_returns(self, nav: pd.Series) -> dict[int, float]:
        """Yearly return breakdown."""
        yearly = nav.resample("Y").last()
        rets = yearly.pct_change().dropna()
        return {dt.year: float(v) for dt, v in rets.items()}

    def _analyze_trades(self, trades_df: pd.DataFrame) -> TradeStats:
        """Compute round-trip trade statistics."""
        stats = TradeStats()

        if trades_df.empty:
            return stats

        # Match buys to sells (FIFO per symbol)
        pnls: list[float] = []
        hold_bars: list[int] = []
        win_holds: list[int] = []
        loss_holds: list[int] = []

        for sym, grp in trades_df.groupby("symbol"):
            grp = grp.sort_values("time").reset_index(drop=True)
            buy_queue: list[tuple[float, float, int]] = []  # (qty, price, bar_idx)

            for idx, row in grp.iterrows():
                qty_r = float(row["qty"])
                price_r = float(row["price"])
                comm_r = float(row.get("commission", 0))

                if row["side"] == "BUY":
                    buy_queue.append((qty_r, price_r + comm_r / max(qty_r, 1), int(idx)))
                elif row["side"] == "SELL" and buy_queue:
                    remaining = qty_r
                    trip_pnl = -comm_r
                    earliest_buy_idx = buy_queue[0][2] if buy_queue else int(idx)

                    while remaining > 0 and buy_queue:
                        bqty, bprice, bidx = buy_queue[0]
                        matched = min(remaining, bqty)
                        trip_pnl += matched * (price_r - bprice)
                        remaining -= matched
                        if matched < bqty:
                            buy_queue[0] = (bqty - matched, bprice, bidx)
                        else:
                            buy_queue.pop(0)

                    pnls.append(trip_pnl)
                    bars_held = int(idx) - earliest_buy_idx
                    hold_bars.append(bars_held)
                    if trip_pnl > 0:
                        win_holds.append(bars_held)
                    else:
                        loss_holds.append(bars_held)

        if not pnls:
            return stats

        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]

        stats.total_trades = len(pnls)
        stats.winning_trades = len(wins)
        stats.losing_trades = len(losses)
        stats.win_rate = len(wins) / len(pnls)
        stats.avg_win = sum(wins) / len(wins) if wins else 0.0
        stats.avg_loss = sum(losses) / len(losses) if losses else 0.0
        stats.largest_win = max(wins) if wins else 0.0
        stats.largest_loss = min(losses) if losses else 0.0
        stats.avg_hold_bars = sum(hold_bars) / len(hold_bars) if hold_bars else 0.0
        stats.avg_win_hold = sum(win_holds) / len(win_holds) if win_holds else 0.0
        stats.avg_loss_hold = sum(loss_holds) / len(loss_holds) if loss_holds else 0.0

        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))
        stats.profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")
        stats.expectancy = sum(pnls) / len(pnls)
        stats.payoff_ratio = (stats.avg_win / abs(stats.avg_loss)) if stats.avg_loss != 0 else float("inf")

        # Consecutive win/loss streaks
        stats.max_consecutive_wins = self._max_streak(pnls, positive=True)
        stats.max_consecutive_losses = self._max_streak(pnls, positive=False)

        return stats

    @staticmethod
    def _max_streak(pnls: list[float], positive: bool) -> int:
        """Count max consecutive wins or losses."""
        max_s = current = 0
        for p in pnls:
            if (positive and p > 0) or (not positive and p < 0):
                current += 1
                max_s = max(max_s, current)
            else:
                current = 0
        return max_s
