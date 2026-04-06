"""
Factor Attribution — decompose returns into factor exposures and alpha.

Implements:
  - Returns-based style analysis (Sharpe 1992)
  - Factor contribution breakdown (momentum, value, volatility, size)
  - Strategy-level attribution (which strategy contributed most to P&L)
  - Sector/symbol attribution
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from myquant.config.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class FactorExposure:
    """Single factor exposure and contribution."""
    factor_name: str
    exposure: float         # beta to the factor
    contribution: float     # annualised return contribution
    t_stat: float = 0.0     # statistical significance


@dataclass
class StrategyContribution:
    """P&L contribution from a single strategy."""
    strategy_id: str
    gross_pnl: float
    num_trades: int
    win_rate: float
    avg_pnl_per_trade: float
    pnl_pct_of_total: float  # what fraction of total P&L came from this strategy


@dataclass
class SymbolContribution:
    """P&L contribution from a single symbol."""
    symbol: str
    gross_pnl: float
    num_trades: int
    win_rate: float
    avg_pnl_per_trade: float


@dataclass
class AttributionReport:
    """Complete attribution analysis."""
    # Factor exposures
    factor_exposures: list[FactorExposure] = field(default_factory=list)
    alpha_annualised: float = 0.0     # unexplained return (alpha)
    r_squared: float = 0.0            # how much variance factors explain

    # Strategy attribution
    strategy_contributions: list[StrategyContribution] = field(default_factory=list)

    # Symbol attribution (top 10 best + worst)
    top_symbols: list[SymbolContribution] = field(default_factory=list)
    bottom_symbols: list[SymbolContribution] = field(default_factory=list)

    # Sector attribution
    sector_pnl: dict[str, float] = field(default_factory=dict)

    def summary_dict(self) -> dict:
        return {
            "alpha_annualised": round(self.alpha_annualised, 4),
            "r_squared": round(self.r_squared, 3),
            "factor_exposures": [
                {
                    "factor": f.factor_name,
                    "exposure": round(f.exposure, 3),
                    "contribution": round(f.contribution, 4),
                    "t_stat": round(f.t_stat, 2),
                }
                for f in self.factor_exposures
            ],
            "strategy_contributions": [
                {
                    "strategy": s.strategy_id,
                    "gross_pnl": round(s.gross_pnl, 2),
                    "num_trades": s.num_trades,
                    "win_rate": round(s.win_rate, 3),
                    "pnl_pct": round(s.pnl_pct_of_total, 3),
                }
                for s in self.strategy_contributions
            ],
            "top_symbols": [
                {"symbol": s.symbol, "pnl": round(s.gross_pnl, 2), "trades": s.num_trades}
                for s in self.top_symbols[:5]
            ],
            "bottom_symbols": [
                {"symbol": s.symbol, "pnl": round(s.gross_pnl, 2), "trades": s.num_trades}
                for s in self.bottom_symbols[:5]
            ],
            "sector_pnl": {k: round(v, 2) for k, v in self.sector_pnl.items()},
        }


class FactorAttribution:
    """
    Decomposes portfolio returns into factor contributions.

    Usage:
        attrib = FactorAttribution()
        report = attrib.analyze(
            trades_df=backtest_result.trades,
            nav_series=backtest_result.nav_series,
            sector_map={"sh600519": "Consumer Staples", ...},
        )
    """

    def analyze(
        self,
        trades_df: pd.DataFrame,
        nav_series: Optional[pd.Series] = None,
        sector_map: Optional[dict[str, str]] = None,
        factor_returns: Optional[pd.DataFrame] = None,
    ) -> AttributionReport:
        """
        Generate a full attribution report.

        Args:
            trades_df: Trade log with columns: time, symbol, side, qty, price, commission, strategy
            nav_series: DatetimeIndex → NAV (for factor regression)
            sector_map: symbol → sector name
            factor_returns: DatetimeIndex DataFrame with factor return columns
        """
        report = AttributionReport()

        if trades_df is None or trades_df.empty:
            return report

        # ── Strategy attribution ──
        report.strategy_contributions = self._strategy_attribution(trades_df)

        # ── Symbol attribution ──
        sym_contribs = self._symbol_attribution(trades_df)
        sym_contribs.sort(key=lambda s: s.gross_pnl, reverse=True)
        report.top_symbols = [s for s in sym_contribs if s.gross_pnl > 0][:10]
        report.bottom_symbols = sorted(
            [s for s in sym_contribs if s.gross_pnl < 0],
            key=lambda s: s.gross_pnl,
        )[:10]

        # ── Sector attribution ──
        if sector_map:
            report.sector_pnl = self._sector_attribution(trades_df, sector_map)

        # ── Factor regression ──
        if nav_series is not None and factor_returns is not None and len(nav_series) > 30:
            factor_result = self._factor_regression(nav_series, factor_returns)
            report.factor_exposures = factor_result["exposures"]
            report.alpha_annualised = factor_result["alpha"]
            report.r_squared = factor_result["r_squared"]

        return report

    def _strategy_attribution(self, trades_df: pd.DataFrame) -> list[StrategyContribution]:
        """Break down P&L by strategy."""
        results = []
        total_pnl = 0.0

        if "strategy" not in trades_df.columns:
            return results

        # Compute round-trip P&L per strategy
        strategy_pnls: dict[str, list[float]] = {}

        for strat, grp in trades_df.groupby("strategy"):
            pnls = self._compute_round_trip_pnls(grp)
            strategy_pnls[strat] = pnls
            total_pnl += sum(pnls)

        for strat, pnls in strategy_pnls.items():
            if not pnls:
                continue
            wins = [p for p in pnls if p > 0]
            gross = sum(pnls)
            results.append(StrategyContribution(
                strategy_id=strat,
                gross_pnl=gross,
                num_trades=len(pnls),
                win_rate=len(wins) / len(pnls) if pnls else 0.0,
                avg_pnl_per_trade=gross / len(pnls),
                pnl_pct_of_total=(gross / total_pnl) if total_pnl != 0 else 0.0,
            ))

        results.sort(key=lambda s: s.gross_pnl, reverse=True)
        return results

    def _symbol_attribution(self, trades_df: pd.DataFrame) -> list[SymbolContribution]:
        """Break down P&L by symbol."""
        results = []

        for sym, grp in trades_df.groupby("symbol"):
            pnls = self._compute_round_trip_pnls(grp)
            if not pnls:
                continue
            wins = [p for p in pnls if p > 0]
            results.append(SymbolContribution(
                symbol=sym,
                gross_pnl=sum(pnls),
                num_trades=len(pnls),
                win_rate=len(wins) / len(pnls) if pnls else 0.0,
                avg_pnl_per_trade=sum(pnls) / len(pnls),
            ))

        return results

    def _sector_attribution(
        self, trades_df: pd.DataFrame, sector_map: dict[str, str]
    ) -> dict[str, float]:
        """Aggregate P&L by sector."""
        sector_pnl: dict[str, float] = {}
        sym_contribs = self._symbol_attribution(trades_df)
        for sc in sym_contribs:
            sector = sector_map.get(sc.symbol, "Unknown")
            sector_pnl[sector] = sector_pnl.get(sector, 0.0) + sc.gross_pnl
        return dict(sorted(sector_pnl.items(), key=lambda kv: kv[1], reverse=True))

    def _factor_regression(
        self, nav_series: pd.Series, factor_returns: pd.DataFrame
    ) -> dict:
        """OLS regression of portfolio returns on factor returns."""
        port_rets = nav_series.pct_change().dropna()
        common = port_rets.index.intersection(factor_returns.index)

        if len(common) < 30:
            return {"exposures": [], "alpha": 0.0, "r_squared": 0.0}

        y = port_rets.loc[common].values
        X = factor_returns.loc[common].values
        # Add intercept
        X_with_const = np.column_stack([np.ones(len(X)), X])

        try:
            # OLS: beta = (X'X)^-1 X'y
            betas = np.linalg.lstsq(X_with_const, y, rcond=None)[0]
            y_hat = X_with_const @ betas
            residuals = y - y_hat

            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

            # Standard errors
            n = len(y)
            k = X_with_const.shape[1]
            sigma2 = ss_res / max(n - k, 1)
            try:
                var_betas = sigma2 * np.diag(np.linalg.inv(X_with_const.T @ X_with_const))
                se_betas = np.sqrt(np.maximum(var_betas, 0))
            except np.linalg.LinAlgError:
                se_betas = np.ones(k)

            alpha = float(betas[0]) * 252  # annualise daily alpha

            exposures = []
            for i, col in enumerate(factor_returns.columns):
                beta_i = float(betas[i + 1])
                t_stat = beta_i / se_betas[i + 1] if se_betas[i + 1] > 0 else 0.0
                contribution = beta_i * float(np.mean(factor_returns[col].loc[common])) * 252
                exposures.append(FactorExposure(
                    factor_name=col,
                    exposure=beta_i,
                    contribution=contribution,
                    t_stat=t_stat,
                ))

            return {"exposures": exposures, "alpha": alpha, "r_squared": r_squared}

        except Exception as e:
            logger.warning("Factor regression failed: %s", e)
            return {"exposures": [], "alpha": 0.0, "r_squared": 0.0}

    @staticmethod
    def _compute_round_trip_pnls(grp: pd.DataFrame) -> list[float]:
        """FIFO match buys→sells and return list of round-trip P&Ls."""
        grp = grp.sort_values("time").reset_index(drop=True)
        buy_queue: list[tuple[float, float]] = []
        pnls: list[float] = []

        for _, row in grp.iterrows():
            qty_r = float(row["qty"])
            price_r = float(row["price"])
            comm_r = float(row.get("commission", 0))

            if row["side"] == "BUY":
                buy_queue.append((qty_r, price_r + comm_r / max(qty_r, 1)))
            elif row["side"] == "SELL" and buy_queue:
                remaining = qty_r
                trip_pnl = -comm_r
                while remaining > 0 and buy_queue:
                    bqty, bprice = buy_queue[0]
                    matched = min(remaining, bqty)
                    trip_pnl += matched * (price_r - bprice)
                    remaining -= matched
                    if matched < bqty:
                        buy_queue[0] = (bqty - matched, bprice)
                    else:
                        buy_queue.pop(0)
                pnls.append(trip_pnl)

        return pnls
