"""
Fundamental Fetcher — per-stock fundamental indicators.

Data collected (where available via AKShare):
  ┌──────────────────┬────────────────────────────────────────────────────┐
  │ pe_ttm           │ Price-to-Earnings (TTM)                            │
  │ pb               │ Price-to-Book                                      │
  │ ps_ttm           │ Price-to-Sales (TTM)                               │
  │ roe              │ Return on Equity %                                 │
  │ revenue_growth   │ Revenue YoY growth %                               │
  │ net_margin       │ Net profit margin %                                │
  │ dividend_yield   │ Dividend yield %                                   │
  └──────────────────┴────────────────────────────────────────────────────┘

Supports A-share (sh/sz prefix) and HK stocks (hk prefix).
US stocks fall back to neutral defaults (no free fundamental API available).

Results disk-cached per symbol for `cache_ttl_hours` (default: 24h).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

from myquant.config.logging_config import get_logger

logger = get_logger(__name__)

_CACHE_FILE = Path("data/fundamental_cache.json")


# ── Data Model ───────────────────────────────────────────────────────────────

@dataclass
class FundamentalSnapshot:
    symbol: str
    ts: datetime = field(default_factory=datetime.now)

    pe_ttm:         float = 20.0
    pb:             float = 2.0
    ps_ttm:         float = 3.0
    roe:            float = 10.0    # %
    revenue_growth: float = 0.0     # % YoY
    net_margin:     float = 10.0    # %
    dividend_yield: float = 0.0     # %

    # ── Derived scores ────────────────────────────────────────

    @property
    def value_score(self) -> float:
        """
        Composite value score 0–100.
        Weights: 40% PE, 40% ROE, 20% PB (lower PE/PB and higher ROE = better).
        """
        pe_score  = max(0.0, min(100.0, 100.0 - self.pe_ttm * 2.0))
        roe_score = max(0.0, min(100.0, self.roe  * 3.0))
        pb_score  = max(0.0, min(100.0, 100.0 - self.pb   * 8.0))
        return pe_score * 0.40 + roe_score * 0.40 + pb_score * 0.20

    @property
    def growth_score(self) -> float:
        """Composite growth score 0–100."""
        rev_score    = max(0.0, min(100.0, self.revenue_growth * 2.0))
        margin_score = max(0.0, min(100.0, self.net_margin     * 3.0))
        return rev_score * 0.60 + margin_score * 0.40

    @property
    def quality_score(self) -> float:
        """Blend of value and growth."""
        return self.value_score * 0.5 + self.growth_score * 0.5

    def to_dict(self) -> dict:
        d = asdict(self)
        d["ts"] = self.ts.isoformat()
        return d


# ── Fetcher ──────────────────────────────────────────────────────────────────

class FundamentalFetcher:
    """
    Fetches per-stock fundamental data from AKShare.
    Gracefully degrades to neutral defaults on any failure.
    """

    def __init__(self, cache_ttl_hours: int = 24) -> None:
        self._cache_ttl = timedelta(hours=cache_ttl_hours)
        self._cache: dict[str, FundamentalSnapshot] = {}
        self._load_disk_cache()

    # ── Public API ────────────────────────────────────────────

    def fetch(self, symbol: str) -> FundamentalSnapshot:
        """Return fundamental snapshot for a symbol."""
        cached = self._cache.get(symbol)
        if cached and datetime.now() - cached.ts < self._cache_ttl:
            return cached

        snap     = FundamentalSnapshot(symbol=symbol)
        snap.ts  = datetime.now()

        if symbol.startswith(("sh", "sz")):
            self._fetch_a_share(snap)
        elif symbol.startswith("hk"):
            self._fetch_hk(snap)
        # US stocks: leave at neutral defaults

        self._cache[symbol] = snap
        self._save_disk_cache()
        return snap

    def fetch_batch(self, symbols: list[str]) -> dict[str, FundamentalSnapshot]:
        return {sym: self.fetch(sym) for sym in symbols}

    # ── A-share ───────────────────────────────────────────────

    def _fetch_a_share(self, snap: FundamentalSnapshot) -> None:
        code = snap.symbol[2:]  # strip "sh"/"sz"

        # P/E TTM
        try:
            import akshare as ak
            df = ak.stock_a_lg_indicator(symbol=code, indicator="市盈率TTM")
            if df is not None and not df.empty:
                val = pd.to_numeric(df.iloc[-1].iloc[-1], errors="coerce")
                if not pd.isna(val) and val > 0:
                    snap.pe_ttm = float(val)
        except Exception as e:
            logger.debug("fundamental: A-share PE failed %s: %s", snap.symbol, e)

        # P/B
        try:
            import akshare as ak
            df = ak.stock_a_lg_indicator(symbol=code, indicator="市净率")
            if df is not None and not df.empty:
                val = pd.to_numeric(df.iloc[-1].iloc[-1], errors="coerce")
                if not pd.isna(val) and val > 0:
                    snap.pb = float(val)
        except Exception as e:
            logger.debug("fundamental: A-share PB failed %s: %s", snap.symbol, e)

        # ROE + net margin from financial analysis
        try:
            import akshare as ak
            df = ak.stock_financial_analysis_indicator(
                symbol=code, start_year=str(datetime.now().year - 2)
            )
            if df is not None and not df.empty:
                row = df.iloc[-1]
                for col in df.columns:
                    col_l = col.lower()
                    val = pd.to_numeric(row.get(col, None), errors="coerce")
                    if pd.isna(val):
                        continue
                    if "净资产收益率" in col or "roe" in col_l:
                        snap.roe = float(val)
                    elif "净利润率" in col or "net_margin" in col_l or "net margin" in col_l:
                        snap.net_margin = float(val)
        except Exception as e:
            logger.debug("fundamental: A-share ROE failed %s: %s", snap.symbol, e)

    # ── HK stocks ────────────────────────────────────────────

    def _fetch_hk(self, snap: FundamentalSnapshot) -> None:
        # "hk00700" → "700"
        code = snap.symbol[2:].lstrip("0") or "0"

        try:
            import akshare as ak
            df = ak.stock_hk_valuation_baidu(symbol=code, indicator="市盈率TTM")
            if df is not None and not df.empty:
                val = pd.to_numeric(df.iloc[-1].iloc[-1], errors="coerce")
                if not pd.isna(val) and val > 0:
                    snap.pe_ttm = float(val)
        except Exception as e:
            logger.debug("fundamental: HK PE failed %s: %s", snap.symbol, e)

        try:
            import akshare as ak
            df = ak.stock_hk_valuation_baidu(symbol=code, indicator="市净率")
            if df is not None and not df.empty:
                val = pd.to_numeric(df.iloc[-1].iloc[-1], errors="coerce")
                if not pd.isna(val) and val > 0:
                    snap.pb = float(val)
        except Exception as e:
            logger.debug("fundamental: HK PB failed %s: %s", snap.symbol, e)

    # ── Disk cache ────────────────────────────────────────────

    def _load_disk_cache(self) -> None:
        if not _CACHE_FILE.exists():
            return
        try:
            with open(_CACHE_FILE) as f:
                raw = json.load(f)
            known = set(FundamentalSnapshot.__dataclass_fields__)
            for sym, d in raw.items():
                d["ts"] = datetime.fromisoformat(d["ts"])
                self._cache[sym] = FundamentalSnapshot(
                    **{k: v for k, v in d.items() if k in known}
                )
        except Exception as e:
            logger.debug("fundamental: disk cache load failed: %s", e)

    def _save_disk_cache(self) -> None:
        try:
            _CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
            raw = {sym: snap.to_dict() for sym, snap in self._cache.items()}
            with open(_CACHE_FILE, "w") as f:
                json.dump(raw, f, indent=2, default=str)
        except Exception as e:
            logger.debug("fundamental: disk cache save failed: %s", e)
