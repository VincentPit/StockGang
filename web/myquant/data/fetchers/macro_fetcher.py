"""
Macro Fetcher — collects top-level macroeconomic indicators.

Indicators collected:
  ┌─────────────────┬──────────────────────────────────────────────────────┐
  │ china_pmi_mfg   │ NBS Manufacturing PMI  (>50 = expansion)            │
  │ china_pmi_svc   │ NBS Services PMI                                     │
  │ china_cpi_yoy   │ CPI year-on-year %                                   │
  │ china_ppi_yoy   │ PPI year-on-year %                                   │
  │ china_lpr_1y    │ 1-year Loan Prime Rate %                             │
  │ china_lpr_5y    │ 5-year Loan Prime Rate %                             │
  │ usdcny          │ USD/CNY mid-rate                                     │
  │ us_10y_yield    │ US 10-year Treasury yield %                          │
  │ vix             │ CBOE VIX (fear gauge)                                │
  └─────────────────┴──────────────────────────────────────────────────────┘

All fetches use AKShare (free).  Results are disk-cached for `cache_ttl_hours`
so repeated backtests / engine restarts don't hammer the API.

Regime logic (MacroSnapshot.regime):
  RISK_ON   → VIX < 18,  PMI > 51,  US10Y < 5.0%
  RISK_OFF  → VIX > 30   OR PMI < 49  OR US10Y > 5.5%
  NEUTRAL   → everything else
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

_CACHE_FILE = Path("data/macro_cache.json")


# ── Data Model ───────────────────────────────────────────────────────────────

@dataclass
class MacroSnapshot:
    ts: datetime = field(default_factory=datetime.now)

    # ── China ──────────────────────────────────────────────────
    china_pmi_mfg: float = 50.0       # NBS manufacturing PMI
    china_pmi_svc: float = 52.0       # NBS services PMI
    china_cpi_yoy: float  = 2.0       # % YoY
    china_ppi_yoy: float  = 0.0       # % YoY
    china_lpr_1y:  float  = 3.45      # 1-year LPR %
    china_lpr_5y:  float  = 3.95      # 5-year LPR %

    # ── FX ────────────────────────────────────────────────────
    usdcny: float = 7.20              # USD/CNY spot

    # ── Global ────────────────────────────────────────────────
    us_10y_yield: float = 4.50        # US 10Y treasury %
    vix:          float = 18.0        # CBOE VIX

    # ── Derived ───────────────────────────────────────────────
    @property
    def regime(self) -> str:
        """Macro regime string: RISK_ON | NEUTRAL | RISK_OFF"""
        if self.vix > 30 or self.china_pmi_mfg < 49.0 or self.us_10y_yield > 5.5:
            return "RISK_OFF"
        if (self.vix < 18 and self.china_pmi_mfg > 51.0
                and self.us_10y_yield < 5.0 and self.china_cpi_yoy < 4.0):
            return "RISK_ON"
        return "NEUTRAL"

    @property
    def signal_multiplier(self) -> float:
        """Scale signal confidence by macro regime."""
        return {"RISK_ON": 1.20, "NEUTRAL": 1.00, "RISK_OFF": 0.50}[self.regime]

    @property
    def cny_depreciation_pressure(self) -> bool:
        """True when USD/CNY is above 7.25 (CNY under pressure)."""
        return self.usdcny > 7.25

    def describe(self) -> str:
        return (
            f"Macro regime: {self.regime}  "
            f"| PMI={self.china_pmi_mfg}  CPI={self.china_cpi_yoy}%  "
            f"LPR={self.china_lpr_1y}%  USD/CNY={self.usdcny:.4f}  "
            f"US10Y={self.us_10y_yield}%  VIX={self.vix}"
        )

    def to_dict(self) -> dict:
        d = asdict(self)
        d["ts"]     = self.ts.isoformat()
        d["regime"] = self.regime
        return d


# ── Fetcher ──────────────────────────────────────────────────────────────────

class MacroFetcher:
    """
    Fetches macroeconomic indicators via AKShare.
    Gracefully falls back to cached / default values on any API failure.
    """

    def __init__(self, cache_ttl_hours: int = 24) -> None:
        self._cache_ttl = timedelta(hours=cache_ttl_hours)
        self._snapshot: Optional[MacroSnapshot] = None

    # ── Public API ────────────────────────────────────────────

    def fetch(self, use_cache: bool = True) -> MacroSnapshot:
        """Return macro snapshot, using in-memory → disk → live cascade."""
        if use_cache and self._snapshot is not None:
            if datetime.now() - self._snapshot.ts < self._cache_ttl:
                return self._snapshot

        # Try disk cache
        if use_cache:
            cached = self._load_disk_cache()
            if cached is not None:
                self._snapshot = cached
                return cached

        # Fetch live
        snap = MacroSnapshot()
        snap.ts = datetime.now()
        self._fetch_china_pmi(snap)
        self._fetch_china_cpi_ppi(snap)
        self._fetch_china_lpr(snap)
        self._fetch_usdcny(snap)
        self._fetch_us_10y(snap)

        self._snapshot = snap
        self._save_disk_cache(snap)
        logger.info(snap.describe())
        return snap

    # ── Individual fetchers ───────────────────────────────────

    def _fetch_china_pmi(self, snap: MacroSnapshot) -> None:
        try:
            import akshare as ak
            df = ak.macro_china_pmi_monthly()
            if df is None or df.empty:
                return
            row = df.iloc[-1]
            for col in df.columns:
                col_l = col.lower()
                val = pd.to_numeric(row[col], errors="coerce")
                if pd.isna(val):
                    continue
                if "制造" in col or "mfg" in col_l or (
                    "综合" not in col and "服务" not in col and 40 < val < 65
                ):
                    snap.china_pmi_mfg = float(val)
                elif "服务" in col or "service" in col_l or "non" in col_l:
                    snap.china_pmi_svc = float(val)
        except Exception as e:
            logger.debug("macro: PMI fetch failed: %s", e)

    def _fetch_china_cpi_ppi(self, snap: MacroSnapshot) -> None:
        try:
            import akshare as ak
            df = ak.macro_china_cpi_monthly()
            if df is not None and not df.empty:
                row = df.iloc[-1]
                for col in df.columns:
                    if "同比" in col or "yoy" in col.lower():
                        val = pd.to_numeric(row[col], errors="coerce")
                        if not pd.isna(val):
                            snap.china_cpi_yoy = float(val)
                            break
        except Exception as e:
            logger.debug("macro: CPI fetch failed: %s", e)

        try:
            import akshare as ak
            df = ak.macro_china_ppi_monthly()
            if df is not None and not df.empty:
                row = df.iloc[-1]
                for col in df.columns:
                    if "同比" in col or "yoy" in col.lower():
                        val = pd.to_numeric(row[col], errors="coerce")
                        if not pd.isna(val):
                            snap.china_ppi_yoy = float(val)
                            break
        except Exception as e:
            logger.debug("macro: PPI fetch failed: %s", e)

    def _fetch_china_lpr(self, snap: MacroSnapshot) -> None:
        try:
            import akshare as ak
            df = ak.macro_china_lpr()
            if df is None or df.empty:
                return
            row = df.iloc[-1]
            for col in df.columns:
                val = pd.to_numeric(row[col], errors="coerce")
                if pd.isna(val):
                    continue
                if "1年" in col or "1y" in col.lower():
                    snap.china_lpr_1y = float(val)
                elif "5年" in col or "5y" in col.lower():
                    snap.china_lpr_5y = float(val)
        except Exception as e:
            logger.debug("macro: LPR fetch failed: %s", e)

    def _fetch_usdcny(self, snap: MacroSnapshot) -> None:
        try:
            import akshare as ak
            df = ak.currency_boc_sina(symbol="USDCNY")
            if df is None or df.empty:
                return
            row = df.iloc[-1]
            for col in df.columns:
                if "中间价" in col or "mid" in col.lower() or "close" in col.lower():
                    val = pd.to_numeric(row[col], errors="coerce")
                    if not pd.isna(val) and 5 < val < 10:
                        snap.usdcny = float(val)
                        break
        except Exception as e:
            logger.debug("macro: USD/CNY fetch failed: %s", e)

    def _fetch_us_10y(self, snap: MacroSnapshot) -> None:
        try:
            import akshare as ak
            df = ak.bond_zh_us_rate(start_date="20240101")
            if df is None or df.empty:
                return
            row = df.iloc[-1]
            for col in df.columns:
                if "10" in col and ("美" in col or "us" in col.lower()):
                    val = pd.to_numeric(row[col], errors="coerce")
                    if not pd.isna(val) and 0 < val < 20:
                        snap.us_10y_yield = float(val)
                        break
        except Exception as e:
            logger.debug("macro: US 10Y fetch failed: %s", e)

    # ── Cache helpers ─────────────────────────────────────────

    def _load_disk_cache(self) -> Optional[MacroSnapshot]:
        if not _CACHE_FILE.exists():
            return None
        try:
            with open(_CACHE_FILE) as f:
                d = json.load(f)
            cached_ts = datetime.fromisoformat(d["ts"])
            if datetime.now() - cached_ts < self._cache_ttl:
                d.pop("regime", None)
                d["ts"] = cached_ts
                known = set(MacroSnapshot.__dataclass_fields__)
                return MacroSnapshot(**{k: v for k, v in d.items() if k in known})
        except Exception as e:
            logger.debug("macro: disk cache load failed: %s", e)
        return None

    def _save_disk_cache(self, snap: MacroSnapshot) -> None:
        try:
            _CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(_CACHE_FILE, "w") as f:
                json.dump(snap.to_dict(), f, indent=2, default=str)
        except Exception as e:
            logger.debug("macro: disk cache save failed: %s", e)
