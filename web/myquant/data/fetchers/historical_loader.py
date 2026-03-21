"""
Historical OHLCV loader using AKShare (free) with Tushare Pro as fallback.

Caching:
  All successful API fetches are saved as Parquet files under data/cache/.
  On subsequent calls the cache is checked first — if a file exists that
  covers the requested date range, it is returned immediately without any
  network call.  This allows backtests to run fully offline once data has
  been downloaded once via download_data.py.

  Cache file naming:
    data/cache/{symbol}_{interval}_{start}_{end}.parquet
    e.g. data/cache/hk00700_D1_20220101_20260318.parquet

AKShare: pip install akshare
Tushare: pip install tushare
"""
from __future__ import annotations

import hashlib
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

from myquant.config.logging_config import get_logger
from myquant.config.settings import settings
from myquant.models.bar import Bar, BarInterval

logger = get_logger(__name__)

_CACHE_DIR = Path("data/cache")


def _cache_path(symbol: str, interval: BarInterval, start: date, end: date) -> Path:
    key = f"{symbol}_{interval.value}_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}"
    return _CACHE_DIR / f"{key}.parquet"


def _symbol_to_akshare(symbol: str) -> tuple[str, str]:
    """
    Convert Tencent-style symbol to AKShare market+code.
    e.g. "sh600036" → ("600036", "sh")
         "hk00700"  → ("00700", "hk")
         "usTSLA"   → ("TSLA", "us")
    """
    if symbol.startswith("sh"):
        return symbol[2:], "sh"
    elif symbol.startswith("sz"):
        return symbol[2:], "sz"
    elif symbol.startswith("hk"):
        return symbol[2:], "hk"
    elif symbol.startswith("us"):
        return symbol[2:], "us"
    return symbol, ""


def _symbol_to_yfinance(symbol: str) -> str:
    """
    Convert Tencent-style symbol to Yahoo Finance ticker.
    e.g. "hk00700"  → "0700.HK"   (AKShare 5-digit → Yahoo 4-digit)
         "sh600036" → "600036.SS"
         "sz000858" → "000858.SZ"
         "usTSLA"   → "TSLA"
    """
    if symbol.startswith("hk"):
        # AKShare pads HK codes to 5 digits; Yahoo Finance uses 4-digit codes.
        # e.g. "00700" → int 700 → "0700.HK"
        try:
            num = int(symbol[2:])
        except ValueError:
            return ""
        return f"{num:04d}.HK"
    elif symbol.startswith("sh"):
        return f"{symbol[2:]}.SS"
    elif symbol.startswith("sz"):
        return f"{symbol[2:]}.SZ"
    elif symbol.startswith("us"):
        return symbol[2:]
    return symbol


class HistoricalLoader:
    """
    Loads historical OHLCV data into a DataFrame or list of Bar objects.

    Supports A-shares, HK stocks, and US stocks via AKShare.
    Falls back to Tushare Pro for A-shares if AKShare fails.
    """

    def __init__(self) -> None:
        self._tushare_pro = None
        if settings.TUSHARE_TOKEN:
            try:
                import tushare as ts
                ts.set_token(settings.TUSHARE_TOKEN)
                self._tushare_pro = ts.pro_api()
                logger.info("Tushare Pro initialized.")
            except Exception as exc:
                logger.warning("Tushare init failed: %s", exc)

    # ── Public API ────────────────────────────────────────────

    def load_df(
        self,
        symbol: str,
        start: Optional[date] = None,
        end: Optional[date] = None,
        interval: BarInterval = BarInterval.D1,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """
        Load OHLCV as a DataFrame indexed by datetime.
        Columns: open, high, low, close, volume, turnover

        If a local Parquet cache file exists for the exact (symbol, interval,
        start, end) key it is returned immediately without any network call.
        Pass force_refresh=True to bypass the cache and re-download.
        """
        if start is None:
            start = date.today() - timedelta(days=365)
        if end is None:
            end = date.today()

        # ── 1. Check disk cache ───────────────────────────────
        if not force_refresh:
            cached = self._read_cache(symbol, interval, start, end)
            if cached is not None:
                logger.info("Cache HIT  %s [%s → %s]", symbol, start, end)
                return cached

        # ── 2. Fetch from API ─────────────────────────────────
        code, market = _symbol_to_akshare(symbol)
        df = pd.DataFrame()
        try:
            if market in ("sh", "sz"):
                df = self._load_ashare(code, market, start, end, interval)
            elif market == "hk":
                df = self._load_hk(code, start, end)
            elif market == "us":
                df = self._load_us(code, start, end)
            else:
                raise ValueError(f"Unknown market for symbol: {symbol}")
        except Exception as exc:
            logger.warning("AKShare failed for %s (%s), trying Yahoo Finance…", symbol, exc)

        # ── 3. Yahoo Finance fallback ──────────────────────────────────────────
        if df.empty:
            try:
                df = self._load_yfinance(symbol, start, end)
            except Exception as exc:
                logger.error("Failed to load history for %s: %s", symbol, exc)
                return pd.DataFrame()

        # ── 3. Persist to cache ───────────────────────────────
        if not df.empty:
            self._write_cache(df, symbol, interval, start, end)

        return df

    def save_to_cache(
        self,
        df: pd.DataFrame,
        symbol: str,
        interval: BarInterval,
        start: date,
        end: date,
    ) -> None:
        """Manually persist a DataFrame to the cache (used by download_data.py)."""
        self._write_cache(df, symbol, interval, start, end)

    # ── Cache helpers ─────────────────────────────────────────

    def _read_cache(
        self,
        symbol: str,
        interval: BarInterval,
        start: date,
        end: date,
    ) -> Optional[pd.DataFrame]:
        path = _cache_path(symbol, interval, start, end)
        if not path.exists():
            # Also check if we have a superset file (wider date range)
            return self._read_superset_cache(symbol, interval, start, end)
        try:
            df = pd.read_parquet(path)
            return df
        except Exception as e:
            logger.debug("Cache read failed %s: %s", path, e)
            return None

    def _read_superset_cache(
        self,
        symbol: str,
        interval: BarInterval,
        start: date,
        end: date,
    ) -> Optional[pd.DataFrame]:
        """Return a slice from any cached file whose date range is a superset."""
        pattern = f"{symbol}_{interval.value}_*.parquet"
        for path in sorted(_CACHE_DIR.glob(pattern)):
            try:
                parts = path.stem.split("_")
                # stem format: {symbol}_{interval}_{start}_{end}
                # symbol may contain underscores — take last 3 parts as [interval, start, end]
                cached_start = datetime.strptime(parts[-2], "%Y%m%d").date()
                cached_end   = datetime.strptime(parts[-1], "%Y%m%d").date()
                if cached_start <= start and cached_end >= end:
                    df = pd.read_parquet(path)
                    mask = (df.index.date >= start) & (df.index.date <= end)
                    sliced = df[mask]
                    if not sliced.empty:
                        logger.info("Cache HIT (superset) %s [%s → %s]", symbol, start, end)
                        return sliced
            except Exception:
                continue
        return None

    def _write_cache(
        self,
        df: pd.DataFrame,
        symbol: str,
        interval: BarInterval,
        start: date,
        end: date,
    ) -> None:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path = _cache_path(symbol, interval, start, end)
        try:
            df.to_parquet(path)
            logger.debug("Cache WRITE %s → %s", symbol, path.name)
        except Exception as e:
            logger.debug("Cache write failed: %s", e)

    def load_bars(
        self,
        symbol: str,
        start: Optional[date] = None,
        end: Optional[date] = None,
        interval: BarInterval = BarInterval.D1,
    ) -> list[Bar]:
        """Load history as a list of Bar objects."""
        df = self.load_df(symbol, start, end, interval)
        if df.empty:
            return []
        bars: list[Bar] = []
        for ts_idx, row in df.iterrows():
            bars.append(
                Bar(
                    symbol=symbol,
                    ts=ts_idx if isinstance(ts_idx, datetime) else datetime.combine(ts_idx, datetime.min.time()),
                    interval=interval,
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=int(row.get("volume", 0)),
                    turnover=float(row.get("turnover", 0.0)),
                )
            )
        return bars

    # ── Internal loaders ──────────────────────────────────────

    def _load_ashare(
        self,
        code: str,
        market: str,
        start: date,
        end: date,
        interval: BarInterval,
    ) -> pd.DataFrame:
        try:
            import akshare as ak
            period_map = {
                BarInterval.D1: "daily",
                BarInterval.W1: "weekly",
                BarInterval.M1: "1",
                BarInterval.M5: "5",
                BarInterval.M15: "15",
                BarInterval.M30: "30",
                BarInterval.H1: "60",
            }
            period = period_map.get(interval, "daily")

            if interval in (BarInterval.D1, BarInterval.W1):
                df = ak.stock_zh_a_hist(
                    symbol=code,
                    period=period,
                    start_date=start.strftime("%Y%m%d"),
                    end_date=end.strftime("%Y%m%d"),
                    adjust="qfq",
                )
                df = df.rename(columns={
                    "日期": "date", "开盘": "open", "最高": "high",
                    "最低": "low", "收盘": "close", "成交量": "volume",
                    "成交额": "turnover",
                })
                df["date"] = pd.to_datetime(df["date"])
                df = df.set_index("date")
            else:
                df = ak.stock_zh_a_hist_min_em(
                    symbol=code,
                    period=period,
                    start_date=start.strftime("%Y-%m-%d %H:%M:%S"),
                    end_date=end.strftime("%Y-%m-%d %H:%M:%S"),
                    adjust="qfq",
                )
                df = df.rename(columns={
                    "时间": "date", "开盘": "open", "最高": "high",
                    "最低": "low", "收盘": "close", "成交量": "volume",
                    "成交额": "turnover",
                })
                df["date"] = pd.to_datetime(df["date"])
                df = df.set_index("date")

            return df[["open", "high", "low", "close", "volume", "turnover"]]
        except Exception as exc:
            logger.warning("AKShare A-share load failed: %s. Trying Tushare.", exc)
            return self._load_ashare_tushare(code, market, start, end)

    def _load_ashare_tushare(
        self, code: str, market: str, start: date, end: date
    ) -> pd.DataFrame:
        if self._tushare_pro is None:
            raise RuntimeError("Tushare Pro not available.")
        ts_code = f"{code}.{'SH' if market == 'sh' else 'SZ'}"
        df = self._tushare_pro.daily(
            ts_code=ts_code,
            start_date=start.strftime("%Y%m%d"),
            end_date=end.strftime("%Y%m%d"),
        )
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.sort_values("trade_date")
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        df = df.set_index("trade_date")
        df = df.rename(columns={"vol": "volume", "amount": "turnover"})
        return df[["open", "high", "low", "close", "volume", "turnover"]]

    def _load_hk(self, code: str, start: date, end: date) -> pd.DataFrame:
        import akshare as ak
        df = ak.stock_hk_hist(
            symbol=code,
            period="daily",
            start_date=start.strftime("%Y%m%d"),
            end_date=end.strftime("%Y%m%d"),
            adjust="qfq",
        )
        df = df.rename(columns={
            "日期": "date", "开盘": "open", "最高": "high",
            "最低": "low", "收盘": "close", "成交量": "volume", "成交额": "turnover",
        })
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
        return df[["open", "high", "low", "close", "volume", "turnover"]]

    def _load_us(self, code: str, start: date, end: date) -> pd.DataFrame:
        import akshare as ak
        df = ak.stock_us_hist(
            symbol=code,
            period="daily",
            start_date=start.strftime("%Y%m%d"),
            end_date=end.strftime("%Y%m%d"),
            adjust="qfq",
        )
        df = df.rename(columns={
            "日期": "date", "开盘": "open", "最高": "high",
            "最低": "low", "收盘": "close", "成交量": "volume", "成交额": "turnover",
        })
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
        return df[["open", "high", "low", "close", "volume", "turnover"]]

    def _load_yfinance(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        """
        Yahoo Finance fallback — globally accessible, no rate limits.
        Converts Tencent-style symbols automatically via _symbol_to_yfinance().
        """
        try:
            import yfinance as yf
        except ImportError:
            logger.error("yfinance not installed; run: pip install yfinance")
            return pd.DataFrame()

        yf_symbol = _symbol_to_yfinance(symbol)
        if not yf_symbol:
            logger.warning("Cannot map %s to a Yahoo Finance ticker", symbol)
            return pd.DataFrame()

        try:
            ticker = yf.Ticker(yf_symbol)
            df = ticker.history(
                start=start.strftime("%Y-%m-%d"),
                end=(end + timedelta(days=1)).strftime("%Y-%m-%d"),
                auto_adjust=True,
            )
        except Exception as exc:
            logger.error("yfinance download error for %s (%s): %s", symbol, yf_symbol, exc)
            return pd.DataFrame()

        if df is None or df.empty:
            logger.warning("yfinance returned no data for %s (%s)", symbol, yf_symbol)
            return pd.DataFrame()

        # Drop timezone so the index is tz-naive (consistent with AKShare output)
        dt_idx: pd.DatetimeIndex = df.index  # type: ignore[assignment]
        if dt_idx.tz is not None:
            df.index = dt_idx.tz_convert(None)
        df.index.name = "date"

        df = df.rename(columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        })
        df["turnover"] = (df["close"] * df["volume"]).fillna(0.0)

        logger.info("yfinance loaded %d bars for %s (%s)", len(df), symbol, yf_symbol)
        return df[["open", "high", "low", "close", "volume", "turnover"]]
