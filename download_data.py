"""
download_data.py — Pre-download and cache historical bar data.

Run this ONCE to pull years of OHLCV data from AKShare and persist it
as Parquet files in data/cache/.  After that, all backtests run fully
offline — no network calls, instant loads.

Usage:
    python download_data.py                    # default: 3 years, all symbols
    python download_data.py --years 5          # 5-year history
    python download_data.py --symbols hk00700  # single symbol

Retry logic:
    Each symbol retries up to MAX_RETRIES times with exponential back-off
    (2s → 4s → 8s) to handle AKShare rate-limiting gracefully.

Output:
    data/cache/{symbol}_{interval}_{start}_{end}.parquet
"""
import argparse
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from myquant.config.logging_config import setup_logging, get_logger
from myquant.data.fetchers.historical_loader import HistoricalLoader, _cache_path
from myquant.models.bar import BarInterval

logger = get_logger(__name__)

# ── Default universe ──────────────────────────────────────────────────────────
DEFAULT_SYMBOLS = [
    # HK
    "hk00700",   # Tencent
    "hk03690",   # Meituan
    "hk09988",   # Alibaba HK
    "hk02318",   # Ping An Insurance
    "hk01299",   # AIA
    "hk00005",   # HSBC
    # A-share
    "sh600036",  # CMB
    "sh601318",  # Ping An A
    "sh600519",  # Kweichow Moutai
    "sh601166",  # Industrial Bank
    "sz000858",  # Wuliangye
    "sz300750",  # CATL
    # US
    "usTSLA",
    "usAAPL",
    "usNVDA",
    "usBABA",
    "usMSFT",
]

MAX_RETRIES  = 4
RETRY_DELAYS = [3, 6, 12, 24]   # seconds between retries (exponential)
INTER_SYMBOL_DELAY = 2.0         # pause between each symbol (rate-limit friendly)


def download_symbol(
    loader: HistoricalLoader,
    symbol: str,
    start: datetime,
    end: datetime,
    interval: BarInterval,
    force_refresh: bool,
) -> bool:
    """
    Download and cache one symbol.  Returns True on success.
    """
    cache = _cache_path(symbol, interval, start.date(), end.date())

    if not force_refresh and cache.exists():
        logger.info("  ✓ Already cached: %s", cache.name)
        return True

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logger.info(
                "  → Downloading %s [%s → %s] (attempt %d/%d)…",
                symbol, start.date(), end.date(), attempt, MAX_RETRIES,
            )
            df = loader.load_df(
                symbol,
                start=start.date(),
                end=end.date(),
                interval=interval,
                force_refresh=True,   # bypass cache to actually hit the API
            )
            if df.empty:
                logger.warning("  ✗ Empty result for %s", symbol)
                return False

            logger.info("  ✓ %s: %d bars saved → %s", symbol, len(df), cache.name)
            return True

        except Exception as exc:
            logger.warning("  ✗ Attempt %d failed for %s: %s", attempt, symbol, exc)
            if attempt < MAX_RETRIES:
                delay = RETRY_DELAYS[attempt - 1]
                logger.info("    Waiting %ds before retry…", delay)
                time.sleep(delay)

    logger.error("  ✗ All %d attempts failed for %s — skipping.", MAX_RETRIES, symbol)
    return False


def main() -> None:
    setup_logging()

    parser = argparse.ArgumentParser(description="Download and cache historical OHLCV data.")
    parser.add_argument("--symbols",  nargs="+",  default=DEFAULT_SYMBOLS,
                        help="Space-separated list of symbols (default: full universe)")
    parser.add_argument("--years",    type=float, default=3.0,
                        help="Number of years of history to download (default: 3)")
    parser.add_argument("--interval", default="D1",
                        choices=["D1", "W1", "M5", "M15", "M30", "H1"],
                        help="Bar interval (default: D1)")
    parser.add_argument("--force",    action="store_true",
                        help="Re-download even if cache already exists")
    args = parser.parse_args()

    interval_map = {
        "D1": BarInterval.D1, "W1": BarInterval.W1,
        "M5": BarInterval.M5, "M15": BarInterval.M15,
        "M30": BarInterval.M30, "H1": BarInterval.H1,
    }
    interval = interval_map[args.interval]

    end   = datetime.now()
    start = end - timedelta(days=int(365 * args.years))

    logger.info("=" * 60)
    logger.info("Download parameters:")
    logger.info("  Symbols  : %d symbols", len(args.symbols))
    logger.info("  Range    : %s → %s  (%.1f years)", start.date(), end.date(), args.years)
    logger.info("  Interval : %s", args.interval)
    logger.info("  Cache dir: %s", Path("data/cache").resolve())
    logger.info("=" * 60)

    Path("data/cache").mkdir(parents=True, exist_ok=True)
    loader = HistoricalLoader()

    ok_count   = 0
    fail_count = 0
    fail_list: list[str] = []

    for i, symbol in enumerate(args.symbols, 1):
        logger.info("[%d/%d] %s", i, len(args.symbols), symbol)
        ok = download_symbol(loader, symbol, start, end, interval, args.force)
        if ok:
            ok_count += 1
        else:
            fail_count += 1
            fail_list.append(symbol)

        # Polite delay between symbols to avoid rate limiting
        if i < len(args.symbols):
            time.sleep(INTER_SYMBOL_DELAY)

    logger.info("=" * 60)
    logger.info("Download complete: %d succeeded, %d failed", ok_count, fail_count)
    if fail_list:
        logger.warning("Failed symbols: %s", ", ".join(fail_list))
        logger.warning("Re-run with --force or check network connectivity.")
    else:
        logger.info("All symbols downloaded. Run: python backtest_run.py")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
