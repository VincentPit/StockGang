"""
data/fetchers/universe_fetcher.py — Live A-share investable universe.

Fetches CSI index constituents from akshare, converts codes to our internal
sym/yf_ticker format, and caches the result on disk (refreshed once per day).

Supported index codes (passed to akshare.index_stock_cons):
    "000300"  CSI 300   — ~300 large-cap blue chips
    "000905"  CSI 500   — ~500 mid-cap stocks
    "000852"  CSI 1000  — ~1000 small-cap stocks

Example
-------
>>> from myquant.data.fetchers.universe_fetcher import fetch_universe
>>> stocks = fetch_universe(indices=["000300", "000905"])
>>> stocks[0]
{'sym': 'sh600036', 'yf_ticker': '600036.SS', 'name': '招商银行'}
"""
from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path

_log = logging.getLogger(__name__)

_CACHE_DIR  = Path(__file__).parent.parent / "cache"
_CACHE_FILE = _CACHE_DIR / "universe_cache.json"

# ---------------------------------------------------------------------------
# Thin fallback — used when akshare is unreachable at startup
# (15 well-known SH/SZ A-shares covering multiple sectors)
# ---------------------------------------------------------------------------
_FALLBACK: list[dict] = [
    {"sym": "sz300059", "yf_ticker": "300059.SZ", "name": "东方财富"},
    {"sym": "sz300750", "yf_ticker": "300750.SZ", "name": "宁德时代"},
    {"sym": "sz000858", "yf_ticker": "000858.SZ", "name": "五粮液"},
    {"sym": "sz000333", "yf_ticker": "000333.SZ", "name": "美的集团"},
    {"sym": "sz002594", "yf_ticker": "002594.SZ", "name": "比亚迪"},
    {"sym": "sh601318", "yf_ticker": "601318.SS", "name": "中国平安"},
    {"sym": "sh600036", "yf_ticker": "600036.SS", "name": "招商银行"},
    {"sym": "sh600900", "yf_ticker": "600900.SS", "name": "长江电力"},
    {"sym": "sh601899", "yf_ticker": "601899.SS", "name": "紫金矿业"},
    {"sym": "sh601088", "yf_ticker": "601088.SS", "name": "中国神华"},
    {"sym": "sz002415", "yf_ticker": "002415.SZ", "name": "海康威视"},
    {"sym": "sz000001", "yf_ticker": "000001.SZ", "name": "平安银行"},
    {"sym": "sh600887", "yf_ticker": "600887.SS", "name": "伊利股份"},
    {"sym": "sz002714", "yf_ticker": "002714.SZ", "name": "牧原股份"},
    {"sym": "sz000100", "yf_ticker": "000100.SZ", "name": "TCL科技"},
]

# Human-readable names for the supported index codes
INDEX_NAMES: dict[str, str] = {
    "000300": "CSI 300",
    "000905": "CSI 500",
    "000852": "CSI 1000",
}


def _code_to_sym_yf(code: str) -> tuple[str, str]:
    """
    Convert a 6-digit akshare stock code to (sym, yf_ticker).

    Rules (Shanghai Stock Exchange codes start with 6 or 9; everything else
    is Shenzhen — including STAR Market 688xxx and ChiNext 3xxxxx):
        "600036" → ("sh600036", "600036.SS")
        "000858" → ("sz000858", "000858.SZ")
        "688981" → ("sh688981", "688981.SS")   STAR Market
        "300750" → ("sz300750", "300750.SZ")   ChiNext
    """
    code = code.strip().zfill(6)
    if code.startswith(("6", "9")):
        return f"sh{code}", f"{code}.SS"
    return f"sz{code}", f"{code}.SZ"


def fetch_universe(
    indices: list[str] | None = None,
    force_refresh: bool = False,
) -> list[dict]:
    """
    Return the investable A-share universe for the given CSI indices.

    Each element is a dict with keys:
        sym       — e.g. "sh600036"
        yf_ticker — e.g. "600036.SS"
        name      — Chinese company name, e.g. "招商银行"

    Results are cached to disk at ``data/cache/universe_cache.json`` and
    refreshed once per calendar day.  Pass ``force_refresh=True`` to bypass
    the cache (e.g. after an index rebalance).

    Parameters
    ----------
    indices : list of CSI index codes to union together.
              Default ["000300"] (CSI 300 — ~300 stocks, fastest to screen).
    force_refresh : ignore today's cache and re-fetch from akshare.
    """
    if indices is None:
        indices = ["000300"]

    # Stable cache key — order independent
    cache_key = "_".join(sorted(set(indices)))

    # --- disk cache (valid for today) ---
    if not force_refresh and _CACHE_FILE.exists():
        try:
            cached = json.loads(_CACHE_FILE.read_text(encoding="utf-8"))
            if (
                cached.get("date") == date.today().isoformat()
                and cached.get("key") == cache_key
            ):
                _log.info(
                    "universe_fetcher: cache hit — %d stocks (key=%s)",
                    len(cached["symbols"]),
                    cache_key,
                )
                return cached["symbols"]
        except Exception as exc:
            _log.warning("universe_fetcher: cache read error: %s", exc)

    # --- live fetch from akshare ---
    try:
        import akshare as ak

        seen: set[str] = set()
        symbols: list[dict] = []

        for idx in sorted(set(indices)):
            try:
                df = ak.index_stock_cons(symbol=idx)
            except Exception as exc:
                _log.warning(
                    "universe_fetcher: index %s (%s) fetch failed: %s",
                    idx,
                    INDEX_NAMES.get(idx, idx),
                    exc,
                )
                continue

            for _, row in df.iterrows():
                code = str(row["品种代码"]).zfill(6)
                if code in seen:
                    continue
                seen.add(code)
                sym, yf_ticker = _code_to_sym_yf(code)
                symbols.append({
                    "sym":       sym,
                    "yf_ticker": yf_ticker,
                    "name":      str(row["品种名称"]),
                })

        if symbols:
            _CACHE_DIR.mkdir(parents=True, exist_ok=True)
            _CACHE_FILE.write_text(
                json.dumps(
                    {
                        "date":    date.today().isoformat(),
                        "key":     cache_key,
                        "symbols": symbols,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            _log.info(
                "universe_fetcher: fetched %d stocks from %s",
                len(symbols),
                [INDEX_NAMES.get(i, i) for i in sorted(set(indices))],
            )
            return symbols

        _log.warning("universe_fetcher: akshare returned 0 stocks, using fallback")

    except Exception as exc:
        _log.warning("universe_fetcher: akshare unavailable (%s), using fallback", exc)

    _log.warning(
        "universe_fetcher: returning fallback universe (%d stocks)",
        len(_FALLBACK),
    )
    return _FALLBACK
