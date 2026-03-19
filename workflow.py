"""
workflow.py — End-to-end MyQuant pipeline.

  1. Screen 33-stock universe with real yfinance data
  2. Auto-update SYMBOLS in backtest_run.py with the top-N scorers
  3. Run the backtest
  4. Print a clean results summary with per-symbol P&L

Usage:
    python workflow.py               # top-6 (default)
    python workflow.py --top 8
    python workflow.py --top 6 --dry-run   # screen only, don't modify backtest_run.py
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))


# ── helpers ─────────────────────────────────────────────────────────────────

def _step(n: int, label: str) -> None:
    print(f"\n{'─'*70}")
    print(f"  STEP {n}  {label}")
    print(f"{'─'*70}")


def _update_symbols(top_syms: list[str], results: list[dict]) -> str:
    """Rewrite the SYMBOLS block in backtest_run.py and return old symbol list."""
    bp = ROOT / "backtest_run.py"
    text = bp.read_text()

    # Build new SYMBOLS block
    lines = ["SYMBOLS = ["]
    for sym in top_syms:
        r = next(x for x in results if x["sym"] == sym)
        lines.append(
            f'    "{sym}",   '
            f'# {r["name"]:<20}  score={r["score"]:.3f} | '
            f'1Y={r["ret_1y"]:+.1%} | ATR={r["atr_pct"]:.2%} | Trend={r["trend_pct"]:.1%}'
        )
    lines.append("]")
    new_block = "\n".join(lines)

    # Replace existing SYMBOLS = [...] block (handles multi-line)
    pattern = r"^SYMBOLS\s*=\s*\[.*?\]"
    new_text, n_subs = re.subn(pattern, new_block, text, flags=re.DOTALL | re.MULTILINE)
    if n_subs == 0:
        print("  ⚠  Could not find SYMBOLS block — backtest_run.py NOT modified.")
        return ""

    # Extract old symbols before overwriting
    old_match = re.search(r'SYMBOLS\s*=\s*\[(.*?)\]', text, re.DOTALL)
    old_syms = re.findall(r'"(\w+)"', old_match.group(1)) if old_match else []

    bp.write_text(new_text)
    return ", ".join(old_syms)


def _run_backtest() -> str:
    """Run backtest_run.py and return captured stdout."""
    python = ROOT / ".venv" / "bin" / "python"
    if not python.exists():
        python = Path(sys.executable)

    result = subprocess.run(
        [str(python), str(ROOT / "backtest_run.py")],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    if result.returncode != 0:
        print("  ⚠  Backtest exited with errors:")
        print(textwrap.indent(result.stderr[-2000:], "    "))
    return result.stdout + result.stderr


def _parse_and_print_results(output: str, top_syms: list[str], results: list[dict]) -> None:
    """Extract key metrics from myquant.backtest stdout and render a clean summary."""
    def grab(label: str) -> str:
        m = re.search(rf"{re.escape(label)}\s*:\s*(.+)", output)
        return m.group(1).strip() if m else "n/a"

    period      = grab("Period")
    final_nav   = grab("Final NAV")
    total_pnl   = grab("Total PnL")
    sharpe      = grab("Sharpe")
    max_dd      = grab("Max DD")
    n_trades    = grab("# Trades")
    win_rate    = grab("Win Rate")
    avg_win     = grab("Avg Win")
    avg_loss    = grab("Avg Loss")
    prof_fac    = grab("Profit Fac")

    print(f"\n  {'Period':<14}: {period}")
    print(f"  {'Final NAV':<14}: {final_nav}")
    print(f"  {'Total PnL':<14}: {total_pnl}")
    print(f"  {'Sharpe':<14}: {sharpe}")
    print(f"  {'Max DD':<14}: {max_dd}")
    print(f"  {'# Trades':<14}: {n_trades}")
    print(f"  {'Win Rate':<14}: {win_rate}")
    print(f"  {'Avg Win':<14}: {avg_win}")
    print(f"  {'Avg Loss':<14}: {avg_loss}")
    print(f"  {'Profit Fac':<14}: {prof_fac}")

    # Per-symbol table from raw output
    pnl_section = re.search(
        r"Per-symbol P&L attribution:(.*?)Per-strategy fill count:", output, re.DOTALL
    )
    if pnl_section:
        print(f"\n  Per-symbol P&L:")
        print(f"  {'Symbol':<16} {'Fills':>6}  {'Net P&L':>12}  {'Buys':>5}  {'Sells':>5}")
        print("  " + "-" * 54)
        for line in pnl_section.group(1).strip().splitlines():
            line = line.strip()
            # skip raw output's own header/separator lines
            if line and not line.startswith("Symbol") and not set(line) <= {'-', ' '}:
                print(f"  {line}")

    strat_section = re.search(
        r"Per-strategy fill count:(.*?)$", output, re.DOTALL
    )
    if strat_section:
        print(f"\n  Per-strategy fills:")
        print(f"  {'Strategy':<26} {'Fills':>6}  {'Buys':>5}  {'Sells':>5}")
        print("  " + "-" * 46)
        for line in strat_section.group(1).strip().splitlines():
            line = line.strip()
            if line and not line.startswith("Strategy") and not set(line) <= {'-', ' '}:
                print(f"  {line}")


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="MyQuant end-to-end workflow")
    parser.add_argument("--top",     type=int,  default=6,
                        help="Number of top-scoring stocks to select (default: 6)")
    parser.add_argument("--min-bars",type=int,  default=200,
                        help="Minimum data bars required per stock (default: 200)")
    parser.add_argument("--lookback",type=int,  default=1,
                        help="Scoring lookback in years (default: 1)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Screen only — do not modify backtest_run.py or run backtest")
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"  MyQuant  ·  End-to-end workflow  ·  top-{args.top} universe")
    print(f"{'='*70}")

    # ── Step 1: Screen ───────────────────────────────────────────────────
    _step(1, "Stock screener  (real yfinance data)")
    from myquant.tools.stock_screener import screen
    top_syms, results = screen(
        top_n=args.top,
        min_bars=args.min_bars,
        lookback_years=args.lookback,
    )

    if not top_syms:
        print("  No stocks selected — aborting.")
        sys.exit(1)

    if args.dry_run:
        print("\n  [dry-run] Stopping after screen — backtest_run.py NOT modified.")
        return

    # ── Step 2: Update SYMBOLS ───────────────────────────────────────────
    _step(2, "Updating SYMBOLS in backtest_run.py")
    old_syms = _update_symbols(top_syms, results)
    print(f"\n  Before : {old_syms or '(unknown)'}")
    print(f"  After  : {', '.join(top_syms)}")

    # ── Step 3: Backtest ─────────────────────────────────────────────────
    _step(3, "Running backtest")
    print("  (this may take a minute — training LGBM walk-forward...)\n")
    output = _run_backtest()

    # ── Step 4: Results ──────────────────────────────────────────────────
    _step(4, "Results")
    _parse_and_print_results(output, top_syms, results)

    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    main()
