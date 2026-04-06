"""
myquant/__main__.py — CLI entry point for running the scheduler standalone.

Usage:
    python -m myquant scheduler     # Run the auto-update scheduler
    python -m myquant trigger       # Trigger a one-time full pipeline run
    python -m myquant status        # Print current scheduler state
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _print_status():
    from myquant.scheduler import scheduler_manager
    status = scheduler_manager.get_status()
    print(json.dumps(status, indent=2, default=str))


async def _trigger_once():
    from myquant.scheduler import (
        _refresh_data_sync,
        _retrain_models_sync,
        _update_strategy_sync,
        PipelineState,
    )
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)-30s %(message)s",
    )

    state = PipelineState()
    print("=" * 60)
    print("  MyQuant — One-Time Pipeline Run")
    print("=" * 60)

    print("\n📊 Step 1/3: Refreshing market data...")
    data_result = _refresh_data_sync(state)
    print(f"  ✅ Fetched {data_result['bars_fetched']} bars for {data_result['symbols_count']} symbols")
    if data_result["errors"]:
        print(f"  ⚠️  {data_result['error_count']} symbol(s) had fetch errors")

    print("\n🧠 Step 2/3: Retraining models...")
    retrain_result = _retrain_models_sync(state)
    print(f"  ✅ Trained {retrain_result['total_trained']} models, skipped {retrain_result['total_skipped']}")
    print(f"  📈 Average OOS accuracy: {retrain_result['avg_oos_accuracy']:.1%}")

    print("\n⚙️  Step 3/3: Updating strategy parameters...")
    strategy_result = _update_strategy_sync(state, retrain_result)
    print(f"  ✅ {len(strategy_result.get('actions', []))} action(s) taken")
    if strategy_result.get("auto_tune_triggered"):
        print("  🔄 Auto-tune was triggered due to quality degradation")

    print("\n" + "=" * 60)
    print("  Pipeline complete!")
    print("=" * 60)


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m myquant <command>")
        print("Commands:")
        print("  scheduler  — Run the auto-update scheduler (long-running)")
        print("  trigger    — Run the full pipeline once and exit")
        print("  status     — Print current scheduler state")
        sys.exit(1)

    cmd = sys.argv[1].lower()

    if cmd == "scheduler":
        from myquant.scheduler import _cli_main
        asyncio.run(_cli_main())

    elif cmd == "trigger":
        asyncio.run(_trigger_once())

    elif cmd == "status":
        _print_status()

    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)


if __name__ == "__main__":
    main()
