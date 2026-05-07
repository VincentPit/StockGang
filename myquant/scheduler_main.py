"""
myquant/scheduler_main.py — Standalone scheduler entrypoint (T1c).

Run with:
    python -m myquant.scheduler_main

Replaces the in-API scheduler that ``api/main.py`` used to start in its
lifespan. Holds a Redis leader lock so multiple scheduler replicas — even
accidentally — never double-fire cron triggers. The single elected leader
runs the APScheduler instance; non-leaders idle and re-attempt acquisition
periodically.

Logging goes to stdout so the orchestrator's log driver picks it up.
"""
from __future__ import annotations

import asyncio
import logging
import signal
import sys
from pathlib import Path

# Project root on path so ``api.*`` imports work the same as in the API process
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from myquant.leader_lock import leader_lock
from myquant.scheduler import scheduler_manager

_log = logging.getLogger("myquant.scheduler_main")

# Redis key — single name across the cluster
_LOCK_KEY        = "myquant:scheduler:leader"
# How often a non-leader re-tries acquisition
_FOLLOWER_RETRY  = 15.0


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)-30s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )


async def _run_as_leader(stop: asyncio.Event) -> None:
    """Start APScheduler and block until ``stop`` is set."""
    await scheduler_manager.start()
    _log.info("Scheduler running as leader — waiting for shutdown signal")
    try:
        await stop.wait()
    finally:
        await scheduler_manager.shutdown()


async def _follower_wait(stop: asyncio.Event) -> None:
    """Idle until either shutdown or it's time to retry leader election."""
    try:
        await asyncio.wait_for(stop.wait(), timeout=_FOLLOWER_RETRY)
    except asyncio.TimeoutError:
        pass


async def main() -> None:
    _setup_logging()
    stop = asyncio.Event()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop.set)
        except NotImplementedError:
            # Windows / non-Unix — fall back to default signal handling
            pass

    _log.info("Scheduler container starting; competing for leader lock %s", _LOCK_KEY)

    while not stop.is_set():
        async with leader_lock(_LOCK_KEY) as got_it:
            if got_it:
                await _run_as_leader(stop)
                # Once a leader exits (shutdown signal), bail out of the loop.
                break
            await _follower_wait(stop)

    _log.info("Scheduler container exiting")


if __name__ == "__main__":
    asyncio.run(main())
