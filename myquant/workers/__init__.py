"""
myquant/workers — Redis-backed async job queue (Arq).

Layout
------
  worker.py   Arq WorkerSettings + Redis connection helpers. Both the API
              (enqueue side) and the worker container (consume side) load
              the same module so task names stay in sync.
  tasks.py    One coroutine per `launch_*` function in api/runner.py. The
              tasks delegate to the existing `_run_*_sync` implementations
              so we don't duplicate business logic during the cutover.

Activation
----------
The API only reaches into this package when MYQUANT_QUEUE=arq. With the
default (threadpool) the package is dormant — importing it must not pull
in heavy deps. Worker entrypoint:

    arq myquant.workers.worker.WorkerSettings
"""
from __future__ import annotations
