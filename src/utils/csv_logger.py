"""Lightweight CSV metrics logger — a wandb-free replacement for wandb.log.

The training drivers log a dict of scalar metrics (``info_vals``) once per iteration. This
module persists those rows to a CSV file and generates a local run name to stand in for
``wandb.run.name`` (used by the drivers to build a per-run log subdirectory and to name
checkpoints). Modeled on RxnFlow's CSV logging (``rxnflow/base/gflownet/online_trainer.py``):
one wide CSV, one row per step, flushed after every write so partial runs stay inspectable.

``CSVLogger`` is a context manager so it slots into the drivers exactly where
``with wandb.init(...):`` used to sit, leaving the training-loop body unchanged.
"""

import csv
import json
import os
from datetime import datetime


def generate_run_name(prefix: str = "run") -> str:
    """Local stand-in for ``wandb.run.name``: ``<prefix>_<YYYYmmdd_HHMMSS>``."""
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def save_run_config(config: dict, path: str) -> None:
    """Persist the run config (what used to be handed to wandb.init's ``config=``).

    Written as JSON with ``default=str`` so EasyDict / numpy / Path values never raise.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(config, f, indent=2, default=str)


class CSVLogger:
    """Append-only CSV metrics writer (replaces ``wandb.log``).

    Columns are fixed from the keys of the first logged dict — the drivers log a stable set of
    keys each iteration. Keys absent from a later call are written empty; unexpected new keys
    are dropped (with a one-time notice) so the header stays consistent. Flushes every row.

    The file is opened lazily on the first ``log()`` call (mirroring RxnFlow). Since the drivers
    only call ``log()`` on rank 0, non-logging ranks in a distributed run never open — and so
    never truncate — the shared CSV.
    """

    def __init__(self, path: str):
        self.path = path
        self._file = None
        self._writer = None
        self._columns = None
        self._warned_extra = set()

    @staticmethod
    def _fmt(v):
        # torch tensors / numpy scalars -> python scalar
        return v.item() if hasattr(v, "item") else v

    def log(self, info: dict):
        if self._columns is None:
            os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
            self._file = open(self.path, "w", newline="")
            self._writer = csv.writer(self._file)
            self._columns = list(info.keys())
            self._writer.writerow(self._columns)
        else:
            for k in info:
                if k not in self._columns and k not in self._warned_extra:
                    print(f"[CSVLogger] dropping metric key not in CSV header: {k!r}")
                    self._warned_extra.add(k)
        self._writer.writerow([self._fmt(info[c]) if c in info else "" for c in self._columns])
        self._file.flush()

    def close(self):
        if self._file is not None and not self._file.closed:
            self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False
