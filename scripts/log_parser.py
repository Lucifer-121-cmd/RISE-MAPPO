"""Shared log-parsing helpers for Paper 3 batch tooling.

Imported by both ``scripts/check_status.py`` and ``scripts/collect_results.py``
so the regex stays in one place.

Log line format (emitted by ``paper3.runner``)::

    2026-05-12 15:24:45,470 paper3.runner INFO | update 225  R=388.895  cov=0.640  det=1.47  pl=-0.038  vl=0.222  H=1.913  kl=0.0190  352.9s
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional

LOG_PATTERN = re.compile(
    r"update\s+(?P<update>\d+)\s+"
    r"R=(?P<reward>-?[\d.]+)\s+"
    r"cov=(?P<coverage>-?[\d.]+)\s+"
    r"det=(?P<detections>-?[\d.]+)\s+"
    r"pl=(?P<policy_loss>-?[\d.]+)\s+"
    r"vl=(?P<value_loss>-?[\d.]+)\s+"
    r"H=(?P<entropy>-?[\d.]+)\s+"
    r"kl=(?P<kl>-?[\d.]+)\s+"
    r"(?P<seconds>[\d.]+)s"
)


@dataclass
class UpdateRecord:
    update: int
    reward: float
    coverage: float
    detections: float
    policy_loss: float
    value_loss: float
    entropy: float
    kl: float
    seconds: float


def parse_line(line: str) -> Optional[UpdateRecord]:
    """Parse a single log line; return None if it does not match."""
    m = LOG_PATTERN.search(line)
    if not m:
        return None
    g = m.groupdict()
    try:
        return UpdateRecord(
            update=int(g["update"]),
            reward=float(g["reward"]),
            coverage=float(g["coverage"]),
            detections=float(g["detections"]),
            policy_loss=float(g["policy_loss"]),
            value_loss=float(g["value_loss"]),
            entropy=float(g["entropy"]),
            kl=float(g["kl"]),
            seconds=float(g["seconds"]),
        )
    except (TypeError, ValueError):
        return None


def iter_updates(log_path: Path) -> Iterator[UpdateRecord]:
    """Yield every matching record in the log, in file order."""
    if not log_path.exists():
        return
    with log_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            rec = parse_line(line)
            if rec is not None:
                yield rec


def last_update(log_path: Path, tail_bytes: int = 65536) -> Optional[UpdateRecord]:
    """Return the last matching record by tailing the file.

    Avoids reading huge logs end-to-end; reads the trailing ``tail_bytes``
    and scans backwards for the most recent match. Falls back to a full
    scan if no match is found in the tail.
    """
    if not log_path.exists():
        return None
    size = log_path.stat().st_size
    if size == 0:
        return None
    with log_path.open("rb") as f:
        if size > tail_bytes:
            f.seek(size - tail_bytes, os.SEEK_SET)
            f.readline()  # discard partial first line
        chunk = f.read().decode("utf-8", errors="replace")
    lines = chunk.splitlines()
    for line in reversed(lines):
        rec = parse_line(line)
        if rec is not None:
            return rec
    # Fallback: full scan
    last: Optional[UpdateRecord] = None
    for rec in iter_updates(log_path):
        last = rec
    return last


def average_seconds_per_update(records: Iterable[UpdateRecord]) -> Optional[float]:
    vals = [r.seconds for r in records if r.seconds > 0]
    if not vals:
        return None
    return sum(vals) / len(vals)
