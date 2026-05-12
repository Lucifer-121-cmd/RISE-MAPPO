#!/usr/bin/env python3
"""Live status dashboard for Paper 3 multi-seed + ablation training runs.

Usage::

    python scripts/check_status.py
    python scripts/check_status.py --watch 60   # refresh every 60s
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

# Allow running as a script from project root or scripts/
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from log_parser import UpdateRecord, last_update, average_seconds_per_update, iter_updates  # noqa: E402

_LOG = logging.getLogger("paper3.status")

PROJECT_ROOT = _HERE.parent
RUNS_DIR = PROJECT_ROOT / "results" / "runs"

# (config_path, seed, phase_label)
RUN_REGISTRY: List[tuple] = [
    # Phase A — main method
    ("configs/default.yaml", 42, "A"),
    ("configs/default.yaml", 123, "A"),
    ("configs/default.yaml", 456, "A"),
    ("configs/default.yaml", 789, "A"),
    ("configs/default.yaml", 1024, "A"),
    # Phase B — ablation
    ("configs/ablation_no_rise.yaml", 42, "B"),
    ("configs/ablation_no_rise.yaml", 123, "B"),
    ("configs/ablation_no_rise.yaml", 456, "B"),
    ("configs/ablation_no_attention.yaml", 42, "B"),
    ("configs/ablation_no_attention.yaml", 123, "B"),
    ("configs/ablation_no_attention.yaml", 456, "B"),
    ("configs/ablation_no_cvar_head.yaml", 42, "B"),
    ("configs/ablation_no_cvar_head.yaml", 123, "B"),
    ("configs/ablation_no_cvar_head.yaml", 456, "B"),
]

# Default total updates (matches configs/default.yaml training.n_training_updates).
DEFAULT_TOTAL_UPDATES = 1000


def run_name_for(config_path: str, seed: int) -> str:
    base = Path(config_path).stem
    mapping = {
        "default": "full",
        "ablation_no_rise": "no_rise",
        "ablation_no_attention": "no_attention",
        "ablation_no_cvar_head": "no_cvar",
    }
    prefix = mapping.get(base, base.removeprefix("ablation_"))
    return f"{prefix}_seed{seed}"


@dataclass
class RunStatus:
    name: str
    phase: str
    state: str  # "done" | "running" | "pending" | "failed"
    last: Optional[UpdateRecord]
    total_updates: int


def total_updates_for(run_dir: Path) -> int:
    """Read training.n_training_updates from run_config.yaml if present."""
    cfg_path = run_dir / "run_config.yaml"
    if not cfg_path.exists():
        return DEFAULT_TOTAL_UPDATES
    try:
        import yaml
        with cfg_path.open() as f:
            cfg = yaml.safe_load(f) or {}
        return int(cfg.get("training", {}).get("n_training_updates", DEFAULT_TOTAL_UPDATES))
    except Exception:
        return DEFAULT_TOTAL_UPDATES


def status_for(config_path: str, seed: int, phase: str) -> RunStatus:
    name = run_name_for(config_path, seed)
    run_dir = RUNS_DIR / name
    total = total_updates_for(run_dir)
    log_path = run_dir / "train.log"
    last = last_update(log_path) if log_path.exists() else None

    if (run_dir / "DONE").exists():
        state = "done"
    elif (run_dir / "FAILED").exists():
        state = "failed"
    elif log_path.exists():
        state = "running"
    else:
        state = "pending"
    return RunStatus(name=name, phase=phase, state=state, last=last, total_updates=total)


# ---- ANSI helpers ----
class _C:
    RESET = "\x1b[0m"
    GREEN = "\x1b[32m"
    YELLOW = "\x1b[33m"
    GRAY = "\x1b[90m"
    RED = "\x1b[31m"
    BOLD = "\x1b[1m"


_GLYPH = {
    "done": (f"{_C.GREEN}✓{_C.RESET}", "DONE   "),
    "running": (f"{_C.YELLOW}◉{_C.RESET}", "RUN    "),
    "pending": (f"{_C.GRAY}○{_C.RESET}", "PENDING"),
    "failed": (f"{_C.RED}✗{_C.RESET}", "FAILED "),
}


def format_row(rs: RunStatus) -> str:
    glyph, label = _GLYPH[rs.state]
    name_col = f"{rs.name:<22}"
    if rs.last is None:
        body = f"{label}"
    else:
        progress = f"{rs.last.update}/{rs.total_updates}"
        body = (
            f"{label}  {progress:>10}  "
            f"R={rs.last.reward:7.2f}  "
            f"cov={rs.last.coverage:5.3f}  "
            f"det={rs.last.detections:5.2f}  "
            f"{rs.last.seconds:5.1f}s/upd"
        )
    return f"  {glyph} {name_col} {body}"


def render(statuses: List[RunStatus]) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = []
    lines.append("╔" + "═" * 78 + "╗")
    title = f"RISE-MAPPO Training Status — {now}"
    lines.append("║ " + f"{_C.BOLD}{title}{_C.RESET}" + " " * (78 - len(title) - 1) + "║")
    lines.append("╠" + "═" * 78 + "╣")

    lines.append(f"{_C.BOLD}Phase A — Main Method (5 seeds):{_C.RESET}")
    for rs in [s for s in statuses if s.phase == "A"]:
        lines.append(format_row(rs))
    lines.append("")
    lines.append(f"{_C.BOLD}Phase B — Ablation (3 × 3 seeds):{_C.RESET}")
    for rs in [s for s in statuses if s.phase == "B"]:
        lines.append(format_row(rs))

    # Counters + ETA
    counts = {"done": 0, "running": 0, "pending": 0, "failed": 0}
    for s in statuses:
        counts[s.state] += 1

    # Average seconds/update from any data we have
    all_records: List[UpdateRecord] = []
    for s in statuses:
        run_dir = RUNS_DIR / s.name
        if (run_dir / "train.log").exists():
            all_records.extend(iter_updates(run_dir / "train.log"))
    avg_s = average_seconds_per_update(all_records)

    eta_str = "n/a"
    if avg_s is not None:
        remaining_updates = 0
        for s in statuses:
            if s.state == "pending":
                remaining_updates += s.total_updates
            elif s.state == "running" and s.last is not None:
                remaining_updates += max(0, s.total_updates - s.last.update)
        # Two GPUs assumed when parallel
        parallel_factor = 2
        eta_h = (remaining_updates * avg_s) / parallel_factor / 3600.0
        eta_str = f"~{eta_h:.1f}h (assuming 2 GPUs)"

    lines.append("")
    lines.append(
        f"Completed: {counts['done']}/{len(statuses)}  "
        f"Running: {counts['running']}  "
        f"Pending: {counts['pending']}  "
        f"Failed: {counts['failed']}"
    )
    if avg_s is not None:
        lines.append(f"Avg: {avg_s:.1f}s/update    Est. remaining: {eta_str}")
    lines.append("╚" + "═" * 78 + "╝")
    return "\n".join(lines)


def collect() -> List[RunStatus]:
    return [status_for(cfg, seed, phase) for cfg, seed, phase in RUN_REGISTRY]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--watch",
        type=int,
        default=None,
        metavar="SECONDS",
        help="Refresh every N seconds (Ctrl+C to exit).",
    )
    p.add_argument("--no-color", action="store_true", help="Disable ANSI color output.")
    args = p.parse_args()

    if args.no_color or not sys.stdout.isatty():
        for k in list(vars(_C)):
            if not k.startswith("_"):
                setattr(_C, k, "")

    if args.watch is None:
        print(render(collect()))
        return

    try:
        while True:
            os.system("clear" if os.name != "nt" else "cls")
            print(render(collect()))
            time.sleep(max(1, args.watch))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
