#!/usr/bin/env python3
"""Parse all RISE-MAPPO training logs into CSV files for analysis/plotting.

Outputs (overwritten on each run, idempotent):
    results/all_training_curves.csv     — per-update rows for every run
    results/final_metrics_summary.csv   — one row per completed run

Also prints an aggregated method-level summary to stdout.

Usage::

    python scripts/collect_results.py
    python scripts/collect_results.py --only-final
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List, Optional, Tuple

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from log_parser import UpdateRecord, iter_updates  # noqa: E402

_LOG = logging.getLogger("paper3.collect")

PROJECT_ROOT = _HERE.parent
RUNS_DIR = PROJECT_ROOT / "results" / "runs"
OUT_DIR = PROJECT_ROOT / "results"

CONFIG_GROUPS: Dict[str, str] = {
    "default": "RISE-MAPPO (Full)",
    "ablation_no_rise": "No RISE (MAPPO)",
    "ablation_no_attention": "No GP-Attention",
    "ablation_no_cvar_head": "No CVaR Head",
}

GROUP_PREFIX = {
    "full": "default",
    "no_rise": "ablation_no_rise",
    "no_attention": "ablation_no_attention",
    "no_cvar": "ablation_no_cvar_head",
}


@dataclass
class RunMeta:
    name: str
    run_dir: Path
    config_key: str  # one of CONFIG_GROUPS keys
    seed: int
    completed: bool
    records: List[UpdateRecord]

    @property
    def label(self) -> str:
        return CONFIG_GROUPS.get(self.config_key, self.config_key)


def parse_run_name(name: str) -> Optional[Tuple[str, int]]:
    """`full_seed42` → ("default", 42)."""
    if "_seed" not in name:
        return None
    prefix, _, seed_str = name.rpartition("_seed")
    try:
        seed = int(seed_str)
    except ValueError:
        return None
    cfg_key = GROUP_PREFIX.get(prefix)
    if cfg_key is None:
        return None
    return cfg_key, seed


def discover_runs() -> List[RunMeta]:
    if not RUNS_DIR.exists():
        return []
    out: List[RunMeta] = []
    for entry in sorted(RUNS_DIR.iterdir()):
        if not entry.is_dir():
            continue
        parsed = parse_run_name(entry.name)
        if parsed is None:
            _LOG.warning("Skipping unrecognised run dir: %s", entry.name)
            continue
        cfg_key, seed = parsed
        log_path = entry / "train.log"
        records = list(iter_updates(log_path)) if log_path.exists() else []
        completed = (entry / "DONE").exists()
        out.append(
            RunMeta(
                name=entry.name,
                run_dir=entry,
                config_key=cfg_key,
                seed=seed,
                completed=completed,
                records=records,
            )
        )
    return out


def write_curves_csv(runs: List[RunMeta], path: Path) -> int:
    rows = 0
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "run_name", "config", "seed", "update",
                "reward", "coverage", "detections",
                "policy_loss", "value_loss", "entropy", "kl",
                "seconds_per_update",
            ]
        )
        for r in runs:
            for rec in r.records:
                w.writerow(
                    [
                        r.name, r.config_key, r.seed, rec.update,
                        rec.reward, rec.coverage, rec.detections,
                        rec.policy_loss, rec.value_loss, rec.entropy, rec.kl,
                        rec.seconds,
                    ]
                )
                rows += 1
    return rows


def write_final_csv(runs: List[RunMeta], path: Path) -> int:
    rows = 0
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "run_name", "config", "seed", "completed",
                "final_update", "final_reward", "final_coverage",
                "final_detections", "final_value_loss", "final_entropy",
                "total_time_hours",
            ]
        )
        for r in runs:
            if not r.records:
                continue
            last = r.records[-1]
            total_s = sum(rec.seconds for rec in r.records)
            w.writerow(
                [
                    r.name, r.config_key, r.seed, int(r.completed),
                    last.update, f"{last.reward:.4f}", f"{last.coverage:.4f}",
                    f"{last.detections:.4f}", f"{last.value_loss:.4f}",
                    f"{last.entropy:.4f}", f"{total_s / 3600.0:.2f}",
                ]
            )
            rows += 1
    return rows


def _mean_std(vals: List[float]) -> Tuple[float, float]:
    if not vals:
        return float("nan"), float("nan")
    if len(vals) == 1:
        return vals[0], 0.0
    return mean(vals), pstdev(vals)


def print_summary(runs: List[RunMeta]) -> None:
    completed = [r for r in runs if r.completed and r.records]
    by_group: Dict[str, List[RunMeta]] = {}
    for r in completed:
        by_group.setdefault(r.config_key, []).append(r)

    print()
    print("RISE-MAPPO Multi-Seed Results Summary")
    print("=" * 38)
    print()
    header = (
        f"{'Method':<22} {'Seeds':<6} "
        f"{'Reward (mean±std)':<22} "
        f"{'Coverage (mean±std)':<22} "
        f"{'Detections (mean±std)':<22}"
    )
    print(header)
    print("-" * len(header))

    summary: Dict[str, Tuple[float, float, float, float, float, float, int]] = {}
    for key, label in CONFIG_GROUPS.items():
        group = by_group.get(key, [])
        rewards = [r.records[-1].reward for r in group]
        covs = [r.records[-1].coverage for r in group]
        dets = [r.records[-1].detections for r in group]
        r_m, r_s = _mean_std(rewards)
        c_m, c_s = _mean_std(covs)
        d_m, d_s = _mean_std(dets)
        n = len(group)
        summary[key] = (r_m, r_s, c_m, c_s, d_m, d_s, n)
        if n == 0:
            print(f"{label:<22} {n:<6} {'(no completed runs)':<22}")
            continue
        print(
            f"{label:<22} {n:<6} "
            f"{r_m:7.2f} ± {r_s:5.2f}        "
            f"{c_m:6.3f} ± {c_s:5.3f}        "
            f"{d_m:6.3f} ± {d_s:5.3f}"
        )

    # Ablation deltas vs Full
    if "default" in summary and summary["default"][6] > 0:
        full_r, _, full_c, _, full_d, _, _ = summary["default"]
        print()
        print("Ablation Δ from Full:")
        for key in ("ablation_no_rise", "ablation_no_attention", "ablation_no_cvar_head"):
            if key not in summary or summary[key][6] == 0:
                continue
            r_m, _, c_m, _, d_m, _, _ = summary[key]
            dr = r_m - full_r
            dc = c_m - full_c
            dd = d_m - full_d
            pct_r = (dr / full_r * 100.0) if full_r else float("nan")
            pct_c = (dc / full_c * 100.0) if full_c else float("nan")
            print(
                f"  {CONFIG_GROUPS[key]:<22}  "
                f"ΔR={dr:+7.2f} ({pct_r:+5.1f}%)   "
                f"Δcov={dc:+6.3f} ({pct_c:+5.1f}%)   "
                f"Δdet={dd:+6.3f}"
            )

    partial = [r for r in runs if r.records and not r.completed]
    if partial:
        print()
        print(f"Note: {len(partial)} run(s) parsed but not yet DONE — excluded from aggregates:")
        for r in partial:
            print(f"  • {r.name}  (update {r.records[-1].update})")

    empty = [r for r in runs if not r.records]
    if empty:
        print()
        print(f"Note: {len(empty)} run dir(s) had no parsable updates:")
        for r in empty:
            print(f"  • {r.name}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--only-final", action="store_true", help="Skip per-update CSV.")
    p.add_argument("--log-level", default="WARNING")
    args = p.parse_args()
    logging.basicConfig(
        format="%(asctime)s %(name)s %(levelname)s | %(message)s",
        level=getattr(logging, args.log_level.upper(), logging.WARNING),
    )

    runs = discover_runs()
    if not runs:
        print(f"No runs found under {RUNS_DIR}", file=sys.stderr)
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    final_path = OUT_DIR / "final_metrics_summary.csv"
    n_final = write_final_csv(runs, final_path)
    print(f"Wrote {final_path} ({n_final} rows)")

    if not args.only_final:
        curves_path = OUT_DIR / "all_training_curves.csv"
        n_curves = write_curves_csv(runs, curves_path)
        print(f"Wrote {curves_path} ({n_curves} rows)")

    print_summary(runs)


if __name__ == "__main__":
    main()
