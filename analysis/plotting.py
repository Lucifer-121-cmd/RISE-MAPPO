"""Publication-quality plotting helpers (IEEE two-column friendly).

All functions render to PDF (vector) and PNG (preview) at 300 DPI. The
matplotlib backend is forced to ``Agg`` so the module is safe to import
on a headless box.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_LOG = logging.getLogger("paper3.plotting")

COLUMN_WIDTH = 3.5      # inches, IEEE single column
DOUBLE_WIDTH = 7.16     # inches, IEEE two columns
FONT_SIZE = 9

# Paul Tol "bright" qualitative palette — colorblind-safe.
COLORS = {
    "RISE-MAPPO": "#4477AA",
    "MAPPO-only": "#EE6677",
    "Nearest Frontier": "#228833",
    "Voronoi Partition": "#CCBB44",
    "Random": "#AA3377",
    "Ours-no-CVaR": "#66CCEE",
    "Ours-no-GP": "#BBBBBB",
    "Ours-no-Lyap": "#EE8866",
}
_FALLBACK_PALETTE = ("#4477AA", "#EE6677", "#228833", "#CCBB44", "#AA3377",
                      "#66CCEE", "#BBBBBB", "#EE8866")


def setup_ieee_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": FONT_SIZE,
        "axes.labelsize": FONT_SIZE,
        "axes.titlesize": FONT_SIZE,
        "xtick.labelsize": FONT_SIZE - 1,
        "ytick.labelsize": FONT_SIZE - 1,
        "legend.fontsize": FONT_SIZE - 1,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


def _color_for(name: str, idx: int = 0) -> str:
    return COLORS.get(name, _FALLBACK_PALETTE[idx % len(_FALLBACK_PALETTE)])


def _save(fig, output_path: Union[str, Path]) -> None:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".png"))
    plt.close(fig)
    _LOG.info("wrote %s.{pdf,png}", out.with_suffix(""))


# ---------------------------------------------------------------------------
# Training curves from train_seed*.log
# ---------------------------------------------------------------------------
_LOG_LINE = re.compile(
    r"update\s+(\d+)\s+R=([\-0-9.eE]+)\s+cov=([\-0-9.eE]+)\s+det=([\-0-9.eE]+)\s+"
    r"pl=([\-0-9.eE]+)\s+vl=([\-0-9.eE]+)\s+H=([\-0-9.eE]+)\s+kl=([\-0-9.eE]+)"
)


def parse_training_log(log_path: Union[str, Path]) -> Dict[str, np.ndarray]:
    keys = ("update", "R", "cov", "det", "pl", "vl", "H", "kl")
    rows: Dict[str, List[float]] = {k: [] for k in keys}
    for line in Path(log_path).read_text().splitlines():
        m = _LOG_LINE.search(line)
        if not m:
            continue
        for k, v in zip(keys, m.groups()):
            rows[k].append(float(v))
    return {k: np.asarray(rows[k], dtype=np.float64) for k in keys}


def plot_training_curves(log_path: Union[str, Path], output_path: Union[str, Path]) -> None:
    setup_ieee_style()
    data = parse_training_log(log_path)
    x = data["update"]
    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_WIDTH, DOUBLE_WIDTH * 0.6), sharex=True)
    panels = [
        (axes[0, 0], "R", "Episode return"),
        (axes[0, 1], "cov", "Coverage"),
        (axes[1, 0], "vl", "Value loss"),
        (axes[1, 1], "H", "Entropy"),
    ]
    for ax, key, label in panels:
        ax.plot(x, data[key], color="#4477AA", linewidth=1.0)
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
    for ax in axes[1, :]:
        ax.set_xlabel("Update")
    _save(fig, output_path)


# ---------------------------------------------------------------------------
# Metric comparison bar chart
# ---------------------------------------------------------------------------
def plot_metric_comparison_bar(
    results: Dict[str, Dict[str, Dict[str, float]]],
    metrics: Sequence[str],
    output_path: Union[str, Path],
) -> None:
    """Grouped bar chart, one group per metric, one bar per policy."""
    setup_ieee_style()
    policies = list(results.keys())
    n_p = len(policies)
    n_m = len(metrics)
    fig, ax = plt.subplots(figsize=(DOUBLE_WIDTH, DOUBLE_WIDTH * 0.4))
    x = np.arange(n_m)
    width = 0.8 / max(n_p, 1)
    for i, pol in enumerate(policies):
        means = [results[pol][m]["mean"] for m in metrics]
        stds = [results[pol][m]["std"] for m in metrics]
        ax.bar(
            x + (i - (n_p - 1) / 2) * width,
            means, width, yerr=stds, capsize=2,
            label=pol, color=_color_for(pol, i),
        )
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=15, ha="right")
    ax.set_ylabel("Value")
    ax.legend(loc="best", frameon=False)
    ax.grid(axis="y", alpha=0.3)
    _save(fig, output_path)


# ---------------------------------------------------------------------------
# Coverage-over-time curves
# ---------------------------------------------------------------------------
def plot_coverage_curves(
    curves: Dict[str, np.ndarray],
    output_path: Union[str, Path],
) -> None:
    """Mean ± std coverage curve per policy.

    ``curves[policy]`` has shape ``(N_episodes, T)``.
    """
    setup_ieee_style()
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, COLUMN_WIDTH * 0.7))
    for i, (pol, arr) in enumerate(curves.items()):
        if arr.size == 0:
            continue
        T = arr.shape[1]
        t = np.arange(T)
        # Use nan-safe aggregation because coverage curves are NaN-padded
        # for episodes that terminate early (crashes, energy depletion,
        # all-targets-found).  np.mean / np.std would propagate NaN.
        mu = np.nanmean(arr, axis=0)
        sd = np.nanstd(arr, axis=0)
        c = _color_for(pol, i)
        ax.plot(t, mu, color=c, label=pol, linewidth=1.2)
        ax.fill_between(t, mu - sd, mu + sd, color=c, alpha=0.2, linewidth=0)
    ax.set_xlabel("Step")
    ax.set_ylabel("Coverage")
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", frameon=False)
    _save(fig, output_path)


# ---------------------------------------------------------------------------
# Ablation bar chart
# ---------------------------------------------------------------------------
def plot_ablation_bar(
    results: Dict[str, Dict[str, Dict[str, float]]],
    metric: str = "coverage_rate",
    output_path: Union[str, Path] = "ablation_bar",
) -> None:
    setup_ieee_style()
    names = list(results.keys())
    means = [results[n][metric]["mean"] for n in names]
    stds = [results[n][metric]["std"] for n in names]
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, COLUMN_WIDTH * 0.7))
    colors = [_color_for(n, i) for i, n in enumerate(names)]
    ax.bar(np.arange(len(names)), means, yerr=stds, capsize=2, color=colors)
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel(metric)
    ax.grid(axis="y", alpha=0.3)
    _save(fig, output_path)


# ---------------------------------------------------------------------------
# Scalability plot
# ---------------------------------------------------------------------------
def plot_scalability(
    results: Dict[int, Dict[str, Dict[str, float]]],
    metric: str = "coverage_rate",
    output_path: Union[str, Path] = "scalability",
) -> None:
    setup_ieee_style()
    counts = sorted(results.keys())
    means = [results[c][metric]["mean"] for c in counts]
    stds = [results[c][metric]["std"] for c in counts]
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, COLUMN_WIDTH * 0.7))
    ax.errorbar(counts, means, yerr=stds, fmt="o-", color="#4477AA",
                ecolor="#4477AA", capsize=3, linewidth=1.2, markersize=4)
    ax.set_xlabel("Number of robots")
    ax.set_ylabel(metric)
    ax.grid(alpha=0.3)
    _save(fig, output_path)


# ---------------------------------------------------------------------------
# Trajectory snapshot grid
# ---------------------------------------------------------------------------
def plot_trajectory_snapshot(
    ep_npz_path: Union[str, Path],
    timesteps: Iterable[int],
    output_path: Union[str, Path],
) -> None:
    setup_ieee_style()
    data = np.load(ep_npz_path)
    cov = data["coverage_maps"]            # (T, G, G)
    sigma = data["gp_uncertainty"]         # (T, G, G)
    positions = data["robot_positions"]    # (T, N, 3)
    ws = float(data["world_size"])
    G = cov.shape[1]
    res = ws / G
    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_WIDTH, DOUBLE_WIDTH * 0.9))
    flat = axes.flatten()
    for ax, t in zip(flat, list(timesteps)):
        t = int(np.clip(t, 0, cov.shape[0] - 1))
        ax.imshow(sigma[t], origin="lower", extent=[0, ws, 0, ws], cmap="viridis", alpha=0.6)
        ax.imshow(np.where(cov[t], 1.0, np.nan), origin="lower",
                  extent=[0, ws, 0, ws], cmap="Greys", alpha=0.4)
        for r in range(positions.shape[1]):
            ax.plot(positions[t, r, 0], positions[t, r, 1], "o", color="red", markersize=3)
        ax.set_title(f"t = {t}")
        ax.set_xlim(0, ws); ax.set_ylim(0, ws)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
    _save(fig, output_path)


# ---------------------------------------------------------------------------
# Lyapunov convergence
# ---------------------------------------------------------------------------
def plot_lyapunov_convergence(
    ep_npz_path: Union[str, Path],
    output_path: Union[str, Path],
) -> None:
    setup_ieee_style()
    data = np.load(ep_npz_path)
    V = data["lyapunov_values"]   # (T, N)
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH, COLUMN_WIDTH * 0.7))
    for r in range(V.shape[1]):
        ax.plot(V[:, r], linewidth=1.0, label=f"robot {r}")
    ax.set_xlabel("Step")
    ax.set_ylabel(r"$V(t)$")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=FONT_SIZE - 2)
    _save(fig, output_path)
