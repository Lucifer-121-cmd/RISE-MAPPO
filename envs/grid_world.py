"""Continuous 2D world with obstacles, targets, and stochastic hazards.

The world is continuous; an occupancy grid is rasterised on demand at a
configurable resolution. Obstacles can be circles or axis-aligned
rectangles. Hazards are circular zones whose ground-truth danger field
is a Gaussian bump; the GP layer in Phase 2 will learn this field from
robot observations.

The class is deliberately self-contained: it has no dependency on the
environment or the dynamics module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np


_DEFAULT_RESOLUTION_M = 0.1   # metres per cell
_DEFAULT_GRID_SIZE = 100      # cells per side at world_size=10 m


@dataclass
class CircleObstacle:
    """Solid circular obstacle at ``(cx, cy)`` with radius ``r``."""
    cx: float
    cy: float
    r: float

    def contains(self, x: float, y: float, margin: float = 0.0) -> bool:
        return (x - self.cx) ** 2 + (y - self.cy) ** 2 <= (self.r + margin) ** 2


@dataclass
class RectObstacle:
    """Axis-aligned rectangular obstacle ``[x0, y0, x1, y1]``."""
    x0: float
    y0: float
    x1: float
    y1: float

    def contains(self, x: float, y: float, margin: float = 0.0) -> bool:
        return (
            self.x0 - margin <= x <= self.x1 + margin
            and self.y0 - margin <= y <= self.y1 + margin
        )


@dataclass
class Target:
    """A hidden target the team must detect by approaching within range."""
    x: float
    y: float
    detected: bool = False


@dataclass
class HazardZone:
    """Stochastic hazard with Gaussian intensity profile.

    The mean intensity at a query point is
    ``amplitude * exp(-0.5 * ((dx, dy) / sigma)^2)``. Sampled hazard
    values add zero-mean Gaussian observation noise of std
    :attr:`obs_noise`.
    """
    cx: float
    cy: float
    sigma: float
    amplitude: float = 1.0
    obs_noise: float = 0.05

    def mean(self, x: float | np.ndarray, y: float | np.ndarray) -> np.ndarray:
        d2 = (np.asarray(x) - self.cx) ** 2 + (np.asarray(y) - self.cy) ** 2
        return self.amplitude * np.exp(-0.5 * d2 / (self.sigma ** 2))


@dataclass
class GridWorldConfig:
    world_size: float = 10.0
    resolution: float = _DEFAULT_RESOLUTION_M
    num_obstacles: int = 10
    num_targets: int = 5
    num_hazards: int = 3
    difficulty: str = "medium"
    detect_range: float = 0.4   # m — robot detects target within this radius
    obstacle_radius_range: Tuple[float, float] = (0.3, 0.6)
    rect_size_range: Tuple[float, float] = (0.4, 0.8)
    hazard_sigma_range: Tuple[float, float] = (0.5, 1.2)
    margin_from_border: float = 0.5
    rect_fraction: float = 0.4    # fraction of obstacles spawned as rectangles
    seed: Optional[int] = None


class GridWorld:
    """Continuous 2D world with sampling, rendering, and detection helpers."""

    def __init__(self, cfg: Optional[GridWorldConfig] = None) -> None:
        self.cfg = cfg or GridWorldConfig()
        if self.cfg.world_size <= 0:
            raise ValueError("world_size must be positive")
        if self.cfg.resolution <= 0:
            raise ValueError("resolution must be positive")
        self._rng = np.random.default_rng(self.cfg.seed)
        self.obstacles: List[CircleObstacle | RectObstacle] = []
        self.targets: List[Target] = []
        self.hazards: List[HazardZone] = []
        self._occupancy: Optional[np.ndarray] = None
        self.reset()

    @property
    def grid_size(self) -> int:
        return int(round(self.cfg.world_size / self.cfg.resolution))

    def reset(self, seed: Optional[int] = None) -> None:
        """Regenerate the world. Keeps the same dimensions but resamples
        obstacles, targets, and hazards."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.obstacles = self._sample_obstacles()
        self.targets = self._sample_targets()
        self.hazards = self._sample_hazards()
        self._occupancy = None

    def get_occupancy_grid(self) -> np.ndarray:
        """Return a binary ``(G, G)`` occupancy grid (1 = obstacle).

        Cached; invalidated on :meth:`reset`.
        """
        if self._occupancy is not None:
            return self._occupancy
        g = self.grid_size
        occ = np.zeros((g, g), dtype=np.float32)
        # Sample cell centres.
        xs = (np.arange(g) + 0.5) * self.cfg.resolution
        ys = (np.arange(g) + 0.5) * self.cfg.resolution
        xx, yy = np.meshgrid(xs, ys, indexing="xy")
        for obs in self.obstacles:
            if isinstance(obs, CircleObstacle):
                mask = (xx - obs.cx) ** 2 + (yy - obs.cy) ** 2 <= obs.r ** 2
            else:
                mask = (
                    (xx >= obs.x0) & (xx <= obs.x1)
                    & (yy >= obs.y0) & (yy <= obs.y1)
                )
            occ[mask] = 1.0
        self._occupancy = occ
        return occ

    def get_visibility_mask(
        self,
        robot_pos: Sequence[float],
        sensor_range: float,
    ) -> np.ndarray:
        """Return a boolean ``(G, G)`` mask of cells visible to the robot.

        A cell is visible iff (a) it lies within ``sensor_range`` and
        (b) the line segment from the robot to the cell centre is not
        blocked by an obstacle. Bresenham-style raycast on the occupancy
        grid is used for the line-of-sight check.
        """
        if sensor_range <= 0:
            raise ValueError("sensor_range must be positive")
        occ = self.get_occupancy_grid()
        g = self.grid_size
        rx, ry = float(robot_pos[0]), float(robot_pos[1])
        rxi = int(np.clip(rx / self.cfg.resolution, 0, g - 1))
        ryi = int(np.clip(ry / self.cfg.resolution, 0, g - 1))
        max_cells = int(np.ceil(sensor_range / self.cfg.resolution))
        vis = np.zeros((g, g), dtype=bool)
        # Iterate cells in the bounding square; skip those outside range.
        x_lo = max(0, rxi - max_cells)
        x_hi = min(g, rxi + max_cells + 1)
        y_lo = max(0, ryi - max_cells)
        y_hi = min(g, ryi + max_cells + 1)
        for ix in range(x_lo, x_hi):
            for iy in range(y_lo, y_hi):
                cx = (ix + 0.5) * self.cfg.resolution
                cy = (iy + 0.5) * self.cfg.resolution
                d2 = (cx - rx) ** 2 + (cy - ry) ** 2
                if d2 > sensor_range ** 2:
                    continue
                if self._line_clear(rxi, ryi, ix, iy, occ):
                    vis[iy, ix] = True
        return vis

    def check_collision(self, pos: Sequence[float], radius: float = 0.0) -> bool:
        """Return True if a circular footprint of ``radius`` collides
        with the world boundary or any obstacle."""
        x, y = float(pos[0]), float(pos[1])
        if (
            x - radius < 0
            or x + radius > self.cfg.world_size
            or y - radius < 0
            or y + radius > self.cfg.world_size
        ):
            return True
        for obs in self.obstacles:
            if obs.contains(x, y, margin=radius):
                return True
        return False

    def check_target_detection(
        self,
        pos: Sequence[float],
        detect_range: Optional[float] = None,
    ) -> List[int]:
        """Mark previously undetected targets within ``detect_range`` of
        ``pos`` as detected, and return their indices."""
        rng = self.cfg.detect_range if detect_range is None else detect_range
        x, y = float(pos[0]), float(pos[1])
        new_idx: List[int] = []
        for i, t in enumerate(self.targets):
            if t.detected:
                continue
            if (t.x - x) ** 2 + (t.y - y) ** 2 <= rng ** 2:
                t.detected = True
                new_idx.append(i)
        return new_idx

    def get_ground_truth_hazard(
        self,
        x: float | np.ndarray,
        y: float | np.ndarray,
        noisy: bool = False,
    ) -> np.ndarray:
        """Sum hazard intensities at ``(x, y)``; optionally with obs noise."""
        x_arr = np.asarray(x, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        out = np.zeros(np.broadcast(x_arr, y_arr).shape, dtype=float)
        for h in self.hazards:
            out = out + h.mean(x_arr, y_arr)
        if noisy and self.hazards:
            std = float(np.mean([h.obs_noise for h in self.hazards]))
            out = out + self._rng.normal(0.0, std, size=out.shape)
        return out

    def render(self, ax=None):
        """Render the world to a matplotlib axis (lazy import)."""
        import matplotlib.patches as mpatches
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(6, 6))
        ax.set_xlim(0, self.cfg.world_size)
        ax.set_ylim(0, self.cfg.world_size)
        ax.set_aspect("equal")
        # Hazard intensity heatmap.
        if self.hazards:
            g = 80
            xs = np.linspace(0, self.cfg.world_size, g)
            ys = np.linspace(0, self.cfg.world_size, g)
            xx, yy = np.meshgrid(xs, ys)
            zz = self.get_ground_truth_hazard(xx, yy)
            ax.imshow(
                zz,
                extent=(0, self.cfg.world_size, 0, self.cfg.world_size),
                origin="lower",
                alpha=0.3,
                cmap="Reds",
            )
        for obs in self.obstacles:
            if isinstance(obs, CircleObstacle):
                ax.add_patch(mpatches.Circle((obs.cx, obs.cy), obs.r, color="gray"))
            else:
                ax.add_patch(
                    mpatches.Rectangle(
                        (obs.x0, obs.y0),
                        obs.x1 - obs.x0,
                        obs.y1 - obs.y0,
                        color="gray",
                    )
                )
        for t in self.targets:
            color = "green" if t.detected else "blue"
            ax.plot(t.x, t.y, "*", color=color, markersize=12)
        return ax

    # ------------------------------------------------------------------
    # Sampling helpers
    # ------------------------------------------------------------------
    def _sample_position(
        self,
        margin: float,
        avoid: Sequence[Tuple[float, float, float]] = (),
        max_tries: int = 200,
    ) -> Tuple[float, float]:
        """Sample a free point with ``margin`` from boundary and existing
        obstacles. ``avoid`` is a list of ``(cx, cy, radius)`` to dodge."""
        lo = margin
        hi = self.cfg.world_size - margin
        for _ in range(max_tries):
            x = float(self._rng.uniform(lo, hi))
            y = float(self._rng.uniform(lo, hi))
            ok = True
            for (cx, cy, r) in avoid:
                if (x - cx) ** 2 + (y - cy) ** 2 < r ** 2:
                    ok = False
                    break
            if ok and not self._point_in_obstacle(x, y, margin=margin):
                return x, y
        return float(self._rng.uniform(lo, hi)), float(self._rng.uniform(lo, hi))

    def _point_in_obstacle(self, x: float, y: float, margin: float = 0.0) -> bool:
        for obs in self.obstacles:
            if obs.contains(x, y, margin=margin):
                return True
        return False

    def _sample_obstacles(self) -> List[CircleObstacle | RectObstacle]:
        n = self._difficulty_scaled(self.cfg.num_obstacles)
        obs: List[CircleObstacle | RectObstacle] = []
        for _ in range(n):
            if self._rng.random() < self.cfg.rect_fraction:
                w = float(self._rng.uniform(*self.cfg.rect_size_range))
                h = float(self._rng.uniform(*self.cfg.rect_size_range))
                x0 = float(self._rng.uniform(
                    self.cfg.margin_from_border,
                    self.cfg.world_size - self.cfg.margin_from_border - w,
                ))
                y0 = float(self._rng.uniform(
                    self.cfg.margin_from_border,
                    self.cfg.world_size - self.cfg.margin_from_border - h,
                ))
                obs.append(RectObstacle(x0, y0, x0 + w, y0 + h))
            else:
                r = float(self._rng.uniform(*self.cfg.obstacle_radius_range))
                cx = float(self._rng.uniform(
                    self.cfg.margin_from_border + r,
                    self.cfg.world_size - self.cfg.margin_from_border - r,
                ))
                cy = float(self._rng.uniform(
                    self.cfg.margin_from_border + r,
                    self.cfg.world_size - self.cfg.margin_from_border - r,
                ))
                obs.append(CircleObstacle(cx, cy, r))
        # Briefly install obstacles so position sampling can avoid them.
        self.obstacles = obs
        return obs

    def _sample_targets(self) -> List[Target]:
        targets: List[Target] = []
        for _ in range(self.cfg.num_targets):
            x, y = self._sample_position(margin=0.2)
            targets.append(Target(x=x, y=y))
        return targets

    def _sample_hazards(self) -> List[HazardZone]:
        hazards: List[HazardZone] = []
        for _ in range(self.cfg.num_hazards):
            x, y = self._sample_position(margin=0.5)
            sigma = float(self._rng.uniform(*self.cfg.hazard_sigma_range))
            hazards.append(HazardZone(cx=x, cy=y, sigma=sigma))
        return hazards

    def _difficulty_scaled(self, base: int) -> int:
        d = self.cfg.difficulty.lower()
        if d == "easy":
            return max(0, base // 2)
        if d == "hard":
            return int(base * 1.5)
        return base

    # ------------------------------------------------------------------
    # Bresenham line clearance over occupancy grid
    # ------------------------------------------------------------------
    @staticmethod
    def _line_clear(
        x0: int,
        y0: int,
        x1: int,
        y1: int,
        occ: np.ndarray,
    ) -> bool:
        dx = abs(x1 - x0)
        dy = -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx + dy
        ix, iy = x0, y0
        h, w = occ.shape
        while True:
            if 0 <= ix < w and 0 <= iy < h and occ[iy, ix] > 0.5:
                # Hitting the target cell itself counts as visible (an
                # obstacle is observable from outside it).
                if ix == x1 and iy == y1:
                    return True
                return False
            if ix == x1 and iy == y1:
                return True
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                ix += sx
            if e2 <= dx:
                err += dx
                iy += sy
