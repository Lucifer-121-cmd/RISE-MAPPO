"""Observation utilities and dict→array flatteners for MARL wrappers.

These helpers exist so the env can return rich, structured observations
(images + vectors) while the MAPPO actor/critic receive consistent
tensors.
"""

from __future__ import annotations

from typing import Dict, Mapping

import numpy as np


def flatten_obs_dict(obs: Mapping[str, np.ndarray]) -> np.ndarray:
    """Concatenate all leaves of an observation dict into a 1-D float32
    array. Used as a fallback when an MLP-only actor is desired."""
    parts = []
    for k in sorted(obs.keys()):
        parts.append(np.asarray(obs[k], dtype=np.float32).reshape(-1))
    return np.concatenate(parts, axis=0)


def stack_agent_obs(per_agent: Dict[str, Mapping[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    """Stack a dict ``{agent_id: obs_dict}`` into ``{key: (N, ...) array}``."""
    if not per_agent:
        return {}
    keys = list(next(iter(per_agent.values())).keys())
    out: Dict[str, np.ndarray] = {}
    for k in keys:
        arrays = [np.asarray(per_agent[a][k]) for a in per_agent]
        out[k] = np.stack(arrays, axis=0)
    return out
