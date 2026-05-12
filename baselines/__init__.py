"""Baselines + policy wrappers for evaluation."""
from baselines.base_policy import BasePolicy
from baselines.nearest_frontier import NearestFrontierPolicy
from baselines.random_policy import RandomPolicy
from baselines.voronoi_partition import VoronoiPartitionPolicy


BASELINE_REGISTRY = {
    "random": RandomPolicy,
    "nearest_frontier": NearestFrontierPolicy,
    "voronoi_partition": VoronoiPartitionPolicy,
}


__all__ = [
    "BasePolicy",
    "RandomPolicy",
    "NearestFrontierPolicy",
    "VoronoiPartitionPolicy",
    "BASELINE_REGISTRY",
]
