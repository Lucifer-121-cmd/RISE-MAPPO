"""Package setup for Paper 3.

Risk-Aware Multi-Robot Cooperative Search via Hierarchical MARL with
GP-Guided Lyapunov-MPC.
"""
from __future__ import annotations

from pathlib import Path

from setuptools import find_packages, setup


_HERE = Path(__file__).resolve().parent
_REQS = (_HERE / "requirements.txt").read_text().splitlines()
_INSTALL_REQUIRES = [
    line.strip()
    for line in _REQS
    if line.strip() and not line.strip().startswith("#")
]


setup(
    name="paper3",
    version="0.1.0",
    description=(
        "Risk-Aware Multi-Robot Cooperative Search: Hierarchical MARL with "
        "GP-Guided Lyapunov-MPC"
    ),
    author="Nischal Dhakal",
    python_requires=">=3.10",
    packages=find_packages(exclude=("tests", "scripts", "docs", "results")),
    install_requires=_INSTALL_REQUIRES,
)
