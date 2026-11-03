"""Copied snapshots for plotting; recording does not perform estimation."""

from dataclasses import dataclass

import numpy as np

from .types import Observation


@dataclass(frozen=True)
class Frame:
    observation: Observation
    belief: np.ndarray | None = None

    def __post_init__(self):
        if self.belief is not None:
            snapshot = np.array(self.belief, dtype=float, copy=True)
            snapshot.setflags(write=False)
            object.__setattr__(self, "belief", snapshot)


def trajectory(frames) -> np.ndarray:
    """Include initial position and repeated positions after blocked commands."""
    return np.array([f.observation.robot_xy for f in frames], dtype=float).reshape(-1, 2)
