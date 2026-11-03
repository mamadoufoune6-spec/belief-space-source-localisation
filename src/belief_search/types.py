"""Public configuration and immutable records; observations contain no source truth."""

from dataclasses import dataclass
from enum import Enum
import math

import numpy as np

Position = tuple[float, float]


class Action(Enum):
    FORWARD = (0, 1)
    BACKWARD = (0, -1)
    LEFT = (-1, 0)
    RIGHT = (1, 0)


@dataclass(frozen=True)
class EnvironmentConfig:
    width_m: float = 12.0
    height_m: float = 12.0
    spacing_m: float = 0.5
    success_radius_m: float = 0.5
    max_steps: int = 80
    reference_power_dbm: float = -40.0
    reference_distance_m: float = 1.0
    path_loss_exponent: float = 2.0
    min_distance_m: float = 0.5
    noise_std_db: float = 2.0

    def __post_init__(self):
        for name in ("width_m", "height_m", "spacing_m", "reference_distance_m",
                     "path_loss_exponent", "min_distance_m"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("success_radius_m", "noise_std_db"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not math.isfinite(self.reference_power_dbm):
            raise ValueError("reference_power_dbm must be finite")
        if isinstance(self.max_steps, bool) or not isinstance(self.max_steps, int) or self.max_steps < 1:
            raise ValueError("max_steps must be a positive integer")
        for length in (self.width_m, self.height_m):
            if not math.isclose(length / self.spacing_m, round(length / self.spacing_m), abs_tol=1e-9):
                raise ValueError("Arena dimensions must be multiples of spacing_m")

    @property
    def shape(self) -> tuple[int, int]:
        """Array convention: (ny, nx), so belief[y_index, x_index]."""
        return (round(self.height_m / self.spacing_m) + 1,
                round(self.width_m / self.spacing_m) + 1)

    def grid_axes(self) -> tuple[np.ndarray, np.ndarray]:
        ny, nx = self.shape
        return np.linspace(0, self.width_m, nx), np.linspace(0, self.height_m, ny)


def grid_index(position, config: EnvironmentConfig) -> tuple[int, int]:
    """Validate a position and return its integer (x, y) grid indices."""
    xy = np.asarray(position, dtype=float)
    if xy.shape != (2,) or not np.isfinite(xy).all():
        raise ValueError("Position must contain two finite coordinates")
    if np.any(xy < 0) or np.any(xy > (config.width_m, config.height_m)):
        raise ValueError("Position is outside the arena")
    indices = xy / config.spacing_m
    if not np.allclose(indices, np.rint(indices), rtol=0, atol=1e-9):
        raise ValueError("Position must lie on the grid")
    return tuple(int(i) for i in np.rint(indices))


@dataclass(frozen=True)
class Observation:
    robot_xy: Position
    rssi_dbm: float
    attempt: int
    boundary_blocked: bool


@dataclass(frozen=True)
class Evaluation:
    """Runner-only evaluation; do not pass this record to a controller."""
    reached_source: bool
    episode_ended: bool


@dataclass(frozen=True)
class GroundTruth:
    """Explicit evaluation/visualisation data, not a controller observation."""
    source_xy: Position
