"""Your MAP/Euclidean policy. No source truth, sensing or belief updates here."""

from dataclasses import dataclass
from enum import Enum
import math

import numpy as np

from .types import Action, EnvironmentConfig, Position, grid_index


class Command(Enum):
    WAIT = "wait"


@dataclass(frozen=True)
class SourceEstimate:
    xy: Position
    probability: float


def select_map(belief, config: EnvironmentConfig, rng) -> SourceEstimate:
    """Choose a maximum-probability cell; break exact MAP ties randomly."""
    belief = np.asarray(belief, dtype=float)
    if (belief.shape != config.shape or not np.isfinite(belief).all()
            or np.any(belief < 0) or not np.isclose(belief.sum(), 1)):
        raise ValueError("Belief must be finite, nonnegative, normalised and match config.shape")
    candidates = np.argwhere(belief == belief.max())  # Rows contain [y, x].
    y, x = candidates[int(rng.integers(len(candidates)))]
    x_axis, y_axis = config.grid_axes()
    return SourceEstimate((float(x_axis[x]), float(y_axis[y])), float(belief[y, x]))


def greedy_action(robot_xy, target: SourceEstimate, config: EnvironmentConfig,
                  past_action, rng) -> Action | Command:
    """Return one command, using last SUCCESSFUL direction only for movement ties."""
    grid_index(robot_xy, config)
    grid_index(target.xy, config)
    if past_action is not None and not isinstance(past_action, Action):
        raise ValueError("past_action must be a movement Action or None")
    if tuple(robot_xy) == target.xy:
        return Command.WAIT

    distances = {}
    for action in Action:
        dx, dy = action.value
        destination = (robot_xy[0] + dx * config.spacing_m,
                       robot_xy[1] + dy * config.spacing_m)
        distances[action] = math.dist(destination, target.xy)

    best_distance = min(distances.values())
    tied = [action for action, distance in distances.items()
            if math.isclose(distance, best_distance, rel_tol=1e-12, abs_tol=1e-12)]
    if past_action in tied:
        return past_action
    return tied[int(rng.integers(len(tied)))]
