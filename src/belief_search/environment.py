"""NumPy-only simulation. No estimator, controller, plotting, or global RNG state."""

import numpy as np

from .types import (
    Action, EnvironmentConfig, Evaluation, GroundTruth, Observation, grid_index,
)


def valid_actions(robot_xy, config: EnvironmentConfig) -> tuple[Action, ...]:
    x, y = grid_index(robot_xy, config)
    ny, nx = config.shape
    return tuple(a for a in Action
                 if 0 <= x + a.value[0] < nx and 0 <= y + a.value[1] < ny)


def mean_rssi(robot_xy, source_xy, config: EnvironmentConfig) -> float:
    """Forward sensor model only, not an observation or Bayesian likelihood."""
    distance = float(np.linalg.norm(np.asarray(robot_xy) - np.asarray(source_xy)))
    effective = max(distance, config.min_distance_m)
    return float(config.reference_power_dbm - 10 * config.path_loss_exponent
                 * np.log10(effective / config.reference_distance_m))


class SearchEnvironment:
    """Owned by the runner. Controllers receive observations, never this object."""

    def __init__(self, config: EnvironmentConfig | None = None, *, runner_managed=False):
        self.config = config if config is not None else EnvironmentConfig()
        # Greedy episodes stop on declaration/budget; legacy episodes stop on reaching.
        self.runner_managed = runner_managed
        self._initialised = False

    def reset(self, *, robot_xy, source_xy, seed: int) -> tuple[Observation, Evaluation]:
        """Use explicit starts until the random-start sampling policy is decided."""
        robot_index = grid_index(robot_xy, self.config)
        source_index = grid_index(source_xy, self.config)
        rng = np.random.default_rng(seed)
        self._robot_index = robot_index
        self._source_xy = self._position(source_index)
        self._rng = rng
        self._attempt = 0
        self._initialised = True
        return self._observe(blocked=False)

    def _position(self, index):
        return tuple(float(i * self.config.spacing_m) for i in index)

    def _observe(self, *, blocked):
        robot = self._position(self._robot_index)
        signal = mean_rssi(robot, self._source_xy, self.config)
        noisy = signal + self._rng.normal(0, self.config.noise_std_db)
        reached = np.linalg.norm(np.subtract(robot, self._source_xy)) <= self.config.success_radius_m
        self._ended = (not self.runner_managed
                       and bool(reached or self._attempt >= self.config.max_steps))
        return (Observation(robot, float(noisy), self._attempt, blocked),
                Evaluation(bool(reached), self._ended))

    def step(self, action: Action) -> tuple[Observation, Evaluation]:
        if not self._initialised:
            raise RuntimeError("Call reset before step")
        if self._ended:
            raise RuntimeError("Episode has ended; call reset")
        if not isinstance(action, Action):
            raise ValueError("action must be an Action member")
        if self._attempt >= self.config.max_steps:
            raise RuntimeError("Movement-attempt budget exhausted")
        robot = self._position(self._robot_index)
        blocked = action not in valid_actions(robot, self.config)
        self._attempt += 1
        if not blocked:
            x, y = self._robot_index
            dx, dy = action.value
            self._robot_index = (x + dx, y + dy)
        return self._observe(blocked=blocked)

    def wait(self) -> tuple[Observation, Evaluation]:
        """Fresh stationary sample; the runner settles its charge after updating MAP.

        Allowed at the movement limit, only in runner-managed episodes.
        Observation.attempt still counts movement attempts, not sensor samples.
        """
        if not self._initialised:
            raise RuntimeError("Call reset before wait")
        if not self.runner_managed:
            raise RuntimeError("WAIT requires runner-managed termination")
        return self._observe(blocked=False)

    def ground_truth(self) -> GroundTruth:
        if not self._initialised:
            raise RuntimeError("Call reset before requesting ground truth")
        return GroundTruth(self._source_xy)
