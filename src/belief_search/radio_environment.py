"""Published-map environment only; no new estimator, routing or control policy.

IRT2 is the primary simulation bank. Load DPM separately as a possible future
estimator model; never use the hidden active transmitter ID in a controller.
Zero PNG pixels are treated as the published clipped floor, not recovered RSSI
below that floor. Added Gaussian noise is an experiment setting, not field data.
"""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .types import Action, Evaluation, GroundTruth, Observation


def decode_rssi(pixels):
    """Original RadioMapSeer encoding: grayscale -> path gain -> RSSI in dBm."""
    return 23.0 - 147.0 + np.asarray(pixels, dtype=float) / 255.0 * (147.0 - 47.84)


@dataclass(frozen=True)
class RadioMapBank:
    buildings: np.ndarray  # [row, column], True means blocked.
    transmitter_xy: np.ndarray  # [candidate, (x,y)] in dataset JSON coordinates.
    gain_pixels: np.ndarray  # [candidate, row, column], published uint8 values.
    simulation: str
    map_id: int

    def __post_init__(self):
        buildings = np.array(self.buildings, dtype=bool, copy=True)
        positions = np.asarray(self.transmitter_xy, dtype=float)
        pixels = np.asarray(self.gain_pixels)
        if buildings.ndim != 2 or buildings.size == 0:
            raise ValueError("Building mask must be a nonempty 2D array")
        if (positions.ndim != 2 or positions.shape[1] != 2 or len(positions) == 0
                or not np.isfinite(positions).all() or not np.equal(positions, np.rint(positions)).all()):
            raise ValueError("Candidate positions must be integer (x,y) pairs")
        if pixels.shape != (len(positions), *buildings.shape) or pixels.dtype != np.uint8:
            raise ValueError("Gain pixels must be uint8 [candidate, row, column]")
        positions = positions.astype(int)
        height, width = buildings.shape
        if np.any(positions < 0) or np.any(positions >= (width, height)):
            raise ValueError("Candidate position is outside the map")
        pixels = pixels.copy()
        for values in (buildings, positions, pixels):
            values.setflags(write=False)
        object.__setattr__(self, "buildings", buildings)
        object.__setattr__(self, "transmitter_xy", positions)
        object.__setattr__(self, "gain_pixels", pixels)

    def array_index(self, robot_xy):
        """Convert known JSON-frame (x,y) metres to PNG [row,column]."""
        xy = np.asarray(robot_xy, dtype=float)
        if xy.shape != (2,) or not np.isfinite(xy).all() or not np.equal(xy, np.rint(xy)).all():
            raise ValueError("Robot position must be an integer (x,y) cell")
        x, y = map(int, xy)
        height, width = self.buildings.shape
        if not (0 <= x < width and 0 <= y < height):
            raise ValueError("Position is outside the map")
        return height - 1 - y, x

    def predicted_rssi(self, robot_xy):
        """Public forward predictions for ALL hypotheses, not an active-source lookup."""
        row, column = self.array_index(robot_xy)
        return decode_rssi(self.gain_pixels[:, row, column])


def load_radio_map_bank(data_root, map_id: int, simulation="IRT2") -> RadioMapBank:
    """Load original-release files with the coordinate/encoding audit preserved."""
    if isinstance(map_id, bool) or not isinstance(map_id, int) or map_id < 1:
        raise ValueError("map_id must be a positive integer")
    if simulation not in {"IRT2", "DPM"}:
        raise ValueError("Choose IRT2 (primary) or DPM (secondary)")
    root = Path(data_root)
    buildings = np.asarray(Image.open(root / f"png/buildings_complete/{map_id}.png"))
    if buildings.shape != (256, 256) or set(np.unique(buildings)) != {0, 255}:
        raise ValueError("Expected the published 256x256 binary building mask")
    positions = np.asarray(json.loads((root / f"antenna/{map_id}.json").read_text()))
    if positions.shape != (80, 2):
        raise ValueError("Expected 80 published transmitter candidates")
    images = [np.asarray(Image.open(root / f"gain/{simulation}/{map_id}_{i}.png"))
              for i in range(80)]
    return RadioMapBank(buildings > 0, positions, np.stack(images), simulation, map_id)


@dataclass(frozen=True)
class RadioEnvironmentConfig:
    noise_std_db: float = 2.0
    success_radius_m: float = 0.5
    max_steps: int = 80

    def __post_init__(self):
        for value in (self.noise_std_db, self.success_radius_m):
            if not np.isfinite(value) or value < 0:
                raise ValueError("Noise and reaching radius must be finite and nonnegative")
        if isinstance(self.max_steps, bool) or not isinstance(self.max_steps, int) or self.max_steps < 1:
            raise ValueError("max_steps must be a positive integer")


class RadioMapEnvironment:
    """Runner-owned environment. Controllers receive observations and candidate bank.

    Four one-metre cardinal moves; buildings/boundaries reject with feedback.
    Source candidates are not filtered or moved, even when their markers overlap
    buildings. Such episodes can be unreachable and need explicit benchmark review.
    """

    def __init__(self, bank: RadioMapBank, config=None, *, runner_managed=False):
        self.bank = bank
        self.config = config if config is not None else RadioEnvironmentConfig()
        self.runner_managed = runner_managed
        self._initialised = False

    def reset(self, *, robot_xy, source_id: int, seed: int):
        row, column = self.bank.array_index(robot_xy)
        if self.bank.buildings[row, column]:
            raise ValueError("Robot cannot start inside a building")
        if (isinstance(source_id, bool) or not isinstance(source_id, int)
                or not 0 <= source_id < len(self.bank.transmitter_xy)):
            raise ValueError("source_id must identify a published candidate")
        self._robot_xy = tuple(map(int, robot_xy))
        self._source_id = source_id  # Private simulator choice, never in Observation.
        self._rng = np.random.default_rng(seed)
        self._attempt = 0
        self._initialised = True
        return self._observe(blocked=False)

    def _observe(self, *, blocked):
        mean = float(self.bank.predicted_rssi(self._robot_xy)[self._source_id])
        rssi = mean + self._rng.normal(0, self.config.noise_std_db)
        source = self.bank.transmitter_xy[self._source_id]
        reached = bool(np.linalg.norm(np.subtract(self._robot_xy, source)) <= self.config.success_radius_m)
        self._ended = (not self.runner_managed
                       and (reached or self._attempt >= self.config.max_steps))
        return (Observation(self._robot_xy, float(rssi), self._attempt, blocked),
                Evaluation(reached, self._ended))

    def _check_active(self):
        if not self._initialised:
            raise RuntimeError("Call reset before requesting a reading or movement")
        if self._ended:
            raise RuntimeError("Episode has ended; call reset")

    def step(self, action: Action):
        self._check_active()
        if not isinstance(action, Action):
            raise ValueError("action must be a movement Action")
        if self._attempt >= self.config.max_steps:
            raise RuntimeError("Movement-attempt budget exhausted")
        x, y = self._robot_xy
        dx, dy = action.value
        destination = (x + dx, y + dy)
        height, width = self.bank.buildings.shape
        blocked = not (0 <= destination[0] < width and 0 <= destination[1] < height)
        if not blocked:
            row, column = self.bank.array_index(destination)
            blocked = bool(self.bank.buildings[row, column])
        self._attempt += 1
        if not blocked:
            self._robot_xy = destination
        return self._observe(blocked=blocked)

    def wait(self):
        """Fresh stationary reading; confirmation charging belongs to a future runner."""
        self._check_active()
        return self._observe(blocked=False)

    def ground_truth(self):
        if not self._initialised:
            raise RuntimeError("Call reset before requesting ground truth")
        return GroundTruth(tuple(map(int, self.bank.transmitter_xy[self._source_id])))
