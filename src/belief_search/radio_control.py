"""Authorised candidate-map adapters; source truth is never a policy input."""

from collections import deque

import numpy as np

from .estimator import update_belief, belief_entropy
from .radio_environment import RadioMapBank
from .types import Action


def gaussian_likelihood(means_dbm, reading_dbm, noise_std_db):
    """Same Gaussian density as the analytic estimator; one row per candidate."""
    if not np.isfinite(noise_std_db) or noise_std_db <= 0 or not np.isfinite(reading_dbm):
        raise ValueError("Reading must be finite; Gaussian noise standard deviation must be positive")
    means = np.asarray(means_dbm, dtype=float).reshape(-1, 1)
    if not np.isfinite(means).all():
        raise ValueError("Candidate means must be finite")
    residual = (reading_dbm - means) / noise_std_db
    return np.exp(-0.5 * residual ** 2) / (noise_std_db * np.sqrt(2 * np.pi))


def candidate_likelihood(observation, model_bank, noise_std_db):
    return gaussian_likelihood(model_bank.predicted_rssi(observation.robot_xy),
                               observation.rssi_dbm, noise_std_db)


def validate_belief(belief, bank):
    values = np.asarray(belief, dtype=float)
    if (values.shape != (len(bank.transmitter_xy), 1) or not np.isfinite(values).all()
            or np.any(values < 0) or np.any(values > 1)
            or not np.isclose(values.sum(), 1, rtol=0, atol=1e-8)):
        raise ValueError("Belief must be normalised [candidate, 1] probabilities")
    return values


def select_candidate(belief, bank, rng):
    values = validate_belief(belief, bank).ravel()
    tied = np.flatnonzero(values == values.max())
    return int(tied[int(rng.integers(len(tied)))])


def measurement_position(bank: RadioMapBank, robot_xy, action):
    """Public geometry only: rejected commands measure at the unchanged position."""
    bank.array_index(robot_xy)
    if not isinstance(action, Action):
        raise ValueError("action must be a movement Action")
    x, y = robot_xy
    dx, dy = action.value
    destination = (int(x + dx), int(y + dy))
    height, width = bank.buildings.shape
    if not (0 <= destination[0] < width and 0 <= destination[1] < height):
        return tuple(robot_xy)
    if bank.buildings[bank.array_index(destination)]:
        return tuple(robot_xy)
    return destination


def map_information_gain(belief, measurement_xy, model_bank, noise_std_db, rng, n_samples=8):
    """Sample sources/readings from the MODEL bank, updating the same prior each time."""
    belief = validate_belief(belief, model_bank)
    if isinstance(n_samples, bool) or not isinstance(n_samples, int) or n_samples < 1:
        raise ValueError("n_samples must be a positive integer")
    means = model_bank.predicted_rssi(measurement_xy)
    entropies = []
    for _ in range(n_samples):
        assumed_id = int(rng.choice(len(means), p=belief.ravel()))
        reading = float(means[assumed_id] + rng.normal(0, noise_std_db))
        posterior = update_belief(belief, gaussian_likelihood(means, reading, noise_std_db))
        entropies.append(belief_entropy(posterior))
    gain = float(belief_entropy(belief) - np.mean(entropies))
    if not np.isfinite(gain):
        raise ValueError("Information gain must be finite")
    return gain  # Do not clip finite-sample negative estimates.


def map_information_action(belief, robot_xy, model_bank, noise_std_db, rng, n_samples=8):
    gains = {action: map_information_gain(
        belief, measurement_position(model_bank, robot_xy, action), model_bank,
        noise_std_db, rng, n_samples) for action in Action}
    best = max(gains.values())
    tied = [action for action in Action if gains[action] == best]
    return tied[int(rng.integers(len(tied)))], gains


class GridRouter:
    """Cached BFS distances over the known building mask; no radio/source truth.

    A fresh action is chosen toward the CURRENT MAP after every observation.
    The cache changes computation only, not the route objective or target.
    """

    def __init__(self, bank):
        self.bank = bank
        self._cache = {}

    def distances_to(self, target_xy):
        goal_row, goal_column = self.bank.array_index(target_xy)
        key = tuple(target_xy)
        if key in self._cache:
            return self._cache[key]
        height, width = self.bank.buildings.shape
        free = (~self.bank.buildings).ravel().tolist()
        distances = [-1] * (height * width)
        goal = goal_row * width + goal_column
        queue = deque()
        if free[goal]:
            distances[goal] = 0
            queue.append(goal)
        while queue:
            cell = queue.popleft()
            row, column = divmod(cell, width)
            neighbors = []
            if row > 0:
                neighbors.append(cell - width)
            if row + 1 < height:
                neighbors.append(cell + width)
            if column > 0:
                neighbors.append(cell - 1)
            if column + 1 < width:
                neighbors.append(cell + 1)
            for neighbor in neighbors:
                if free[neighbor] and distances[neighbor] < 0:
                    distances[neighbor] = distances[cell] + 1
                    queue.append(neighbor)
        field = np.array(distances, dtype=np.int32).reshape(height, width)
        field.setflags(write=False)
        self._cache[key] = field
        return field

    def action(self, robot_xy, target_xy, past_action, rng):
        field = self.distances_to(target_xy)
        distance = int(field[self.bank.array_index(robot_xy)])
        if distance <= 0:
            return None  # Already at target, or no route; caller distinguishes them.
        tied = []
        for action in Action:
            destination = measurement_position(self.bank, robot_xy, action)
            if int(field[self.bank.array_index(destination)]) == distance - 1:
                tied.append(action)
        if past_action in tied:
            return past_action
        return tied[int(rng.integers(len(tied)))]
