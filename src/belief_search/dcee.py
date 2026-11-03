"""The researcher's two-stage task-cost design, consolidated on 29 October.

Only the first command is executed. Hypothetical readings change the best
second command within each branch; neither real belief nor environment moves.
"""

import numpy as np

from .estimator import update_belief
from .radio_control import gaussian_likelihood, measurement_position, validate_belief
from .types import Action


def position_task_cost(robot_xy, candidate_xy, belief):
    """Sum source probabilities times squared distances; return square metres."""
    position = np.asarray(robot_xy, dtype=float)
    sources = np.asarray(candidate_xy, dtype=float)
    probabilities = np.asarray(belief, dtype=float)
    if position.shape != (2,) or not np.isfinite(position).all():
        raise ValueError("Robot position must contain two finite coordinates")
    if (sources.ndim != 2 or sources.shape[1] != 2 or len(sources) == 0
            or not np.isfinite(sources).all()):
        raise ValueError("Source coordinates must be finite [candidate, 2]")
    if (probabilities.shape != (len(sources), 1)
            or not np.isfinite(probabilities).all() or np.any(probabilities < 0)
            or np.any(probabilities > 1)
            or not np.isclose(probabilities.sum(), 1, rtol=0, atol=1e-8)):
        raise ValueError("Belief must be normalised [candidate, 1] probabilities")
    squared_distances = np.sum((sources - position) ** 2, axis=1)
    cost = float(np.dot(probabilities.ravel(), squared_distances))
    if not np.isfinite(cost):
        raise ValueError("Task cost must be finite")
    return cost


def evaluate_first_move(belief, robot_xy, action, model_bank, noise_std_db,
                        rng, n_samples=8, task_cost=None):
    """Current first cost plus mean branch-specific minimum second cost.

    Branch diagnostics store readings and costs. Their posteriors can be
    reconstructed from those readings and the unchanged input belief.
    """
    belief = validate_belief(belief, model_bank)
    if not np.isfinite(noise_std_db) or noise_std_db <= 0:
        raise ValueError("Gaussian noise standard deviation must be positive")
    if isinstance(n_samples, bool) or not isinstance(n_samples, int) or n_samples < 1:
        raise ValueError("n_samples must be a positive integer")
    cost_function = position_task_cost if task_cost is None else task_cost
    first_xy = measurement_position(model_bank, robot_xy, action)
    first_cost = cost_function(first_xy, model_bank.transmitter_xy, belief)
    means = model_bank.predicted_rssi(first_xy)
    second_positions = {a: measurement_position(model_bank, first_xy, a) for a in Action}
    branches = []
    for _ in range(n_samples):
        assumed_id = int(rng.choice(len(means), p=belief.ravel()))
        reading = float(means[assumed_id] + rng.normal(0, noise_std_db))
        posterior = update_belief(belief, gaussian_likelihood(means, reading, noise_std_db))
        second_costs = {a.name: cost_function(xy, model_bank.transmitter_xy, posterior)
                        for a, xy in second_positions.items()}
        best = min(second_costs.values())
        branches.append({"assumed_candidate_id": assumed_id, "reading_dbm": reading,
                         "second_costs_m2": second_costs, "best_cost_m2": best,
                         "best_second_commands": [a for a, c in second_costs.items() if c == best]})
    mean_second = float(np.mean([branch["best_cost_m2"] for branch in branches]))
    score = first_cost + mean_second
    return score, {"first_xy": first_xy, "first_cost_m2": first_cost,
                   "mean_best_second_cost_m2": mean_second, "branches": branches}


def dcee_action(belief, robot_xy, model_bank, noise_std_db, rng,
                past_action=None, n_samples=8, task_cost=None):
    """Return (first command, four costs, hypothetical diagnostics); no BFS."""
    if past_action is not None and not isinstance(past_action, Action):
        raise ValueError("past_action must be a movement Action or None")
    costs, details = {}, {}
    for action in Action:
        if task_cost is None:
            costs[action], details[action] = evaluate_first_move(
                belief, robot_xy, action, model_bank, noise_std_db, rng, n_samples)
        else:
            costs[action], details[action] = evaluate_first_move(
                belief, robot_xy, action, model_bank, noise_std_db, rng, n_samples,
                task_cost=task_cost)
    best = min(costs.values())
    tied = [action for action in Action if costs[action] == best]
    selected = past_action if past_action in tied else tied[int(rng.integers(len(tied)))]
    return selected, costs, details
