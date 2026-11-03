"""One-step information seeking; uses hypothetical readings, never source truth."""

import numpy as np

from .environment import mean_rssi
from .estimator import compute_likelihood, update_belief, belief_entropy
from .types import Action, EnvironmentConfig, Observation, grid_index


def expected_information_gain(belief, measurement_xy, config: EnvironmentConfig,
                              rng, n_samples: int = 8) -> float:
    """Current entropy minus mean hypothetical posterior entropy, in nats.

    measurement_xy is already the position AFTER the proposed move/rejection.
    rng is a planning generator, not the simulator's sensor generator.
    Small negative Monte Carlo estimates are returned without clipping.
    """
    belief = np.asarray(belief, dtype=float)
    if (belief.shape != config.shape or not np.isfinite(belief).all()
            or np.any(belief < 0) or np.any(belief > 1)
            or not np.isclose(belief.sum(), 1, rtol=0, atol=1e-8)):
        raise ValueError("Belief must be finite, nonnegative, normalised and match config.shape")
    grid_index(measurement_xy, config)
    if isinstance(n_samples, bool) or not isinstance(n_samples, int) or n_samples < 1:
        raise ValueError("n_samples must be a positive integer")
    if config.noise_std_db <= 0:
        raise ValueError("Gaussian likelihood requires positive noise_std_db")

    x_axis, y_axis = config.grid_axes()
    posterior_entropies = []
    for _ in range(n_samples):
        # Sample a possible source using the WHOLE belief, not just its MAP.
        index = int(rng.choice(belief.size, p=belief.ravel()))
        y, x = np.unravel_index(index, belief.shape)
        assumed_source = (float(x_axis[x]), float(y_axis[y]))
        predicted_mean = mean_rssi(measurement_xy, assumed_source, config)
        imagined_rssi = predicted_mean + rng.normal(0, config.noise_std_db)

        imagined_observation = Observation(tuple(measurement_xy), float(imagined_rssi),
                                          attempt=0, boundary_blocked=False)
        likelihood = compute_likelihood(imagined_observation, config)
        posterior = update_belief(belief, likelihood)  # Same prior for EVERY alternative.
        posterior_entropies.append(belief_entropy(posterior))

    gain = float(belief_entropy(belief) - np.mean(posterior_entropies))
    if not np.isfinite(gain):
        raise ValueError("Information-gain estimate must be finite")
    return gain


def information_action(belief, robot_xy, config: EnvironmentConfig,
                       rng, n_samples: int = 8) -> tuple[Action, dict[Action, float]]:
    """Score all four commands and return (selected action, gains in nats).

    Boundary rejections are evaluated at the unchanged position. Fresh planning
    samples are drawn for each action in enum order. Only exact gains are tied.
    This function neither moves the robot nor implements episode stopping.
    """
    x, y = grid_index(robot_xy, config)
    ny, nx = config.shape
    gains = {}
    for action in Action:
        dx, dy = action.value
        next_x, next_y = x + dx, y + dy
        if 0 <= next_x < nx and 0 <= next_y < ny:
            measurement_xy = (next_x * config.spacing_m, next_y * config.spacing_m)
        else:
            measurement_xy = tuple(robot_xy)
        gains[action] = expected_information_gain(belief, measurement_xy, config,
                                                  rng, n_samples)

    best_gain = max(gains.values())
    tied = [action for action, gain in gains.items() if gain == best_gain]
    chosen = tied[int(rng.integers(len(tied)))]
    return chosen, gains
