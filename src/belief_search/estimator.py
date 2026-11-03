"""Bayesian estimator workspace.

Belief arrays use [y_index, x_index] and config.shape. Initialisation,
likelihood evaluation, posterior updates, and the user's entropy calculation
are available to the prescribed-action sequence runner.
"""

import numpy as np

from .types import EnvironmentConfig, Observation, grid_index


def initialise_belief(config: EnvironmentConfig) -> np.ndarray:
    """Return a uniform prior over all source-grid positions, in [y, x] order.

    No starting-separation exclusions are applied: that policy is not yet defined.
    """
    return np.full(config.shape, 1.0 / np.prod(config.shape), dtype=float)


def compute_likelihood(observation: Observation, config: EnvironmentConfig) -> np.ndarray:
    """Return Gaussian RSSI densities for every candidate source-grid position.

    These are p(observed RSSI | candidate source, known robot position), not
    posterior probabilities: do not normalise this array across source cells.
    Noise standard deviation must be positive; a zero-noise model has no ordinary
    Gaussian density. Extremely unlikely observations can underflow to zero;
    numerical handling of such likelihoods belongs in your posterior update.
    """
    grid_index(observation.robot_xy, config)
    if not np.isfinite(observation.rssi_dbm):
        raise ValueError("Observed RSSI must be finite")
    if config.noise_std_db <= 0:
        raise ValueError("Gaussian likelihood requires positive noise_std_db")

    x, y = config.grid_axes()
    source_x, source_y = np.meshgrid(x, y, indexing="xy")
    robot_x, robot_y = observation.robot_xy
    distances = np.hypot(source_x - robot_x, source_y - robot_y)
    effective_distances = np.maximum(distances, config.min_distance_m)
    expected_rssi = (config.reference_power_dbm - 10 * config.path_loss_exponent
                     * np.log10(effective_distances / config.reference_distance_m))
    standardised_error = (observation.rssi_dbm - expected_rssi) / config.noise_std_db
    return (np.exp(-0.5 * standardised_error ** 2)
            / (config.noise_std_db * np.sqrt(2 * np.pi)))


def update_belief(prior: np.ndarray, likelihood: np.ndarray) -> np.ndarray:
    """Return posterior = prior * likelihood / sum(prior * likelihood).

    Inputs must be matching nonempty 2D arrays, finite and nonnegative. The
    prior must sum to one; likelihood densities need not. Neither input is
    modified. Evaluate positive weights in log space to avoid losing tiny
    products or overflowing large densities before normalisation.

    Raise ValueError if there is no positive likelihood on the prior's support.
    Values already rounded to zero by compute_likelihood cannot be recovered;
    do not silently replace an impossible observation with a uniform belief.
    """
    prior = np.asarray(prior, dtype=float)
    likelihood = np.asarray(likelihood, dtype=float)
    if prior.ndim != 2 or prior.size == 0 or prior.shape != likelihood.shape:
        raise ValueError("Prior and likelihood must have the same nonempty 2D shape")
    for name, values in (("Prior", prior), ("Likelihood", likelihood)):
        if not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError(f"{name} must be finite and nonnegative")
    if np.any(prior > 1) or not np.isclose(prior.sum(), 1.0, rtol=0, atol=1e-8):
        raise ValueError("Prior must sum to one")

    support = (prior > 0) & (likelihood > 0)
    if not support.any():
        raise ValueError("Observation has zero likelihood over the prior's support")
    log_weights = np.log(prior[support]) + np.log(likelihood[support])
    posterior = np.zeros_like(prior)
    posterior[support] = np.exp(log_weights - log_weights.max())
    posterior /= posterior.sum()
    return posterior


def belief_entropy(belief: np.ndarray) -> float:
    """Return entropy in nats for a valid probability array, ignoring zero masses."""
    positive_belief = belief[belief > 0]
    H = np.sum(-positive_belief * np.log(positive_belief))
    return H
