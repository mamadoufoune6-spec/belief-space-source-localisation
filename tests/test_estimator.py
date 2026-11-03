"""Checks for initialisation, likelihood, posterior updates, and entropy."""

from dataclasses import replace
import math
import unittest

import numpy as np

from belief_search import EnvironmentConfig, Observation
from belief_search.estimator import initialise_belief, compute_likelihood, update_belief, belief_entropy


class InitialisationTests(unittest.TestCase):
    def test_uniform_normalised_prior_and_independent_arrays(self):
        config = EnvironmentConfig()
        prior = initialise_belief(config)
        self.assertEqual(prior.shape, (25, 25))
        self.assertTrue(np.issubdtype(prior.dtype, np.floating))
        np.testing.assert_allclose(prior, 1 / 625)
        self.assertAlmostEqual(prior.sum(), 1)
        prior[0, 0] = 0
        self.assertEqual(initialise_belief(config)[0, 0], 1 / 625)

    def test_rectangular_grid(self):
        config = EnvironmentConfig(width_m=4, height_m=2)
        prior = initialise_belief(config)
        self.assertEqual(prior.shape, (5, 9))
        np.testing.assert_allclose(prior, 1 / 45)


class LikelihoodTests(unittest.TestCase):
    def setUp(self):
        self.config = EnvironmentConfig(width_m=4, height_m=2)

    def test_scalar_gaussian_reference_at_every_candidate(self):
        observation = Observation((0.5, 1), -47, 0, False)
        actual = compute_likelihood(observation, self.config)
        self.assertEqual(actual.shape, (5, 9))
        for j in range(5):
            for i in range(9):
                distance = max(math.hypot(i * 0.5 - 0.5, j * 0.5 - 1), 0.5)
                expected_mean = -40 - 20 * math.log10(distance)
                error = -47 - expected_mean
                expected = math.exp(-(error ** 2) / 8) / math.sqrt(8 * math.pi)
                self.assertAlmostEqual(actual[j, i], expected, places=14)
        self.assertTrue(np.isfinite(actual).all())
        self.assertTrue((actual >= 0).all())

    def test_matching_distance_has_peak_and_likelihood_is_not_normalised(self):
        likelihood = compute_likelihood(Observation((0, 0), -40, 0, False), self.config)
        peak = 1 / (2 * math.sqrt(2 * math.pi))
        self.assertAlmostEqual(likelihood[0, 2], peak)  # Source (1, 0).
        self.assertAlmostEqual(likelihood[2, 0], peak)  # Source (0, 1).
        self.assertAlmostEqual(likelihood.max(), peak)
        self.assertGreater(likelihood[0, 2], likelihood[0, 4])
        self.assertFalse(np.isclose(likelihood.sum(), 1))

    def test_cutoff_and_changed_parameters(self):
        config = replace(self.config, reference_power_dbm=-50, path_loss_exponent=3,
                         noise_std_db=1, reference_distance_m=2, min_distance_m=1)
        likelihood = compute_likelihood(Observation((0, 0), -40, 0, False), config)
        self.assertEqual(likelihood[0, 0], likelihood[0, 1])
        self.assertEqual(likelihood[0, 0], likelihood[0, 2])
        mean = -50 - 30 * math.log10(1 / 2)
        expected = math.exp(-0.5 * (-40 - mean) ** 2) / math.sqrt(2 * math.pi)
        self.assertAlmostEqual(likelihood[0, 0], expected)

    def test_invalid_observations_and_zero_noise(self):
        for observation in [Observation((0.1, 0), -40, 0, False),
                            Observation((0, 0), float("nan"), 0, False),
                            Observation((0, 0), float("inf"), 0, False)]:
            with self.assertRaises(ValueError):
                compute_likelihood(observation, self.config)
        with self.assertRaises(ValueError):
            compute_likelihood(Observation((0, 0), -40, 0, False),
                               replace(self.config, noise_std_db=0))


class PosteriorTests(unittest.TestCase):
    def test_hand_calculated_bayes_update(self):
        prior = np.array([[0.1, 0.2], [0.3, 0.4]])
        likelihood = np.array([[4.0, 3.0], [2.0, 1.0]])
        original_prior, original_likelihood = prior.copy(), likelihood.copy()
        posterior = update_belief(prior, likelihood)
        # Evidence = .4 + .6 + .6 + .4 = 2.
        np.testing.assert_allclose(posterior, [[0.2, 0.3], [0.3, 0.2]])
        self.assertAlmostEqual(posterior.sum(), 1)
        np.testing.assert_array_equal(prior, original_prior)
        np.testing.assert_array_equal(likelihood, original_likelihood)
        self.assertFalse(np.shares_memory(posterior, prior))

    def test_flat_likelihood_and_zero_support(self):
        prior = np.array([[0.0, 0.2], [0.3, 0.5]])
        np.testing.assert_allclose(update_belief(prior, np.full((2, 2), 7)), prior)
        posterior = update_belief(prior, [[100, 0], [1, 0]])
        np.testing.assert_array_equal(posterior, [[0, 0], [1, 0]])

    def test_tiny_products_and_common_scaling(self):
        prior = np.array([[1e-200, 2e-200, 1.0]])
        likelihood = np.array([[1e-200, 1e-200, 0.0]])
        # Direct multiplication would round both supported weights to zero.
        np.testing.assert_allclose(update_belief(prior, likelihood), [[1 / 3, 2 / 3, 0]])
        ordinary_prior = np.array([[0.25, 0.75]])
        np.testing.assert_allclose(
            update_belief(ordinary_prior, [[1e300, 2e300]]),
            update_belief(ordinary_prior, [[1, 2]]))

    def test_zero_evidence_is_explicit_error(self):
        for likelihood in [[[0, 0]], [[0, 5]]]:
            with self.assertRaisesRegex(ValueError, "zero likelihood"):
                update_belief([[1, 0]], likelihood)

    def test_invalid_inputs(self):
        cases = [([0.5, 0.5], [1, 1]), (np.zeros((0, 2)), np.zeros((0, 2))),
                 ([[0.5, 0.5]], [[1], [1]]), ([[0.2, 0.2]], [[1, 1]]),
                 ([[-0.5, 1.5]], [[1, 1]]), ([[np.nan, 1]], [[1, 1]]),
                 ([[np.inf, 0]], [[1, 1]]), ([[0.5, 0.5]], [[1, -1]]),
                 ([[0.5, 0.5]], [[1, np.nan]]), ([[0.5, 0.5]], [[1, np.inf]])]
        for prior, likelihood in cases:
            with self.subTest(prior=prior, likelihood=likelihood):
                with self.assertRaises(ValueError):
                    update_belief(prior, likelihood)

    def test_repeated_observations_and_real_sensor_likelihood(self):
        prior = np.array([[0.5, 0.5]])
        likelihood = np.array([[1, 2]])
        posterior = update_belief(update_belief(prior, likelihood), likelihood)
        np.testing.assert_allclose(posterior, [[0.2, 0.8]])
        config = EnvironmentConfig()
        posterior = update_belief(initialise_belief(config),
                                 compute_likelihood(Observation((0, 0), -40, 0, False), config))
        self.assertEqual(posterior.shape, config.shape)
        self.assertTrue(np.isfinite(posterior).all())
        self.assertAlmostEqual(posterior.sum(), 1)
        self.assertAlmostEqual(posterior[0, 2], posterior[2, 0])
        self.assertGreater(posterior[0, 2], posterior[24, 24])


class EntropyTests(unittest.TestCase):
    def test_uniform_prior_entropy_in_nats(self):
        prior = initialise_belief(EnvironmentConfig())
        self.assertAlmostEqual(belief_entropy(prior), math.log(625))

    def test_point_mass_zeros_and_input_preserved(self):
        belief = np.array([[0.0, 1.0], [0.0, 0.0]])
        original = belief.copy()
        self.assertEqual(belief_entropy(belief), 0.0)
        np.testing.assert_array_equal(belief, original)

    def test_known_two_candidate_entropy(self):
        expected = -0.25 * math.log(0.25) - 0.75 * math.log(0.75)
        self.assertAlmostEqual(belief_entropy(np.array([[0.25, 0.75]])), expected)


if __name__ == "__main__":
    unittest.main()
