"""User's three theoretical cases plus routine purity and interface checks."""

from dataclasses import replace
from unittest.mock import patch
import unittest

import numpy as np

from belief_search import Action, EnvironmentConfig, SearchEnvironment
from belief_search.estimator import update_belief
from belief_search.information import expected_information_gain, information_action


class InformationTests(unittest.TestCase):
    def setUp(self):
        self.config = EnvironmentConfig()
        self.belief = np.zeros(self.config.shape)
        self.belief[12, 8] = self.belief[12, 16] = 0.5  # (4,6), (8,6).

    def test_certain_source_zero_gain(self):
        belief = np.zeros(self.config.shape)
        belief[12, 16] = 1
        gain = expected_information_gain(belief, (6.5, 6), self.config,
                                         np.random.default_rng(17))
        self.assertAlmostEqual(gain, 0, delta=1e-12)

    def test_identical_predictions_preserve_ambiguity(self):
        gain = expected_information_gain(self.belief, (6, 6), self.config,
                                         np.random.default_rng(17))
        self.assertAlmostEqual(gain, 0, delta=1e-12)

    def test_user_informative_position_ordering(self):
        forward = expected_information_gain(self.belief, (6, 6.5), self.config,
                                            np.random.default_rng(17), n_samples=256)
        right = expected_information_gain(self.belief, (6.5, 6), self.config,
                                          np.random.default_rng(17), n_samples=256)
        # Larger diagnostic budget for the theoretical ordering, not a guarantee
        # that every 8-sample estimate ranks actions correctly.
        self.assertAlmostEqual(forward, 0, delta=1e-12)
        self.assertGreater(right, forward)

    def test_repeatable_scores_and_action(self):
        a, gains_a = information_action(self.belief, (6, 6), self.config,
                                        np.random.default_rng(42))
        b, gains_b = information_action(self.belief, (6, 6), self.config,
                                        np.random.default_rng(42))
        self.assertEqual((a, gains_a), (b, gains_b))
        self.assertEqual(gains_a[a], max(gains_a.values()))
        self.assertEqual(set(gains_a), set(Action))

    def test_each_alternative_uses_original_prior_and_is_normalised(self):
        original = self.belief.copy()
        posteriors = []

        def record_update(prior, likelihood):
            np.testing.assert_array_equal(prior, original)
            posterior = update_belief(prior, likelihood)
            posteriors.append(posterior)
            return posterior

        with patch("belief_search.information.update_belief", side_effect=record_update):
            expected_information_gain(self.belief, (6.5, 6), self.config,
                                      np.random.default_rng(17))
        self.assertEqual(len(posteriors), 8)
        np.testing.assert_array_equal(self.belief, original)
        for posterior in posteriors:
            self.assertTrue(np.isfinite(posterior).all())
            self.assertTrue((posterior >= 0).all())
            self.assertAlmostEqual(posterior.sum(), 1)

    def test_boundary_commands_use_unchanged_measurement_position(self):
        with patch("belief_search.information.expected_information_gain", return_value=0) as score:
            action, gains = information_action(self.belief, (0, 0), self.config,
                                               np.random.default_rng(17))
        positions = {a: call.args[1] for a, call in zip(Action, score.call_args_list)}
        self.assertEqual(positions, {Action.FORWARD: (0, 0.5), Action.BACKWARD: (0, 0),
                                     Action.LEFT: (0, 0), Action.RIGHT: (0.5, 0)})
        self.assertIn(action, Action)

    def test_exact_ties_choose_from_tied_best_set(self):
        selected = []
        for seed in range(10):
            with patch("belief_search.information.expected_information_gain",
                       side_effect=[0.0, 0.0, 1.0, 1.0]):
                action, _ = information_action(self.belief, (6, 6), self.config,
                                               np.random.default_rng(seed))
            self.assertIn(action, (Action.LEFT, Action.RIGHT))
            selected.append(action)
        self.assertEqual(set(selected), {Action.LEFT, Action.RIGHT})

    def test_hidden_truth_and_sensor_noise_are_not_used(self):
        for truth in [(0, 0), (12, 12)]:
            with self.subTest(truth=truth):
                actual, control = SearchEnvironment(), SearchEnvironment()
                start = dict(robot_xy=(6, 6), source_xy=truth, seed=17)
                before = actual.reset(**start)
                control.reset(**start)
                action, gains = information_action(self.belief, before[0].robot_xy, self.config,
                                                   np.random.default_rng(42))
                reference = information_action(self.belief, (6, 6), self.config,
                                               np.random.default_rng(42))
                self.assertEqual((action, gains), reference)
                # Planning must not advance the real environment or its sensor RNG.
                self.assertEqual(actual.step(Action.RIGHT), control.step(Action.RIGHT))

    def test_finite_sample_negative_estimate_is_not_clipped(self):
        # Routine arithmetic check of the agreed gain formula, not a new RSSI oracle.
        with patch("belief_search.information.belief_entropy", side_effect=[0.8, 0.5]):
            gain = expected_information_gain(self.belief, (6.5, 6), self.config,
                                             np.random.default_rng(17), n_samples=1)
        self.assertAlmostEqual(gain, -0.3)

    def test_invalid_inputs(self):
        rng = np.random.default_rng(17)
        for belief in [np.zeros((2, 2)), np.zeros(self.config.shape),
                       np.full(self.config.shape, np.nan), -self.belief]:
            with self.assertRaises(ValueError):
                expected_information_gain(belief, (6, 6), self.config, rng)
        for count in [0, -1, True, 1.5]:
            with self.assertRaises(ValueError):
                expected_information_gain(self.belief, (6, 6), self.config, rng, count)
        with self.assertRaises(ValueError):
            expected_information_gain(self.belief, (6.1, 6), self.config, rng)
        with self.assertRaises(ValueError):
            expected_information_gain(self.belief, (6, 6),
                                      replace(self.config, noise_std_db=0), rng)


if __name__ == "__main__":
    unittest.main()
