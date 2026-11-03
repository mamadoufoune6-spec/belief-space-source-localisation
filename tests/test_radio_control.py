"""Codex-authored component checks for the explicitly authorised adapters."""

import unittest
from unittest.mock import patch

import numpy as np

from belief_search import Action, Observation
from belief_search.estimator import update_belief
from belief_search.radio_control import (
    GridRouter, candidate_likelihood, gaussian_likelihood, map_information_action,
    map_information_gain, measurement_position,
)
from belief_search.radio_environment import RadioMapBank, RadioEnvironmentConfig
from belief_search.radio_episode import run_radio_episode


def fixture_bank(simulation="IRT2", pixels=None):
    buildings = np.zeros((5, 5), dtype=bool)
    buildings[1:, 2] = True  # Wall x=2, y=0..3; passage at y=4.
    if pixels is None:
        pixels = np.full((2, 5, 5), 128, dtype=np.uint8)
    return RadioMapBank(buildings, np.array([[4, 0], [0, 4]]), pixels, simulation, 1)


class RadioControlTests(unittest.TestCase):
    def setUp(self):
        self.bank = fixture_bank()
        self.belief = np.array([[0.5], [0.5]])

    def test_gaussian_density_and_posterior_reference(self):
        sigma = 2
        means = [-40, -50]
        actual = gaussian_likelihood(means, -40, sigma)
        expected = np.array([[1], [np.exp(-12.5)]]) / (2 * np.sqrt(2 * np.pi))
        np.testing.assert_allclose(actual, expected)
        posterior = update_belief(self.belief, actual)
        np.testing.assert_allclose(posterior, expected / expected.sum())
        observation = Observation((0, 0), -70, 0, False)
        np.testing.assert_allclose(candidate_likelihood(observation, self.bank, sigma),
                                   gaussian_likelihood(self.bank.predicted_rssi((0, 0)), -70, sigma))

    def test_certain_and_identical_predictions_zero_gain(self):
        rng = np.random.default_rng(17)
        for prior in [np.array([[1.0], [0]]), self.belief]:
            self.assertAlmostEqual(map_information_gain(prior, (0, 0), self.bank, 2, rng),
                                   0, delta=1e-12)

    def test_hypotheticals_use_same_prior_and_leave_it_unchanged(self):
        original = self.belief.copy()
        posteriors = []

        def check(prior, likelihood):
            np.testing.assert_array_equal(prior, original)
            result = update_belief(prior, likelihood)
            posteriors.append(result)
            return result

        with patch("belief_search.radio_control.update_belief", side_effect=check):
            map_information_gain(self.belief, (0, 0), self.bank, 2, np.random.default_rng(17))
        self.assertEqual(len(posteriors), 8)
        np.testing.assert_array_equal(self.belief, original)
        self.assertTrue(all(np.isclose(p.sum(), 1) for p in posteriors))

    def test_seeded_information_action_and_rejection_position(self):
        self.assertEqual(measurement_position(self.bank, (1, 0), Action.RIGHT), (1, 0))
        self.assertEqual(measurement_position(self.bank, (0, 0), Action.LEFT), (0, 0))
        a = map_information_action(self.belief, (0, 0), self.bank, 2, np.random.default_rng(17))
        b = map_information_action(self.belief, (0, 0), self.bank, 2, np.random.default_rng(17))
        self.assertEqual(a, b)
        self.assertEqual(a[1][a[0]], max(a[1].values()))

    def test_route_around_wall_is_twelve_legal_steps(self):
        router = GridRouter(self.bank)
        self.assertEqual(router.distances_to((4, 0))[self.bank.array_index((0, 0))], 12)
        position, past = (0, 0), None
        rng = np.random.default_rng(17)
        for _ in range(12):
            action = router.action(position, (4, 0), past, rng)
            moved = measurement_position(self.bank, position, action)
            self.assertNotEqual(moved, position)
            position, past = moved, action
        self.assertEqual(position, (4, 0))
        self.assertIsNone(router.action((0, 0), (2, 0), None, rng))  # Blocked target.

    def test_model_mismatch_does_not_change_simulator_readings(self):
        primary_pixels = np.zeros((2, 5, 5), dtype=np.uint8)
        primary_pixels[1] = 255
        truth = fixture_bank(pixels=primary_pixels)
        secondary = fixture_bank("DPM")
        args = dict(robot_xy=(0, 0), source_id=0, seed=17, controller_seed=3,
                    policy="Greedy", config=RadioEnvironmentConfig(max_steps=1))
        a = run_radio_episode(truth, truth, **args)
        b = run_radio_episode(truth, secondary, **args)
        self.assertEqual(a["samples"][0]["rssi_dbm"], b["samples"][0]["rssi_dbm"])
        self.assertEqual(b["truth_simulation"], "IRT2")
        self.assertEqual(b["model_simulation"], "DPM")
        self.assertIsNone(b["error"])

    def test_runner_first_reach_on_last_attempt(self):
        with patch("belief_search.radio_episode.select_candidate", return_value=0):
            result = run_radio_episode(self.bank, self.bank, robot_xy=(3, 0), source_id=0,
                                       seed=17, controller_seed=3, policy="Greedy",
                                       config=RadioEnvironmentConfig(max_steps=1))
        self.assertEqual(result["stop_reason"], "reached")
        self.assertEqual(result["samples"][-1]["robot_xy"], (4, 0))
        self.assertEqual(result["used_budget"], 1)

    def test_zero_evidence_is_logged_not_repaired(self):
        with patch("belief_search.radio_episode.candidate_likelihood", return_value=np.zeros((2, 1))):
            result = run_radio_episode(self.bank, self.bank, robot_xy=(0, 0), source_id=0,
                                       seed=17, controller_seed=3, policy="Greedy")
        self.assertEqual(result["stop_reason"], "estimator_error")
        self.assertIsNone(result["samples"][0]["posterior"])
        self.assertIn("zero likelihood", result["error"])
        self.assertIn("ValueError", result["traceback"])
        self.assertEqual(result["used_budget"], 0)


if __name__ == "__main__":
    unittest.main()
