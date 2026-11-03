"""Adapter checks for the disclosed waypoint/BFS benchmark definition."""

import unittest
from unittest.mock import patch

import numpy as np

from belief_search import Action
from belief_search.radio_control import GridRouter, measurement_position
from belief_search.radio_environment import RadioMapBank
from belief_search.radio_waypoint import waypoint_information_action


class WaypointTests(unittest.TestCase):
    def setUp(self):
        buildings = np.zeros((5, 5), dtype=bool)
        buildings[1:, 2] = True  # Passage is at y=4.
        self.bank = RadioMapBank(buildings, [[4, 0], [0, 4], [2, 0]],
                                 np.full((3, 5, 5), 128, dtype=np.uint8), "IRT2", 1)
        self.prior = np.full((3, 1), 1 / 3)
        self.router = GridRouter(self.bank)

    def test_remote_informative_waypoint_uses_shared_wall_detour(self):
        seen = []

        def gain(prior, destination, bank, sigma, rng, n_samples):
            np.testing.assert_array_equal(prior, self.prior)
            seen.append(destination)
            self.assertEqual(n_samples, 8)
            return 2.0 if destination == (4, 0) else 1.0

        with patch("belief_search.radio_waypoint.map_information_gain", side_effect=gain):
            action, details = waypoint_information_action(
                self.prior, (1, 0), self.bank, 2, np.random.default_rng(17), self.router)
        self.assertEqual(seen, [(4, 0), (0, 4)])  # Building waypoint excluded, not its source belief.
        self.assertEqual(details["selected"]["xy"], (4, 0))
        self.assertEqual(details["selected"]["distance_steps"], 11)
        self.assertEqual(action, Action.FORWARD)
        self.assertEqual(measurement_position(self.bank, (1, 0), action), (1, 1))

    def test_reproducible_ties_and_unchanged_whole_belief(self):
        original = self.prior.copy()
        a = waypoint_information_action(self.prior, (1, 0), self.bank, 2,
                                        np.random.default_rng(17), self.router)
        b = waypoint_information_action(self.prior, (1, 0), self.bank, 2,
                                        np.random.default_rng(17), self.router)
        self.assertEqual(a, b)
        np.testing.assert_array_equal(self.prior, original)
        self.assertEqual(len(a[1]["options"]), 2)

    def test_no_distinct_reachable_waypoint_is_explicit(self):
        bank = RadioMapBank(self.bank.buildings, [[0, 0], [2, 0]],
                            np.full((2, 5, 5), 128, dtype=np.uint8), "IRT2", 1)
        action, details = waypoint_information_action(
            np.full((2, 1), 0.5), (0, 0), bank, 2, np.random.default_rng(17), GridRouter(bank))
        self.assertIsNone(action)
        self.assertEqual(details["options"], [])


if __name__ == "__main__":
    unittest.main()
