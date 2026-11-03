"""Known-route oracle baselines under existing movement/sensing/budget rules."""

import unittest

import numpy as np

from belief_search.oracle import run_oracle_episode
from belief_search.radio_environment import RadioMapBank, RadioEnvironmentConfig


def fixture_bank():
    mask = np.zeros((5, 5), dtype=bool)
    mask[1:, 2] = True  # Source (4,0) needs the passage at y=4.
    return RadioMapBank(mask, [[4, 0], [0, 4], [2, 0]],
                        np.full((3, 5, 5), 128, dtype=np.uint8), "IRT2", 1)


class OracleTests(unittest.TestCase):
    def run_case(self, *, start=(0, 0), source=0, budget=13, seed=17, reserve=True):
        return run_oracle_episode(fixture_bank(), robot_xy=start, source_id=source,
                                  seed=seed, controller_seed=3,
                                  config=RadioEnvironmentConfig(max_steps=budget),
                                  reserve_last_unit=reserve)

    def test_shortest_wall_route_reaches_in_twelve_moves(self):
        result = self.run_case()
        self.assertEqual(result["optimal_steps_to_source"], 12)
        self.assertEqual(result["movement_attempts"], 12)
        self.assertEqual(result["stop_reason"], "reached")
        self.assertEqual(result["blocked_attempts"], 0)
        self.assertEqual(result["observations"], 13)
        self.assertEqual([s["remaining_route_steps"] for s in result["samples"]], list(range(12, -1, -1)))
        model = fixture_bank()
        for sample in result["samples"]:
            self.assertFalse(model.buildings[model.array_index(sample["robot_xy"])])
        self.assertEqual(result["localisation"], "given_not_inferred")

    def test_shared_reserve_gate_and_first_reach_precedence(self):
        stopped = self.run_case(budget=12)
        self.assertEqual(stopped["stop_reason"], "shared_reserve_stop")
        self.assertEqual(stopped["movement_attempts"], 11)
        self.assertFalse(stopped["reached"])
        full = self.run_case(budget=12, reserve=False)
        self.assertEqual(full["stop_reason"], "reached")
        self.assertEqual(full["movement_attempts"], 12)

    def test_initially_at_true_source_needs_no_move(self):
        result = self.run_case(start=(4, 0), budget=1)
        self.assertTrue(result["reached"])
        self.assertEqual(result["movement_attempts"], 0)
        self.assertEqual(result["optimal_steps_to_source"], 0)
        self.assertEqual(result["observations"], 1)

    def test_unreachable_true_source_is_not_teleported_or_removed(self):
        result = self.run_case(source=2)
        self.assertEqual(result["stop_reason"], "unreachable_true_source")
        self.assertFalse(result["reached"])
        self.assertIsNone(result["optimal_steps_to_source"])
        self.assertEqual(result["true_source_xy"], (2, 0))
        self.assertEqual(result["movement_attempts"], 0)

    def test_noise_does_not_select_actions_and_stream_is_reproducible(self):
        a, b, c = self.run_case(seed=0), self.run_case(seed=17), self.run_case(seed=0)
        self.assertEqual(a["samples"], c["samples"])
        self.assertEqual([s["command"] for s in a["samples"]], [s["command"] for s in b["samples"]])
        self.assertNotEqual(a["samples"][0]["rssi_dbm"], b["samples"][0]["rssi_dbm"])


if __name__ == "__main__":
    unittest.main()
