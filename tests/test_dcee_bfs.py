"""Disclosed route-cost and unreachable-penalty examples for the BFS variant."""

import unittest

import numpy as np

from belief_search import Action
from belief_search.dcee import dcee_action, position_task_cost
from belief_search.dcee_bfs import BFSTaskCost
from belief_search.radio_control import GridRouter
from belief_search.radio_environment import RadioMapBank, RadioEnvironmentConfig
from belief_search.radio_episode import run_radio_episode


def bank():
    mask = np.zeros((5, 5), dtype=bool)
    mask[1:, 2] = True  # Passage at y=4.
    return RadioMapBank(mask, [[4, 0], [0, 4], [2, 0]],
                        np.full((3, 5, 5), 128, dtype=np.uint8), "IRT2", 1)


class DCEEBFSTests(unittest.TestCase):
    def test_squared_route_distance_differs_from_euclidean(self):
        model = bank()
        cost = BFSTaskCost(model)
        belief = np.array([[0.75], [0.25], [0]])
        self.assertEqual(cost((0, 0), model.transmitter_xy, belief), 112)
        self.assertEqual(position_task_cost((0, 0), model.transmitter_xy, belief), 16)

    def test_blocked_source_retains_probability_with_fixed_penalty(self):
        model = bank()
        cost = BFSTaskCost(model)
        self.assertEqual(cost.unreachable_distance_m, 22)
        self.assertEqual(cost.unreachable_ids((0, 0)), [2])
        self.assertEqual(cost((0, 0), model.transmitter_xy, [[0], [0], [1]]), 484)
        self.assertEqual(cost((1, 1), model.transmitter_xy, [[0], [0], [1]]), 484)
        self.assertEqual(cost((0, 0), model.transmitter_xy, [[0.5], [0], [0.5]]), 314)

    def test_two_stage_bfs_cost_starts_legal_wall_detour(self):
        model = bank()
        router = GridRouter(model)
        cost = BFSTaskCost(model, router)
        action, costs, details = dcee_action([[1], [0], [0]], (1, 0), model, 2,
                                            np.random.default_rng(17), task_cost=cost)
        self.assertEqual(action, Action.FORWARD)
        self.assertEqual(costs[Action.FORWARD], 181)
        self.assertEqual(costs[Action.RIGHT], 221)
        self.assertEqual(len(details), 4)  # Blocked commands still evaluated.

    def test_runner_uses_explicit_cost_variant(self):
        model = bank()
        cost = BFSTaskCost(model)
        result = run_radio_episode(model, model, robot_xy=(0, 0), source_id=0,
                                   seed=17, controller_seed=3, policy="DCEE",
                                   config=RadioEnvironmentConfig(max_steps=2),
                                   dcee_navigation="bfs", dcee_task_cost=cost)
        self.assertEqual(result["navigation_mode"], "two_stage_bfs_cost")
        self.assertEqual(result["unreachable_distance_m"], 22)
        self.assertEqual(result["samples_per_action"], 8)
        self.assertIsNone(result["error"])


if __name__ == "__main__":
    unittest.main()
