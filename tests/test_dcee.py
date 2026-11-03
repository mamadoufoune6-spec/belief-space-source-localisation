"""Reference and integration checks from the approved DCEE worksheet."""

import unittest
from unittest.mock import patch

import numpy as np

from belief_search import Action
from belief_search.dcee import position_task_cost, evaluate_first_move, dcee_action
from belief_search.radio_environment import RadioMapBank, RadioEnvironmentConfig
from belief_search.radio_episode import run_radio_episode


def fixture_bank(blocked=False):
    mask = np.zeros((9, 9), dtype=bool)
    if blocked:
        mask[2, 7] = True  # JSON coordinate (7,6).
    return RadioMapBank(mask, [[8, 6], [4, 6]],
                        np.full((2, 9, 9), 128, dtype=np.uint8), "IRT2", 1)


def command_result(action):
    return action, {a: float(i) for i, a in enumerate(Action)}, {}


class DCEETests(unittest.TestCase):
    def setUp(self):
        self.bank = fixture_bank()
        self.prior = np.array([[0.75], [0.25]])

    def test_position_cost_researcher_case(self):
        for xy, expected in [((6, 6), 4), ((5, 6), 7), ((7, 6), 3)]:
            self.assertEqual(position_task_cost(xy, self.bank.transmitter_xy, self.prior), expected)
        self.assertEqual(position_task_cost((8, 6), self.bank.transmitter_xy, [[1], [0]]), 0)
        with self.assertRaises(ValueError):
            position_task_cost((6, 6), self.bank.transmitter_xy, [[0.75], [0.75]])

    def test_eight_alternatives_same_prior_different_second_decisions(self):
        original = self.prior.copy()
        count = 0

        def posterior(prior, likelihood):
            nonlocal count
            np.testing.assert_array_equal(prior, original)
            count += 1
            return original.copy() if count <= 4 else original[::-1].copy()

        with patch("belief_search.dcee.update_belief", side_effect=posterior):
            score, details = evaluate_first_move(self.prior, (6, 5), Action.FORWARD,
                                                 self.bank, 2, np.random.default_rng(17))
        self.assertEqual(count, 8)
        self.assertEqual(score, 7)
        self.assertEqual(details["first_cost_m2"], 4)
        self.assertEqual([b["best_second_commands"] for b in details["branches"]],
                         [["RIGHT"]] * 4 + [["LEFT"]] * 4)
        np.testing.assert_array_equal(self.prior, original)

    def test_first_cost_uses_current_belief(self):
        with patch("belief_search.dcee.update_belief", return_value=self.prior[::-1].copy()):
            score, details = evaluate_first_move(self.prior, (6, 6), Action.RIGHT,
                                                 self.bank, 2, np.random.default_rng(17))
        self.assertEqual(details["first_cost_m2"], 3)
        self.assertEqual(details["mean_best_second_cost_m2"], 4)
        self.assertEqual(score, 7)

    def test_known_source_scores_and_reproducibility(self):
        args = (np.array([[1.0], [0]]), (6, 6), self.bank, 2)
        a = dcee_action(*args, np.random.default_rng(17))
        b = dcee_action(*args, np.random.default_rng(17))
        self.assertEqual(a, b)
        self.assertEqual(a[0], Action.RIGHT)
        self.assertEqual(a[1], {Action.RIGHT: 1, Action.LEFT: 13,
                                Action.FORWARD: 7, Action.BACKWARD: 7})

    def test_momentum_and_seeded_fallback_only_among_minima(self):
        def score(prior, xy, action, bank, sigma, rng, samples):
            return (4 if action in [Action.LEFT, Action.RIGHT] else 9), {}

        with patch("belief_search.dcee.evaluate_first_move", side_effect=score):
            a = dcee_action(self.prior, (6, 6), self.bank, 2,
                            np.random.default_rng(17), Action.RIGHT)
            self.assertEqual(a[0], Action.RIGHT)
            choices = [dcee_action(self.prior, (6, 6), self.bank, 2,
                                  np.random.default_rng(17))[0] for _ in range(2)]
            self.assertEqual(choices[0], choices[1])
            self.assertIn(choices[0], [Action.LEFT, Action.RIGHT])

    def test_blocked_first_command_scores_unchanged_position(self):
        bank = fixture_bank(blocked=True)
        score, details = evaluate_first_move([[1], [0]], (6, 6), Action.RIGHT,
                                             bank, 2, np.random.default_rng(17))
        self.assertEqual(details["first_xy"], (6, 6))
        self.assertEqual(details["first_cost_m2"], 4)
        self.assertEqual(score, 8)
        self.assertEqual(len(details["branches"]), 8)


class DCEERunnerTests(unittest.TestCase):
    def setUp(self):
        self.bank = fixture_bank()

    def run_case(self, *, start=(6, 6), source=0, budget=2, policy="DCEE", **extra):
        return run_radio_episode(self.bank, self.bank, robot_xy=start, source_id=source,
                                 seed=17, controller_seed=3, policy=policy,
                                 config=RadioEnvironmentConfig(max_steps=budget), **extra)

    def test_one_unit_remaining_stops_without_policy_or_bfs(self):
        with patch("belief_search.radio_episode.select_candidate", return_value=0), \
             patch("belief_search.radio_episode.GridRouter", side_effect=AssertionError("No BFS")), \
             patch("belief_search.radio_episode.dcee_action") as select:
            result = self.run_case(budget=1)
        self.assertEqual(result["stop_reason"], "insufficient_horizon_budget")
        self.assertEqual(result["used_budget"], 0)
        select.assert_not_called()

    def test_two_units_execute_one_then_leave_last_unused(self):
        with patch("belief_search.radio_episode.select_candidate", return_value=0), \
             patch("belief_search.radio_episode.dcee_action", return_value=command_result(Action.RIGHT)) as select:
            result = self.run_case()
        self.assertEqual(result["stop_reason"], "insufficient_horizon_budget")
        self.assertEqual(result["used_budget"], 1)
        self.assertEqual(result["samples"][-1]["robot_xy"], (7, 6))
        self.assertEqual(select.call_count, 1)

    def test_confirmation_with_one_unit_left_can_declare(self):
        with patch("belief_search.radio_episode.select_candidate", return_value=0), \
             patch("belief_search.radio_episode.dcee_action", return_value=command_result(Action.RIGHT)):
            result = self.run_case(start=(7, 6), source=1)
        self.assertEqual(result["stop_reason"], "declared")
        self.assertEqual(result["declaration"]["xy"], (8, 6))
        self.assertEqual(result["used_budget"], 1)
        self.assertEqual(result["samples"][-1]["command"], "WAIT")

    def test_changed_confirmation_uses_last_unit_then_stops(self):
        with patch("belief_search.radio_episode.select_candidate", side_effect=[0, 0, 1]), \
             patch("belief_search.radio_episode.dcee_action", return_value=command_result(Action.RIGHT)) as select:
            result = self.run_case(start=(7, 6), source=1)
        self.assertEqual(result["used_budget"], 2)
        self.assertEqual(result["failed_confirmations"], 1)
        self.assertEqual(result["stop_reason"], "budget_exhausted")
        self.assertIsNone(result["declaration"])
        self.assertEqual(select.call_count, 1)

    def test_reaching_precedes_confirmation_and_reserve_stop(self):
        with patch("belief_search.radio_episode.select_candidate", return_value=0), \
             patch("belief_search.radio_episode.dcee_action", return_value=command_result(Action.RIGHT)):
            result = self.run_case(start=(7, 6))
        self.assertEqual(result["stop_reason"], "reached")
        self.assertEqual(len(result["samples"]), 2)
        self.assertEqual(result["used_budget"], 1)

    def test_replans_after_each_real_reading(self):
        commands = [command_result(a) for a in [Action.RIGHT, Action.FORWARD, Action.LEFT]]
        with patch("belief_search.radio_episode.select_candidate", return_value=0), \
             patch("belief_search.radio_episode.dcee_action", side_effect=commands) as select:
            result = self.run_case(start=(3, 3), budget=4)
        self.assertEqual(select.call_count, 3)
        self.assertEqual([c.args[1] for c in select.call_args_list], [(3, 3), (4, 3), (4, 4)])
        for call, sample in zip(select.call_args_list, result["samples"]):
            np.testing.assert_allclose(call.args[0].ravel(), sample["posterior"])

    def test_blocking_costs_attempt_and_keeps_successful_history(self):
        self.bank = fixture_bank(blocked=True)
        with patch("belief_search.radio_episode.select_candidate", return_value=0), \
             patch("belief_search.radio_episode.dcee_action", return_value=command_result(Action.RIGHT)) as select:
            result = self.run_case(budget=3)
        self.assertEqual(result["used_budget"], 2)
        self.assertTrue(all(s["robot_xy"] == (6, 6) for s in result["samples"]))
        self.assertEqual([s["boundary_blocked"] for s in result["samples"]], [False, True, True])
        self.assertTrue(all(c.args[5] is None for c in select.call_args_list))
        self.assertNotEqual(result["samples"][0]["rssi_dbm"], result["samples"][1]["rssi_dbm"])

    def test_planning_zero_evidence_is_logged_without_repair(self):
        with patch("belief_search.radio_episode.select_candidate", return_value=0), \
             patch("belief_search.dcee.gaussian_likelihood", return_value=np.zeros((2, 1))):
            result = self.run_case()
        self.assertEqual(result["stop_reason"], "planning_error")
        self.assertIn("zero likelihood", result["error"])
        self.assertEqual(result["used_budget"], 0)
        self.assertIsNotNone(result["samples"][-1]["posterior"])

    def test_unassisted_greedy_and_shared_reserve_are_explicit(self):
        self.bank = fixture_bank(blocked=True)
        with patch("belief_search.radio_episode.select_candidate", return_value=0), \
             patch("belief_search.radio_episode.GridRouter", side_effect=AssertionError("No BFS")):
            result = self.run_case(budget=2, policy="Greedy", greedy_navigation="euclidean",
                                   reserve_last_unit=True)
        self.assertEqual(result["navigation_mode"], "euclidean")
        self.assertEqual(result["used_budget"], 1)
        self.assertEqual(result["stop_reason"], "shared_reserve_stop")
        self.assertTrue(result["samples"][-1]["boundary_blocked"])


if __name__ == "__main__":
    unittest.main()
