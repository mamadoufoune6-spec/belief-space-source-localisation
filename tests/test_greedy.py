"""Expected behaviours from the user's movement cases and confirmation rules."""

import unittest
from unittest.mock import patch

import numpy as np

from belief_search import Action, EnvironmentConfig, SearchEnvironment
from belief_search.greedy import Command, SourceEstimate, greedy_action, select_map
from belief_search.greedy_episode import run_greedy


class GreedyTests(unittest.TestCase):
    def setUp(self):
        self.config = EnvironmentConfig()

    def test_user_movement_cases(self):
        rng = np.random.default_rng(17)
        east = SourceEstimate((8, 6), 1)
        diagonal = SourceEstimate((7, 7), 1)
        self.assertEqual(greedy_action((6, 6), east, self.config, None, rng), Action.RIGHT)
        self.assertEqual(greedy_action((6, 6), diagonal, self.config, Action.FORWARD, rng),
                         Action.FORWARD)
        draws = [greedy_action((6, 6), diagonal, self.config, None, rng) for _ in range(20)]
        self.assertTrue(set(draws) <= {Action.FORWARD, Action.RIGHT})
        # Reproduce from identical generator state, not by expecting each draw to match.
        a = np.random.default_rng(42)
        b = np.random.default_rng(42)
        self.assertEqual(
            [greedy_action((6, 6), diagonal, self.config, None, a) for _ in range(20)],
            [greedy_action((6, 6), diagonal, self.config, None, b) for _ in range(20)])

    def test_map_coordinates_probability_and_seeded_ties(self):
        belief = np.zeros(self.config.shape)
        belief[12, 16] = 1  # [y, x] -> (8, 6).
        self.assertEqual(select_map(belief, self.config, np.random.default_rng(2)),
                         SourceEstimate((8, 6), 1))
        belief[12, 16] = belief[12, 4] = 0.5
        a, b = np.random.default_rng(7), np.random.default_rng(7)
        targets = [select_map(belief, self.config, a) for _ in range(20)]
        self.assertEqual(targets, [select_map(belief, self.config, b) for _ in range(20)])
        self.assertTrue(all(t.xy in {(2, 6), (8, 6)} for t in targets))

    def test_invalid_belief_and_position(self):
        for belief in [np.zeros((2, 2)), np.zeros(self.config.shape),
                       np.full(self.config.shape, np.nan)]:
            with self.assertRaises(ValueError):
                select_map(belief, self.config, np.random.default_rng(1))
        with self.assertRaises(ValueError):
            greedy_action((0.1, 0), SourceEstimate((8, 6), 1), self.config,
                          None, np.random.default_rng(1))

    def test_wait_is_fresh_stationary_and_not_a_movement_attempt(self):
        env = SearchEnvironment(runner_managed=True)
        initial, evaluation = env.reset(robot_xy=(6, 6), source_xy=(6, 6), seed=17)
        self.assertTrue(evaluation.reached_source)
        self.assertFalse(evaluation.episode_ended)
        next_obs, _ = env.wait()
        self.assertEqual(next_obs.robot_xy, initial.robot_xy)
        self.assertEqual(next_obs.attempt, 0)
        self.assertFalse(next_obs.boundary_blocked)
        self.assertNotEqual(initial.rssi_dbm, next_obs.rssi_dbm)

    def test_unchanged_map_declares_after_one_wait_without_charge(self):
        # Isolate the user-specified confirmation branch. The real Bayesian update
        # preserves this point-mass prior, regardless of ordinary finite readings.
        prior = np.zeros(self.config.shape)
        prior[12, 12] = 1
        with patch("belief_search.greedy_episode.initialise_belief", return_value=prior):
            result = run_greedy(robot_xy=(6, 6), source_xy=(12, 12),
                                seed=17, controller_seed=2)
        self.assertEqual(result.commands, (Command.WAIT,))
        self.assertEqual(result.declaration, SourceEstimate((6, 6), 1))
        self.assertEqual(result.used_budget, 0)
        self.assertFalse(result.evaluations[-1].reached_source)
        self.assertTrue(result.evaluations[-1].episode_ended)
        self.assertEqual(len(result.frames), 2)

    def run_confirmation_at_limit(self, changed):
        config = EnvironmentConfig(max_steps=1)
        target = SourceEstimate((0.5, 0), 0.6)
        after_wait = SourceEstimate((1, 0), 0.7) if changed else target
        # Controlled MAP outcomes test runner control flow, not a new RSSI oracle.
        with patch("belief_search.greedy_episode.select_map",
                   side_effect=[target, target, after_wait]):
            return run_greedy(robot_xy=(0, 0), source_xy=(12, 12), seed=17,
                              controller_seed=2, config=config)

    def test_unchanged_map_can_declare_at_budget_limit(self):
        result = self.run_confirmation_at_limit(changed=False)
        self.assertEqual(result.commands, (Action.RIGHT, Command.WAIT))
        self.assertEqual(result.stop_reason, "declared")
        self.assertEqual(result.used_budget, 1)

    def test_changed_map_at_limit_ends_without_another_movement(self):
        result = self.run_confirmation_at_limit(changed=True)
        self.assertEqual(result.commands, (Action.RIGHT, Command.WAIT))
        self.assertIsNone(result.declaration)
        self.assertEqual(result.stop_reason, "budget_exhausted")
        self.assertEqual(result.failed_confirmations, 1)
        self.assertEqual(result.remaining_budget, 0)
        self.assertEqual(result.frames[-1].observation.attempt, 1)

    def test_changed_map_resumes_with_last_successful_direction(self):
        targets = [SourceEstimate((6, 6.5), 0.6), SourceEstimate((6, 6.5), 0.7),
                   SourceEstimate((7, 7.5), 0.8), SourceEstimate((7, 7.5), 0.8)]
        with patch("belief_search.greedy_episode.select_map", side_effect=targets):
            result = run_greedy(robot_xy=(6, 6), source_xy=(12, 12), seed=17,
                                controller_seed=2, config=EnvironmentConfig(max_steps=3))
        # Forward succeeds, WAIT changes MAP, forward/right tie: retain forward.
        self.assertEqual(result.commands, (Action.FORWARD, Command.WAIT, Action.FORWARD))
        self.assertEqual(result.used_budget, 3)
        self.assertEqual(result.failed_confirmations, 1)

    def test_real_seeded_episodes_repeat_and_posteriors_stay_normalised(self):
        kwargs = dict(robot_xy=(0, 0), source_xy=(10, 8), seed=17, controller_seed=3)
        a, b = run_greedy(**kwargs), run_greedy(**kwargs)
        self.assertEqual(a.commands, b.commands)
        self.assertEqual(a.declaration, b.declaration)
        self.assertEqual(a.stop_reason, b.stop_reason)
        for left, right in zip(a.frames, b.frames):
            self.assertEqual(left.observation, right.observation)
            np.testing.assert_array_equal(left.belief, right.belief)
            self.assertAlmostEqual(left.belief.sum(), 1)


if __name__ == "__main__":
    unittest.main()
