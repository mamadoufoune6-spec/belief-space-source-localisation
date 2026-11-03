"""Shared stopping and accounting must not depend on the movement policy."""

from unittest.mock import patch
import unittest

from belief_search import Action, EnvironmentConfig
from belief_search.greedy import Command, SourceEstimate
from belief_search.greedy_episode import run_episode, run_greedy


class SharedEpisodeTests(unittest.TestCase):
    def test_greedy_entry_point_preserves_existing_record(self):
        kwargs = dict(robot_xy=(0, 0), source_xy=(10, 8), seed=17, controller_seed=3)
        a, b = run_greedy(**kwargs), run_episode(**kwargs)
        self.assertEqual(a.commands, b.commands)
        self.assertEqual(a.declaration, b.declaration)
        self.assertEqual(a.used_budget, b.used_budget)
        self.assertEqual(a.entropies_nats, b.entropies_nats)
        self.assertEqual(a.final_estimate, b.final_estimate)

    def test_movement_callback_cannot_override_confirmation_at_budget_limit(self):
        target = SourceEstimate((0.5, 0), 0.6)
        calls = []

        def movement(belief, robot_xy, config, rng):
            calls.append(robot_xy)
            self.assertEqual(belief.shape, config.shape)
            self.assertAlmostEqual(belief.sum(), 1)
            return Action.RIGHT

        with patch("belief_search.greedy_episode.select_map", return_value=target):
            result = run_episode(robot_xy=(0, 0), source_xy=(12, 12), seed=17,
                                 controller_seed=3, config=EnvironmentConfig(max_steps=1),
                                 movement_policy=movement)
        self.assertEqual(calls, [(0, 0)])
        self.assertEqual(result.commands, (Action.RIGHT, Command.WAIT))
        self.assertEqual(result.declaration, target)
        self.assertEqual(result.used_budget, 1)

    def test_callback_failure_does_not_silently_change_policy(self):
        def broken(*args):
            raise ValueError("policy failure")

        with patch("belief_search.greedy_episode.select_map",
                   return_value=SourceEstimate((12, 12), 0.5)):
            with self.assertRaisesRegex(ValueError, "policy failure"):
                run_episode(robot_xy=(0, 0), source_xy=(12, 12), seed=17,
                            controller_seed=3, movement_policy=broken)

    def test_first_reach_ends_without_confirmation_or_policy_call(self):
        def must_not_move(*args):
            self.fail("Movement requested after reaching")

        result = run_episode(robot_xy=(0, 0), source_xy=(0.5, 0), seed=17,
                             controller_seed=3, movement_policy=must_not_move,
                             stop_on_reach=True)
        self.assertEqual(result.stop_reason, "reached")
        self.assertEqual(result.commands, ())
        self.assertIsNone(result.declaration)
        self.assertEqual(result.used_budget, 0)
        self.assertEqual(len(result.frames), 1)
        self.assertTrue(result.evaluations[-1].episode_ended)

    def test_first_reach_on_final_attempt_beats_timeout_and_wait(self):
        target = SourceEstimate((1, 0), 0.6)
        with patch("belief_search.greedy_episode.select_map", return_value=target):
            result = run_episode(robot_xy=(0, 0), source_xy=(1, 0), seed=17,
                                 controller_seed=3, config=EnvironmentConfig(max_steps=1),
                                 movement_policy=lambda *args: Action.RIGHT, stop_on_reach=True)
        self.assertEqual(result.stop_reason, "reached")
        self.assertEqual(result.commands, (Action.RIGHT,))
        self.assertEqual(result.used_budget, 1)
        self.assertTrue(result.evaluations[-1].reached_source)

    def test_greedy_first_reach_option(self):
        result = run_greedy(robot_xy=(6, 6), source_xy=(6, 6), seed=17,
                            controller_seed=3, stop_on_reach=True)
        self.assertEqual(result.stop_reason, "reached")
        self.assertEqual(result.commands, ())


if __name__ == "__main__":
    unittest.main()
