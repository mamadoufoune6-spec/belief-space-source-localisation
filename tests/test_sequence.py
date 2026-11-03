import math
import unittest
from unittest.mock import patch

import numpy as np

from belief_search import Action, EnvironmentConfig, SearchEnvironment, Observation
from belief_search.sequence import run_sequence
from belief_search.estimator import compute_likelihood


class SequenceTests(unittest.TestCase):
    def run_case(self, actions=(), **kwargs):
        inputs = dict(robot_xy=(0, 0), source_xy=(12, 12), seed=17)
        inputs.update(kwargs)
        return run_sequence(SearchEnvironment(), actions, **inputs)

    def test_initial_observation_is_assimilated_without_move(self):
        result = self.run_case()
        self.assertEqual(len(result.frames), 1)
        self.assertEqual(result.frames[0].observation.attempt, 0)
        likelihood = compute_likelihood(result.frames[0].observation, EnvironmentConfig())
        # Uniform prior cancels: posterior is likelihood divided by its sum.
        np.testing.assert_allclose(result.frames[0].belief, likelihood / likelihood.sum())
        self.assertAlmostEqual(result.initial_entropy_nats, math.log(625))
        self.assertFalse(result.final_evaluation.episode_ended)

    def test_recording_each_update_and_estimator_separation(self):
        with patch("belief_search.sequence.compute_likelihood", wraps=compute_likelihood) as call:
            result = self.run_case([Action.RIGHT, Action.FORWARD, Action.LEFT])
        self.assertEqual(len(result.frames), 4)
        self.assertEqual(len(result.entropies_nats), 4)
        self.assertEqual(len(result.evaluations), 4)
        self.assertEqual(call.call_count, 4)
        self.assertTrue(all(isinstance(args.args[0], Observation) for args in call.call_args_list))
        for frame in result.frames:
            self.assertTrue(np.isfinite(frame.belief).all())
            self.assertAlmostEqual(frame.belief.sum(), 1)
            self.assertFalse(frame.belief.flags.writeable)
        self.assertFalse(np.shares_memory(result.frames[0].belief, result.frames[-1].belief))

    def test_blocked_step_assimilates_fresh_observation(self):
        result = self.run_case([Action.LEFT])
        first, second = result.frames
        self.assertEqual(first.observation.robot_xy, second.observation.robot_xy)
        self.assertEqual(second.observation.attempt, 1)
        self.assertTrue(second.observation.boundary_blocked)
        self.assertNotEqual(first.observation.rssi_dbm, second.observation.rssi_dbm)
        self.assertFalse(np.allclose(first.belief, second.belief))

    def test_success_timeout_and_unused_actions(self):
        result = self.run_case([Action.RIGHT] * 10, source_xy=(1, 0))
        self.assertTrue(result.final_evaluation.reached_source)
        self.assertEqual(len(result.frames), 2)
        initial_success = self.run_case([Action.RIGHT], source_xy=(0.5, 0))
        self.assertEqual(len(initial_success.frames), 1)
        self.assertTrue(initial_success.final_evaluation.reached_source)
        timeout = self.run_case([Action.LEFT] * 90)
        self.assertEqual(len(timeout.frames), 81)
        self.assertEqual(timeout.frames[-1].observation.attempt, 80)
        self.assertTrue(timeout.final_evaluation.episode_ended)
        self.assertFalse(timeout.final_evaluation.reached_source)
        final_success = self.run_case([Action.LEFT] * 79 + [Action.RIGHT] * 5, source_xy=(1, 0))
        self.assertEqual(len(final_success.frames), 81)
        self.assertTrue(final_success.final_evaluation.reached_source)

    def test_seeded_sequences_repeat_and_different_seeds_stay_valid(self):
        actions = [Action.RIGHT] * 4 + [Action.FORWARD] * 4
        first, second = self.run_case(actions), self.run_case(actions)
        self.assertEqual(first.entropies_nats, second.entropies_nats)
        for a, b in zip(first.frames, second.frames):
            self.assertEqual(a.observation, b.observation)
            np.testing.assert_array_equal(a.belief, b.belief)
        for seed in range(10):
            result = self.run_case(actions, seed=seed)
            self.assertTrue(np.isfinite(result.entropies_nats).all())
            self.assertTrue(all(np.isclose(frame.belief.sum(), 1) for frame in result.frames))


if __name__ == "__main__":
    unittest.main()
