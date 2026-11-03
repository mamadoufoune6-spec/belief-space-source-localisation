from dataclasses import fields, replace
import subprocess
import sys
import unittest

import numpy as np

from belief_search import Action, EnvironmentConfig, SearchEnvironment, Observation, valid_actions
from belief_search.environment import mean_rssi


class EnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.config = EnvironmentConfig()

    def test_import_without_matplotlib(self):
        subprocess.run([sys.executable, "-c",
                        "import belief_search, sys; assert 'matplotlib' not in sys.modules"],
                       check=True, capture_output=True)

    def test_grid_and_actions(self):
        self.assertEqual(self.config.shape, (25, 25))
        self.assertEqual(set(valid_actions((6, 6), self.config)), set(Action))
        for action, target in [(Action.FORWARD, (6, 6.5)), (Action.BACKWARD, (6, 5.5)),
                               (Action.LEFT, (5.5, 6)), (Action.RIGHT, (6.5, 6))]:
            env = SearchEnvironment()
            env.reset(robot_xy=(6, 6), source_xy=(12, 12), seed=1)
            obs, _ = env.step(action)
            self.assertEqual(obs.robot_xy, target)
            self.assertEqual(env.ground_truth().source_xy, (12, 12))

    def test_all_grid_boundaries(self):
        x_axis, y_axis = self.config.grid_axes()
        for x in x_axis:
            for y in y_axis:
                allowed = valid_actions((x, y), self.config)
                expected_count = 4 - int(x in (0, 12)) - int(y in (0, 12))
                self.assertEqual(len(allowed), expected_count)
                for action in Action:
                    env = SearchEnvironment()
                    source = (12, 12) if x + y < 12 else (0, 0)
                    env.reset(robot_xy=(x, y), source_xy=source, seed=2)
                    obs, _ = env.step(action)
                    self.assertEqual(obs.boundary_blocked, action not in allowed)
                    self.assertEqual(obs.attempt, 1)
                    dx, dy = action.value if action in allowed else (0, 0)
                    self.assertEqual(obs.robot_xy, (x + 0.5 * dx, y + 0.5 * dy))

    def test_invalid_inputs(self):
        env = SearchEnvironment()
        for position in [(-0.5, 1), (12.5, 0), (0.2, 1), (np.nan, 0), (1, 2, 3)]:
            with self.assertRaises(ValueError):
                env.reset(robot_xy=position, source_xy=(12, 12), seed=1)
            with self.assertRaises(ValueError):
                env.reset(robot_xy=(0, 0), source_xy=position, seed=1)
        for kwargs in [{"spacing_m": 0}, {"width_m": 12.1}, {"noise_std_db": -1},
                       {"max_steps": 1.5}, {"max_steps": True}]:
            with self.assertRaises(ValueError):
                EnvironmentConfig(**kwargs)
        with self.assertRaises(RuntimeError):
            env.step(Action.RIGHT)
        env.reset(robot_xy=(0, 0), source_xy=(12, 12), seed=1)
        with self.assertRaises(ValueError):
            env.step("right")
        self.assertEqual(env.step(Action.RIGHT)[0].attempt, 1)

    def test_sensor_reference_values_and_cutoff(self):
        for distance, expected in [(0, -33.9794000867), (0.5, -33.9794000867),
                                   (1, -40), (2, -46.0205999133),
                                   (4, -52.0411998266), (12 * np.sqrt(2), -64.5939248776)]:
            self.assertAlmostEqual(mean_rssi((0, 0), (distance, 0), self.config), expected, places=7)
        self.assertEqual(mean_rssi((0, 0), (3, 4), self.config),
                         mean_rssi((0, 0), (4, 3), self.config))
        values = [mean_rssi((0, 0), (d, 0), self.config) for d in (0.5, 1, 2, 4, 12)]
        self.assertTrue(np.all(np.diff(values) < 0))
        # Check all relative offsets that can occur between any two grid positions.
        self.assertTrue(all(np.isfinite(mean_rssi((0, 0), (x / 2, y / 2), self.config))
                            for x in range(-24, 25) for y in range(-24, 25)))

    def test_noise_statistics(self):
        env = SearchEnvironment(replace(self.config, max_steps=20000))
        obs, _ = env.reset(robot_xy=(0, 0), source_xy=(12, 12), seed=456)
        readings = [obs.rssi_dbm]
        for _ in range(19999):
            obs, _ = env.step(Action.LEFT)  # Blocked: repeated samples at one position.
            readings.append(obs.rssi_dbm)
        residuals = np.array(readings) - mean_rssi((0, 0), (12, 12), self.config)
        # Loose predeclared tolerances (>4 standard errors at this sample size).
        self.assertLess(abs(residuals.mean()), 0.07)
        self.assertLess(abs(residuals.std() - 2), 0.06)
        self.assertLess(abs(np.corrcoef(residuals[:-1], residuals[1:])[0, 1]), 0.04)

    def test_reproducibility_and_multiple_starts(self):
        for robot, source in [((0, 0), (12, 12)), ((12, 0), (0, 12)),
                              ((6, 6), (2, 2)), ((0.5, 0), (12, 0))]:
            traces = []
            for _ in range(2):
                env = SearchEnvironment()
                trace = [env.reset(robot_xy=robot, source_xy=source, seed=30)]
                trace += [env.step(a) for a in [Action.FORWARD, Action.RIGHT, Action.LEFT]]
                traces.append(trace)
            self.assertEqual(traces[0], traces[1])
        a, b = SearchEnvironment(), SearchEnvironment()
        self.assertNotEqual(a.reset(robot_xy=(0, 0), source_xy=(12, 12), seed=1)[0].rssi_dbm,
                            b.reset(robot_xy=(0, 0), source_xy=(12, 12), seed=2)[0].rssi_dbm)

    def test_termination_and_last_attempt(self):
        env = SearchEnvironment()
        env.reset(robot_xy=(0, 0), source_xy=(1, 0), seed=1)
        for _ in range(79):
            _, result = env.step(Action.LEFT)
            self.assertFalse(result.episode_ended)
        obs, result = env.step(Action.RIGHT)
        self.assertEqual(obs.attempt, 80)
        self.assertTrue(result.reached_source)
        self.assertTrue(result.episode_ended)
        with self.assertRaises(RuntimeError):
            env.step(Action.RIGHT)
        env.reset(robot_xy=(0, 0), source_xy=(12, 12), seed=1)
        for _ in range(80):
            _, result = env.step(Action.LEFT)
        self.assertTrue(result.episode_ended)
        self.assertFalse(result.reached_source)
        with self.assertRaises(RuntimeError):
            env.step(Action.RIGHT)
        obs, result = env.reset(robot_xy=(0, 0), source_xy=(0.5, 0), seed=1)
        self.assertEqual(obs.attempt, 0)
        self.assertTrue(result.reached_source)

    def test_observation_contract(self):
        self.assertEqual({f.name for f in fields(Observation)},
                         {"robot_xy", "rssi_dbm", "attempt", "boundary_blocked"})
        env = SearchEnvironment()
        obs, _ = env.reset(robot_xy=(0, 0), source_xy=(12, 12), seed=1)
        truth = env.ground_truth()
        env.reset(robot_xy=(6, 6), source_xy=(1, 1), seed=2)
        self.assertEqual(obs.robot_xy, (0, 0))
        self.assertEqual(truth.source_xy, (12, 12))


if __name__ == "__main__":
    unittest.main()
