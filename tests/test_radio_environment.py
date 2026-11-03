"""Environment setup checks: movement, published lookup and information boundary."""

from dataclasses import fields
from pathlib import Path
import unittest

import numpy as np

from belief_search import Action, Observation
from belief_search.radio_environment import (
    RadioMapBank, RadioMapEnvironment, RadioEnvironmentConfig, decode_rssi, load_radio_map_bank,
)


class RadioEnvironmentTests(unittest.TestCase):
    def setUp(self):
        buildings = np.zeros((3, 4), dtype=bool)
        buildings[1, 2] = True  # JSON position (2,1).
        pixels = np.arange(36, dtype=np.uint8).reshape(3, 3, 4)
        self.bank = RadioMapBank(buildings, np.array([[3, 2], [1, 0], [2, 1]]), pixels, "IRT2", 1)
        self.config = RadioEnvironmentConfig(noise_std_db=0)

    def test_original_release_encoding_and_coordinate_lookup(self):
        np.testing.assert_allclose(decode_rssi([0, 255]), [-124, -24.84])
        self.assertEqual(self.bank.array_index((1, 0)), (2, 1))
        np.testing.assert_allclose(self.bank.predicted_rssi((1, 0)),
                                   decode_rssi(self.bank.gain_pixels[:, 2, 1]))
        self.assertFalse(self.bank.gain_pixels.flags.writeable)

    def test_building_and_outer_boundary_rejections_count_and_resample(self):
        env = RadioMapEnvironment(self.bank, RadioEnvironmentConfig(noise_std_db=2))
        initial, _ = env.reset(robot_xy=(1, 1), source_id=0, seed=17)
        blocked, _ = env.step(Action.RIGHT)
        self.assertEqual(blocked.robot_xy, (1, 1))
        self.assertTrue(blocked.boundary_blocked)
        self.assertEqual(blocked.attempt, 1)
        self.assertNotEqual(initial.rssi_dbm, blocked.rssi_dbm)
        moved, _ = env.step(Action.FORWARD)
        self.assertEqual(moved.robot_xy, (1, 2))
        self.assertFalse(moved.boundary_blocked)
        self.assertEqual(moved.attempt, 2)
        env.reset(robot_xy=(0, 0), source_id=0, seed=17)
        edge, _ = env.step(Action.LEFT)
        self.assertTrue(edge.boundary_blocked)
        self.assertEqual(edge.robot_xy, (0, 0))
        self.assertEqual(edge.attempt, 1)

    def test_zero_noise_reading_selects_only_active_candidate(self):
        env = RadioMapEnvironment(self.bank, self.config)
        for candidate in range(3):
            observation, _ = env.reset(robot_xy=(0, 0), source_id=candidate, seed=17)
            self.assertEqual(observation.rssi_dbm, self.bank.predicted_rssi((0, 0))[candidate])
        self.assertEqual({f.name for f in fields(Observation)},
                         {"robot_xy", "rssi_dbm", "attempt", "boundary_blocked"})
        # Do not silently drop/relocate a transmitter overlapping a building.
        self.assertEqual(env.ground_truth().source_xy, (2, 1))

    def test_first_reach_initial_final_attempt_and_timeout(self):
        env = RadioMapEnvironment(self.bank, RadioEnvironmentConfig(noise_std_db=0, max_steps=1))
        _, initial = env.reset(robot_xy=(3, 2), source_id=0, seed=17)
        self.assertTrue(initial.reached_source and initial.episode_ended)
        with self.assertRaises(RuntimeError):
            env.step(Action.LEFT)
        env.reset(robot_xy=(3, 1), source_id=0, seed=17)
        observation, final = env.step(Action.FORWARD)
        self.assertEqual(observation.attempt, 1)
        self.assertTrue(final.reached_source and final.episode_ended)
        env.reset(robot_xy=(0, 0), source_id=0, seed=17)
        _, timeout = env.step(Action.LEFT)
        self.assertTrue(timeout.episode_ended)
        self.assertFalse(timeout.reached_source)
        with self.assertRaises(RuntimeError):
            env.wait()

    def test_seeded_reproducibility_and_stationary_wait(self):
        traces = []
        for _ in range(2):
            env = RadioMapEnvironment(self.bank)
            initial, _ = env.reset(robot_xy=(0, 0), source_id=0, seed=17)
            waited, _ = env.wait()
            self.assertEqual(waited.robot_xy, initial.robot_xy)
            self.assertEqual(waited.attempt, initial.attempt)
            self.assertNotEqual(waited.rssi_dbm, initial.rssi_dbm)
            traces.append([initial, waited, env.step(Action.RIGHT)[0]])
        self.assertEqual(traces[0], traces[1])

    def test_invalid_inputs_do_not_consume_movement(self):
        env = RadioMapEnvironment(self.bank)
        with self.assertRaises(RuntimeError):
            env.wait()
        for position in [(2, 1), (0.5, 0), (-1, 0), (4, 0)]:
            with self.assertRaises(ValueError):
                env.reset(robot_xy=position, source_id=0, seed=17)
        with self.assertRaises(ValueError):
            env.reset(robot_xy=(0, 0), source_id=3, seed=17)
        env.reset(robot_xy=(0, 0), source_id=0, seed=17)
        with self.assertRaises(ValueError):
            env.step("right")
        self.assertEqual(env.step(Action.RIGHT)[0].attempt, 1)


DATA = Path(__file__).resolve().parents[1] / "data" / "radiomapseer-subset"


@unittest.skipUnless((DATA / "manifest.json").is_file(), "Optional RadioMapSeer subset not downloaded")
class PublishedEnvironmentTests(unittest.TestCase):
    def test_irt2_environment_and_secondary_dpm_bank(self):
        primary = load_radio_map_bank(DATA, 1)
        secondary = load_radio_map_bank(DATA, 1, simulation="DPM")
        np.testing.assert_array_equal(primary.transmitter_xy, secondary.transmitter_xy)
        self.assertEqual(primary.simulation, "IRT2")
        self.assertEqual(secondary.simulation, "DPM")
        env = RadioMapEnvironment(primary, RadioEnvironmentConfig(noise_std_db=0))
        observation, _ = env.reset(robot_xy=(120, 147), source_id=0, seed=17)
        self.assertEqual(observation.rssi_dbm, primary.predicted_rssi((120, 147))[0])
        self.assertNotEqual(observation.rssi_dbm, secondary.predicted_rssi((120, 147))[0])
        for _ in range(4):
            observation, evaluation = env.step(Action.RIGHT)
        self.assertEqual(observation.robot_xy, (124, 147))
        self.assertTrue(evaluation.reached_source and evaluation.episode_ended)


if __name__ == "__main__":
    unittest.main()
