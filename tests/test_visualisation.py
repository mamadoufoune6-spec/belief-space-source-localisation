import io
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.pyplot as plt
import numpy as np

from belief_search import EnvironmentConfig, GroundTruth, Observation
from belief_search.recording import Frame, trajectory
from belief_search.visualisation import plot_frame, animate_episode, draw_belief


class PlotTests(unittest.TestCase):
    def setUp(self):
        self.config = EnvironmentConfig()
        self.belief = np.zeros(self.config.shape)
        self.belief[3, 8] = 1  # x=4 m, y=1.5 m: asymmetric orientation fixture.
        self.frames = [Frame(Observation((1 + i * 0.5, 1), -50, i, False), self.belief)
                       for i in range(3)]

    def test_recording_copies(self):
        self.belief[:] = 0
        self.assertEqual(self.frames[0].belief[3, 8], 1)
        self.assertFalse(self.frames[0].belief.flags.writeable)
        np.testing.assert_equal(trajectory(self.frames), [[1, 1], [1.5, 1], [2, 1]])
        self.assertEqual(trajectory([]).shape, (0, 2))

    def test_headless_plot_positions_and_heatmap(self):
        before = self.frames[0].belief.copy()
        with patch.object(plt, "show", side_effect=AssertionError("Unexpected window")):
            fig, ax = plot_frame(self.frames[-1], trajectory(self.frames), self.config)
            FigureCanvasAgg(fig)
            output = io.BytesIO()
            fig.savefig(output, format="png")
        self.assertGreater(len(output.getvalue()), 1000)
        self.assertEqual(plt.get_fignums(), [])
        self.assertEqual(ax.get_xlim(), (0, 12))
        self.assertEqual(ax.get_ylim(), (0, 12))
        mesh = ax.collections[0]
        np.testing.assert_equal(np.asarray(mesh.get_array()).reshape(self.config.shape), before)
        vertices = mesh.get_coordinates()
        np.testing.assert_allclose(vertices[3:5, 8:10].mean(axis=(0, 1)), [4, 1.5])
        self.assertNotIn("Source (ground truth)", ax.get_legend_handles_labels()[1])
        np.testing.assert_equal(ax.collections[-1].get_offsets(), [[2, 1]])
        np.testing.assert_equal(ax.lines[0].get_xydata(), trajectory(self.frames))
        np.testing.assert_equal(before, self.frames[0].belief)

    def test_explicit_truth_and_no_belief(self):
        fig, ax = plot_frame(Frame(self.frames[0].observation), trajectory(self.frames[:1]),
                             self.config, truth=GroundTruth((8, 9)))
        self.assertIn("Source (ground truth)", ax.get_legend_handles_labels()[1])
        np.testing.assert_equal(ax.collections[0].get_offsets(), [[8, 9]])
        self.assertEqual(len(fig.axes), 1)  # No heatmap colourbar.

    def test_invalid_beliefs(self):
        from matplotlib.figure import Figure
        ax = Figure().subplots()
        for values in [np.zeros((2, 2)), np.zeros(self.config.shape),
                       np.full(self.config.shape, np.nan), -self.belief]:
            with self.assertRaises(ValueError):
                draw_belief(ax, values, self.config)

    def test_animation_replay(self):
        fig, ax = plt.subplots()
        animation = animate_episode(self.frames, self.config, ax=ax, belief_vmax=0.5)
        # Exercise the callback without an event loop, including replay out of order.
        animation._func(2)
        expected = ax.lines[0].get_xydata().copy()
        animation._func(0)
        self.assertEqual(len(ax.lines[0].get_xdata()), 1)
        animation._func(2)
        np.testing.assert_equal(ax.lines[0].get_xydata(), expected)
        self.assertEqual(ax.collections[-1].get_clim(), (0, 0.5))
        FigureCanvasAgg(fig).draw()
        plt.close(fig)

    def test_animation_validation(self):
        with self.assertRaises(ValueError):
            animate_episode([], self.config)
        with self.assertRaises(ValueError):
            animate_episode([self.frames[0], Frame(self.frames[1].observation)], self.config)


if __name__ == "__main__":
    unittest.main()
