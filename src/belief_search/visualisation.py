"""Matplotlib views of recorded data. No show(), simulation steps, or RNG draws."""

from collections.abc import Sequence

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation

from .recording import Frame, trajectory
from .types import EnvironmentConfig, GroundTruth


def draw_arena(ax: Axes, config: EnvironmentConfig) -> None:
    ax.set(xlim=(0, config.width_m), ylim=(0, config.height_m),
           xlabel="x (m)", ylabel="y (m)")
    ax.set_aspect("equal", adjustable="box")
    x, y = config.grid_axes()
    ax.set_xticks(x, minor=True)
    ax.set_yticks(y, minor=True)
    ax.grid(which="both", alpha=0.15)


def draw_robot(ax: Axes, robot_xy):
    return ax.scatter(*robot_xy, c="tab:red", s=65, label="Robot", zorder=4)


def draw_source(ax: Axes, source_xy):
    return ax.scatter(*source_xy, marker="*", c="cyan", edgecolors="black",
                      s=150, label="Source (ground truth)", zorder=5)


def draw_trajectory(ax: Axes, positions_xy):
    positions = np.asarray(positions_xy, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 2 or not np.isfinite(positions).all():
        raise ValueError("Trajectory must be a finite (N, 2) array")
    return ax.plot(positions[:, 0], positions[:, 1], color="tab:red",
                   linewidth=1.5, label="Trajectory", zorder=3)[0]


def _validate_belief(belief, config):
    values = np.asarray(belief, dtype=float)
    if values.shape != config.shape:
        raise ValueError(f"Belief must have shape {config.shape} in [y, x] order")
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("Belief must be finite and nonnegative")
    if not np.isclose(values.sum(), 1.0, rtol=0, atol=1e-8):
        raise ValueError("Belief must sum to one; plotting does not normalise it")
    return values


def _validate_vmax(vmax):
    if not np.isfinite(vmax) or vmax <= 0:
        raise ValueError("vmax must be finite and positive")


def draw_belief(ax: Axes, belief, config: EnvironmentConfig, *, vmax: float = 1.0):
    values = _validate_belief(belief, config)
    _validate_vmax(vmax)
    x, y = config.grid_axes()
    # Grid coordinates are cell centres. Outer half-cells are clipped by arena axes.
    return ax.pcolormesh(x, y, values, shading="nearest", cmap="viridis",
                         vmin=0, vmax=vmax, zorder=0)


def _canvas(config, truth, *, ax=None):
    if ax is None:
        fig = Figure(figsize=(7, 6), layout="constrained")
        ax = fig.subplots()
    else:
        fig = ax.figure
    draw_arena(ax, config)
    if truth is not None:
        draw_source(ax, truth.source_xy)
    return fig, ax


def plot_frame(frame: Frame, trajectory_xy, config: EnvironmentConfig, *,
               truth: GroundTruth | None = None, belief_vmax: float = 1.0,
               ax: Axes | None = None):
    """Return (figure, axes). Pass GUI-managed axes explicitly for interactive use."""
    fig, ax = _canvas(config, truth, ax=ax)
    if frame.belief is not None:
        mesh = draw_belief(ax, frame.belief, config, vmax=belief_vmax)
        fig.colorbar(mesh, ax=ax, label="Probability per source-grid position")
    draw_trajectory(ax, trajectory_xy)
    draw_robot(ax, frame.observation.robot_xy)
    ax.set_title(f"Attempt {frame.observation.attempt} | RSSI {frame.observation.rssi_dbm:.2f} dBm")
    ax.legend(loc="upper right")
    return fig, ax


def animate_episode(frames: Sequence[Frame], config: EnvironmentConfig, *,
                    truth: GroundTruth | None = None, interval_ms: int = 150,
                    belief_vmax: float = 1.0, ax: Axes | None = None) -> FuncAnimation:
    """Replay snapshots. Retain the returned animation; its figure is caller-owned.

    Beliefs must be present for all frames or none. Colour limits stay fixed.
    Pass pyplot-created axes for interactive playback; default figures have no window.
    """
    frames = tuple(frames)
    if not frames:
        raise ValueError("Animation needs at least one frame")
    if interval_ms <= 0:
        raise ValueError("interval_ms must be positive")
    present = [f.belief is not None for f in frames]
    if any(present) and not all(present):
        raise ValueError("Supply beliefs for every frame or none")
    for frame in frames:
        if frame.belief is not None:
            _validate_belief(frame.belief, config)
    fig, ax = _canvas(config, truth, ax=ax)
    positions = trajectory(frames)
    line = draw_trajectory(ax, positions[:1])
    robot = draw_robot(ax, positions[0])
    mesh = None
    if all(present):
        mesh = draw_belief(ax, frames[0].belief, config, vmax=belief_vmax)
        fig.colorbar(mesh, ax=ax, label="Probability per source-grid position")
    ax.legend(loc="upper right")

    def update(index):
        frame = frames[index]
        line.set_data(positions[:index + 1, 0], positions[:index + 1, 1])
        robot.set_offsets(positions[index:index + 1])
        ax.set_title(f"Attempt {frame.observation.attempt} | RSSI {frame.observation.rssi_dbm:.2f} dBm")
        if mesh is not None:
            mesh.set_array(frame.belief)
        return (line, robot, ax.title) + ((mesh,) if mesh is not None else ())

    return FuncAnimation(fig, update, frames=range(len(frames)),
                         init_func=lambda: update(0), interval=interval_ms,
                         blit=False, repeat=False, cache_frame_data=False)
