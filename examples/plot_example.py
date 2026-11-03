"""Sensing and Bayesian estimation along prescribed movements; no controller."""

import argparse
from pathlib import Path

import matplotlib

from belief_search import Action, EnvironmentConfig, SearchEnvironment
from belief_search.recording import trajectory
from belief_search.sequence import run_sequence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show", action="store_true", help="Explicitly open a plot window")
    parser.add_argument("--animate", action="store_true", help="Save a trajectory GIF")
    parser.add_argument("--output", type=Path, default=Path("outputs/example.png"))
    parser.add_argument("--seed", type=int, default=17, help="Seed for reproducible RSSI noise")
    args = parser.parse_args()
    if not args.show:
        matplotlib.use("Agg")
    from belief_search.visualisation import animate_episode, plot_frame

    config = EnvironmentConfig()
    env = SearchEnvironment(config)
    # A fixed L-shaped path, chosen for demonstration, not by a search controller.
    actions = [Action.RIGHT] * 16 + [Action.FORWARD] * 14
    result = run_sequence(env, actions, robot_xy=(1, 1), source_xy=(9, 8), seed=args.seed)
    frames = result.frames

    ax = None
    if args.show:
        import matplotlib.pyplot as plt
        _, ax = plt.subplots(figsize=(7, 6), layout="constrained")
    fig, ax = plot_frame(frames[-1], trajectory(frames), config,
                         truth=env.ground_truth(), ax=ax)
    ax.set_title(f"Posterior after {frames[-1].observation.attempt} attempts "
                 f"| entropy {result.entropies_nats[-1]:.3f} nats")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150)
    print(f"Saved {args.output}")
    print(f"Entropy: prior {result.initial_entropy_nats:.3f}, "
          f"final posterior {result.entropies_nats[-1]:.3f} nats")
    print(f"Reached source: {result.final_evaluation.reached_source}; "
          f"episode ended: {result.final_evaluation.episode_ended}")
    animation = None
    if args.animate:
        animation = animate_episode(frames, config, truth=env.ground_truth())
        animation.save(args.output.with_suffix(".gif"), writer="pillow", fps=6)
        print(f"Saved {args.output.with_suffix('.gif')}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
