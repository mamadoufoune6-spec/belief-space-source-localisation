"""One offline decision and sampling diagnostics; no environment is advanced."""

from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from belief_search import Action, EnvironmentConfig
from belief_search.information import information_action


def main():
    config = EnvironmentConfig()
    belief = np.zeros(config.shape)
    belief[12, 8] = belief[12, 16] = 0.5  # Sources (4,6) and (8,6).
    started = perf_counter()
    action, gains = information_action(belief, (6, 6), config,
                                       np.random.default_rng(17))
    elapsed_ms = (perf_counter() - started) * 1000
    record = {
        "config": asdict(config),
        "robot_xy": [6, 6],
        "source_hypotheses": [[4, 6, 0.5], [8, 6, 0.5]],
        "planning_seed": 17,
        "samples_per_action": 8,
        "selected_action": action.name,
        "gains_nats": {a.name: score for a, score in gains.items()},
        "elapsed_ms": elapsed_ms,
        "diagnostics": [],
    }
    print(f"Selected {action.name}; gains (nats): {record['gains_nats']}")
    print(f"One decision: {elapsed_ms:.3f} ms on this host")

    # Larger sample budget is diagnostic only; the policy default remains 8.
    for count in [8, 256]:
        scores, timings, selected = [], [], []
        for seed in range(20):
            started = perf_counter()
            chosen, estimates = information_action(belief, (6, 6), config,
                                                    np.random.default_rng(seed), count)
            timings.append((perf_counter() - started) * 1000)
            scores.append(estimates[Action.RIGHT])
            selected.append(chosen.name)
        summary = {
            "samples_per_action": count,
            "planning_seeds": list(range(20)),
            "right_gain_mean_nats": float(np.mean(scores)),
            "right_gain_std_nats": float(np.std(scores)),
            "mean_decision_ms": float(np.mean(timings)),
            "right_gains_nats": scores,
            "selected_actions": selected,
        }
        record["diagnostics"].append(summary)
        print(f"{count} samples/action: RIGHT mean {np.mean(scores):.6f} nats, "
              f"std {np.std(scores):.6f}; mean decision {np.mean(timings):.3f} ms")

    output = Path(__file__).resolve().parents[1] / "outputs" / "information-seed-17.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
