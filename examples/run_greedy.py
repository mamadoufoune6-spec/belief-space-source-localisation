"""Run one reviewable greedy episode and save evaluation-only JSON records."""

from dataclasses import asdict
import json
from pathlib import Path

from belief_search.greedy_episode import run_greedy


def main():
    result = run_greedy(robot_xy=(0, 0), source_xy=(10, 8), seed=17, controller_seed=3)
    record = {
        "config": asdict(result.config),
        "sensor_seed": result.sensor_seed,
        "controller_seed": result.controller_seed,
        "true_source_xy": result.source_xy,
        "declaration": asdict(result.declaration) if result.declaration else None,
        "stop_reason": result.stop_reason,
        "used_budget": result.used_budget,
        "remaining_budget": result.remaining_budget,
        "movement_attempts": result.frames[-1].observation.attempt,
        "failed_confirmations": result.failed_confirmations,
        "samples": [],
    }
    for index, (frame, entropy, evaluation) in enumerate(
            zip(result.frames, result.entropies_nats, result.evaluations)):
        record["samples"].append({
            "sample_index": index,
            "command": "initial" if index == 0 else result.commands[index - 1].name,
            **asdict(frame.observation),
            "entropy_nats": entropy,
            **asdict(evaluation),
        })

    output = Path(__file__).resolve().parents[1] / "outputs" / "greedy-seed-17.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"Stop: {result.stop_reason}; declaration: {result.declaration}")
    print(f"Budget used: {result.used_budget}; movements: {record['movement_attempts']}; "
          f"failed confirmations: {result.failed_confirmations}")
    print(f"Final reaching success: {result.evaluations[-1].reached_source}")
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
