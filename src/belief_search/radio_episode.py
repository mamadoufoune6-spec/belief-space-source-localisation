"""Shared RadioMapSeer episode: true IRT2 sensing, configurable model bank."""

from dataclasses import asdict
from time import perf_counter
import traceback

import numpy as np

from .estimator import update_belief, belief_entropy
from .radio_control import candidate_likelihood, select_candidate, map_information_action, GridRouter
from .radio_environment import RadioMapEnvironment, RadioEnvironmentConfig
from .radio_waypoint import waypoint_information_action
from .dcee import dcee_action
from .dcee_bfs import BFSTaskCost
from .greedy import greedy_action, SourceEstimate
from .types import EnvironmentConfig


def run_radio_episode(truth_bank, model_bank, *, robot_xy, source_id, seed,
                      controller_seed, policy, config=None, router=None, n_samples=8,
                      information_navigation="one_step", greedy_navigation="bfs",
                      reserve_last_unit=False, dcee_navigation="euclidean",
                      dcee_task_cost=None):
    """All returned truth fields are evaluation logs, never policy arguments.

    Numerical failures keep raw readings and traceback; no posterior is invented.
    The old confirmation costs/declaration rule and first-reach precedence apply.
    """
    if policy not in {"Greedy", "Information", "DCEE"}:
        raise ValueError("Choose Greedy, Information or DCEE")
    if information_navigation not in {"one_step", "waypoint_bfs"}:
        raise ValueError("Choose one_step or waypoint_bfs information navigation")
    if greedy_navigation not in {"bfs", "euclidean"}:
        raise ValueError("Choose bfs or euclidean greedy navigation")
    if not isinstance(reserve_last_unit, bool):
        raise ValueError("reserve_last_unit must be boolean")
    if dcee_navigation not in {"euclidean", "bfs"}:
        raise ValueError("Choose euclidean or bfs DCEE task distance")
    if dcee_navigation == "euclidean" and dcee_task_cost is not None:
        raise ValueError("A custom BFS cost requires dcee_navigation='bfs'")
    if (not np.array_equal(truth_bank.transmitter_xy, model_bank.transmitter_xy)
            or not np.array_equal(truth_bank.buildings, model_bank.buildings)):
        raise ValueError("Truth/model banks must share candidate ordering and geometry")
    config = config if config is not None else RadioEnvironmentConfig()
    needs_router = ((policy == "Greedy" and greedy_navigation == "bfs")
                    or (policy == "Information" and information_navigation == "waypoint_bfs")
                    or (policy == "DCEE" and dcee_navigation == "bfs"))
    if needs_router and router is None:
        router = GridRouter(model_bank)
    movement_budget_minimum = 2 if policy == "DCEE" or reserve_last_unit else 1
    if policy == "DCEE" and dcee_navigation == "bfs":
        dcee_task_cost = dcee_task_cost if dcee_task_cost is not None else BFSTaskCost(model_bank, router)
    if policy == "Greedy" and greedy_navigation == "euclidean":
        height, width = model_bank.buildings.shape
        motion_config = EnvironmentConfig(width_m=width - 1, height_m=height - 1, spacing_m=1)
    env = RadioMapEnvironment(truth_bank, config, runner_managed=True)
    rng = np.random.default_rng(controller_seed)
    count = len(model_bank.transmitter_xy)
    belief = np.full((count, 1), 1 / count)
    initial_entropy = float(belief_entropy(belief))
    observation, evaluation = env.reset(robot_xy=robot_xy, source_id=source_id, seed=seed)
    source_xy = tuple(map(int, truth_bank.transmitter_xy[source_id]))
    samples, decision_ms = [], []
    past_action = confirming = declaration = selected = None
    used_budget = failed_confirmations = 0
    command = "initial"
    error = error_traceback = None
    started = perf_counter()

    while True:
        try:
            posterior = update_belief(belief, candidate_likelihood(observation, model_bank,
                                                                config.noise_std_db))
        except (ValueError, FloatingPointError) as exc:
            error, error_traceback = str(exc), traceback.format_exc()
            samples.append({"command": command, **asdict(observation),
                            "reached_source": evaluation.reached_source,
                            "posterior": None, "entropy_nats": None,
                            "map_candidate_id": None, "map_xy": None,
                            "map_probability": None, "used_budget": used_budget})
            stop_reason = "estimator_error"
            break
        belief = posterior
        selected = select_candidate(belief, model_bank, rng)
        target_xy = tuple(map(int, model_bank.transmitter_xy[selected]))
        finished = None
        if evaluation.reached_source:
            finished = "reached"
        elif confirming is not None:
            if target_xy == confirming:
                declaration = {"candidate_id": selected, "xy": target_xy,
                               "probability": float(belief[selected, 0])}
                finished = "declared"
            else:
                used_budget += 1
                failed_confirmations += 1
                confirming = None
        sample = {"command": command, **asdict(observation),
                  "reached_source": evaluation.reached_source,
                  "posterior": belief.ravel().tolist(), "entropy_nats": float(belief_entropy(belief)),
                  "map_candidate_id": selected, "map_xy": target_xy,
                  "map_probability": float(belief[selected, 0]), "used_budget": used_budget}
        samples.append(sample)
        if finished:
            stop_reason = finished
            break

        if observation.robot_xy == target_xy:
            if used_budget > config.max_steps:
                stop_reason = "budget_exhausted"
                break
            confirming = target_xy
            observation, evaluation = env.wait()
            command = "WAIT"
            continue
        if used_budget >= config.max_steps:
            stop_reason = "budget_exhausted"
            break
        if config.max_steps - used_budget < movement_budget_minimum:
            stop_reason = "insufficient_horizon_budget" if policy == "DCEE" else "shared_reserve_stop"
            break

        began = perf_counter()
        try:
            if policy == "Greedy":
                sample["navigation_target_xy"] = target_xy
                if greedy_navigation == "bfs":
                    action = router.action(observation.robot_xy, target_xy, past_action, rng)
                else:
                    action = greedy_action(observation.robot_xy,
                                           SourceEstimate(target_xy, float(belief[selected, 0])),
                                           motion_config, past_action, rng)
                if action is None:
                    stop_reason = "unreachable_map"
                    break
            elif policy == "DCEE":
                action, costs, details = dcee_action(belief, observation.robot_xy, model_bank,
                                                    config.noise_std_db, rng, past_action, n_samples,
                                                    task_cost=dcee_task_cost)
                sample["estimated_task_costs_m2"] = {a.name: c for a, c in costs.items()}
                sample["hypothetical_plans"] = {a.name: d for a, d in details.items()}
            elif information_navigation == "waypoint_bfs":
                action, details = waypoint_information_action(
                    belief, observation.robot_xy, model_bank, config.noise_std_db,
                    rng, router, past_action, n_samples)
                sample["waypoint_information"] = details
                if action is None:
                    stop_reason = "no_reachable_waypoint"
                    break
                sample["navigation_target_xy"] = details["selected"]["xy"]
            else:
                action, gains = map_information_action(belief, observation.robot_xy, model_bank,
                                                       config.noise_std_db, rng, n_samples)
                sample["estimated_gains_nats"] = {a.name: score for a, score in gains.items()}
        except (ValueError, FloatingPointError) as exc:
            error, error_traceback = str(exc), traceback.format_exc()
            stop_reason = "planning_error"
            break
        decision_ms.append((perf_counter() - began) * 1000)
        observation, evaluation = env.step(action)
        used_budget += 1
        if not observation.boundary_blocked:
            past_action = action
        command = action.name

    return {
        "policy": policy, "map_id": truth_bank.map_id,
        "truth_simulation": truth_bank.simulation, "model_simulation": model_bank.simulation,
        "robot_start": tuple(robot_xy), "true_source_id": source_id, "true_source_xy": source_xy,
        "sensor_seed": seed, "controller_seed": controller_seed, "config": asdict(config),
        "samples_per_action": n_samples if policy in {"Information", "DCEE"} else 0,
        "navigation_mode": (("map_bfs" if greedy_navigation == "bfs" else "euclidean")
                            if policy == "Greedy" else
                            (("two_stage_bfs_cost" if dcee_navigation == "bfs" else "two_stage_no_bfs")
                             if policy == "DCEE" else information_navigation)),
        "dcee_distance_metric": dcee_navigation if policy == "DCEE" else None,
        "unreachable_distance_m": (dcee_task_cost.unreachable_distance_m
                                   if policy == "DCEE" and dcee_task_cost is not None else None),
        "movement_budget_minimum": movement_budget_minimum,
        "initial_entropy_nats": initial_entropy, "samples": samples,
        "stop_reason": stop_reason, "declaration": declaration,
        "used_budget": used_budget, "failed_confirmations": failed_confirmations,
        "last_valid_posterior": belief.ravel().tolist(),
        "episode_ms": (perf_counter() - started) * 1000,
        "decision_ms": decision_ms, "error": error, "traceback": error_traceback,
    }
