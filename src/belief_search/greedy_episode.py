"""Shared episode rules for the agreed policies, sensing and Bayesian estimation."""

from dataclasses import dataclass

import numpy as np

from .environment import SearchEnvironment
from .estimator import initialise_belief, compute_likelihood, update_belief, belief_entropy
from .greedy import Command, SourceEstimate, select_map, greedy_action
from .recording import Frame
from .types import Action, EnvironmentConfig, Evaluation, Position


@dataclass(frozen=True)
class GreedyResult:
    config: EnvironmentConfig
    source_xy: Position  # Evaluation-only; never passed to the policy.
    sensor_seed: int
    controller_seed: int
    frames: tuple[Frame, ...]  # Index identifies fresh samples, including WAIT.
    entropies_nats: tuple[float, ...]
    evaluations: tuple[Evaluation, ...]
    commands: tuple[Action | Command, ...]
    declaration: SourceEstimate | None
    used_budget: int
    failed_confirmations: int
    stop_reason: str
    final_estimate: SourceEstimate  # Actual final selected MAP, including its seeded tie choice.

    @property
    def remaining_budget(self):
        # A failed confirmation at the limit is recorded separately, not as -1 remaining.
        return max(0, self.config.max_steps - self.used_budget)


def run_greedy(*, robot_xy, source_xy, seed: int, controller_seed: int,
               config: EnvironmentConfig | None = None, stop_on_reach=False) -> GreedyResult:
    """Run the greedy baseline using the shared episode rules."""
    return run_episode(robot_xy=robot_xy, source_xy=source_xy, seed=seed,
                       controller_seed=controller_seed, config=config, stop_on_reach=stop_on_reach)


def run_episode(*, robot_xy, source_xy, seed: int, controller_seed: int,
                config: EnvironmentConfig | None = None, movement_policy=None,
                stop_on_reach=False) -> GreedyResult:
    """Run shared episode rules, optionally ending at first true reaching success.

    used_budget = movement attempts + failed confirmations. A failed confirmation
    at the limit records its charge (81), ends immediately, and leaves zero budget.
    A WAIT is only requested at the current MAP target; unchanged MAP ends the run.
    Optional movement_policy(belief, robot_xy, config, planning_rng) returns an
    Action. MAP confirmation, sensors and costs are identical for both policies.
    GreedyResult is retained as the existing record format for compatibility.
    stop_on_reach=True is evaluator termination, not a controller input. Check
    after sensing/updating, before declaration or budget, including attempt 80.
    False preserves earlier declaration/budget experiments for reproduction.
    """
    config = config if config is not None else EnvironmentConfig()
    env = SearchEnvironment(config, runner_managed=True)
    rng = np.random.default_rng(controller_seed)  # Separate from hidden sensor noise.
    belief = initialise_belief(config)
    observation, evaluation = env.reset(robot_xy=robot_xy, source_xy=source_xy, seed=seed)
    frames, entropies, evaluations, commands = [], [], [], []
    used_budget = failed_confirmations = 0
    past_action = None
    declaration = None
    confirming = None

    while True:
        likelihood = compute_likelihood(observation, config)
        belief = update_belief(belief, likelihood)
        frames.append(Frame(observation, belief))
        entropies.append(float(belief_entropy(belief)))
        evaluations.append(evaluation)
        target = select_map(belief, config, rng)

        if stop_on_reach and evaluation.reached_source:
            stop_reason = "reached"
            break

        if confirming is not None:
            if target.xy == confirming:
                declaration = target
                stop_reason = "declared"
                break
            used_budget += 1
            failed_confirmations += 1
            confirming = None  # Reset, rather than accumulating visits to old targets.

        if observation.robot_xy == target.xy:
            # One final at-target confirmation is allowed even at the budget limit.
            if used_budget > config.max_steps:
                stop_reason = "budget_exhausted"
                break
            confirming = target.xy
            command = Command.WAIT
            observation, evaluation = env.wait()
        else:
            if used_budget >= config.max_steps:
                stop_reason = "budget_exhausted"
                break
            if movement_policy is None:
                command = greedy_action(observation.robot_xy, target, config, past_action, rng)
            else:
                command = movement_policy(belief, observation.robot_xy, config, rng)
            observation, evaluation = env.step(command)
            used_budget += 1
            if not observation.boundary_blocked:
                past_action = command
        commands.append(command)

    # Termination is now decided by the runner; reaching remains a separate metric.
    evaluations[-1] = Evaluation(evaluations[-1].reached_source, True)
    return GreedyResult(config, env.ground_truth().source_xy, seed, controller_seed,
                        tuple(frames), tuple(entropies), tuple(evaluations),
                        tuple(commands), declaration, used_budget,
                        failed_confirmations, stop_reason, target)
