"""Run prescribed movements through sensing and estimation, without a controller."""

from collections.abc import Iterable
from dataclasses import dataclass

from .environment import SearchEnvironment
from .estimator import initialise_belief, compute_likelihood, update_belief, belief_entropy
from .recording import Frame
from .types import Action, Evaluation, Position


@dataclass(frozen=True)
class SequenceResult:
    """Runner-owned record. Each frame contains the posterior after its observation."""
    frames: tuple[Frame, ...]
    evaluations: tuple[Evaluation, ...]
    entropies_nats: tuple[float, ...]
    initial_entropy_nats: float

    @property
    def final_evaluation(self) -> Evaluation:
        return self.evaluations[-1]


def run_sequence(
    env: SearchEnvironment,
    actions: Iterable[Action],
    *,
    robot_xy: Position,
    source_xy: Position,
    seed: int,
) -> SequenceResult:
    """Initialise -> sense -> likelihood -> posterior -> entropy -> move -> repeat.

    Explicit starts and the seed belong to this runner, not estimator inputs.
    Record the initial observation at attempt zero. Stop on success, timeout,
    or exhaustion of the supplied actions. Action exhaustion alone is not a
    terminal environment outcome. A blocked command still yields a fresh
    observation and belief update. Estimator errors propagate without silently
    resetting belief. No plotting is imported and no actions are selected here.
    """
    config = env.config
    belief = initialise_belief(config)
    initial_entropy = float(belief_entropy(belief))
    observation, evaluation = env.reset(robot_xy=robot_xy, source_xy=source_xy, seed=seed)
    frames, evaluations, entropies = [], [], []

    def assimilate(observation, evaluation):
        nonlocal belief
        likelihood = compute_likelihood(observation, config)
        belief = update_belief(belief, likelihood)
        frames.append(Frame(observation, belief))
        evaluations.append(evaluation)
        entropies.append(float(belief_entropy(belief)))

    assimilate(observation, evaluation)
    if not evaluation.episode_ended:
        for action in actions:
            observation, evaluation = env.step(action)
            assimilate(observation, evaluation)
            if evaluation.episode_ended:
                break

    return SequenceResult(tuple(frames), tuple(evaluations), tuple(entropies), initial_entropy)
