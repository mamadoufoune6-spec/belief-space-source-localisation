"""Codex-authored waypoint adapter for the requested shared-BFS benchmark.

Measurement waypoints are published candidate coordinates, not a restriction
of the source belief. Score endpoint information only, without travel reward,
intermediate hypothetical observations, or a future contingent action tree.
"""

from .radio_control import map_information_gain, validate_belief


def waypoint_information_action(belief, robot_xy, model_bank, noise_std_db,
                                rng, router, past_action=None, n_samples=8):
    """Choose a reachable distinct waypoint, then one shared BFS step toward it.

    Replan after every actual reading. Exclude distance-zero destinations:
    WAIT remains the runner's MAP-confirmation command, as in the baseline.
    All source hypotheses remain in every information-gain calculation.
    """
    validate_belief(belief, model_bank)
    index = model_bank.array_index(robot_xy)
    options = []
    for candidate_id, xy in enumerate(model_bank.transmitter_xy):
        destination = tuple(map(int, xy))
        distance = int(router.distances_to(destination)[index])
        if distance <= 0:
            continue  # Blocked, disconnected, or already at this waypoint.
        gain = map_information_gain(belief, destination, model_bank,
                                    noise_std_db, rng, n_samples)
        options.append({"candidate_id": candidate_id, "xy": destination,
                        "distance_steps": distance, "gain_nats": gain})
    if not options:
        return None, {"selected": None, "options": []}
    best = max(item["gain_nats"] for item in options)
    tied = [item for item in options if item["gain_nats"] == best]
    selected = tied[int(rng.integers(len(tied)))]
    action = router.action(robot_xy, selected["xy"], past_action, rng)
    return action, {"selected": selected, "options": options}
