"""Explicit BFS-distance task-cost variant; no source hypotheses are removed."""

import numpy as np

from .radio_control import GridRouter, validate_belief


class BFSTaskCost:
    """Squared legal route lengths, with a disclosed finite unreachable penalty.

    A simple path uses at most free_cell_count - 1 edges, so free_cell_count
    + 1 metres is larger than any reachable shortest route on this 1 m grid.
    This fixed penalty is part of this cost variant, not a recovered route.
    """

    def __init__(self, bank, router=None):
        self.bank = bank
        self.router = router if router is not None else GridRouter(bank)
        self.unreachable_distance_m = int(np.count_nonzero(~bank.buildings)) + 1
        fields = []
        for xy in bank.transmitter_xy:
            distance = self.router.distances_to(tuple(map(int, xy))).astype(float)
            distance[distance < 0] = self.unreachable_distance_m
            fields.append(distance ** 2)
        self.squared_distances = np.stack(fields)
        self.squared_distances.setflags(write=False)

    def __call__(self, robot_xy, candidate_xy, belief):
        probabilities = validate_belief(belief, self.bank)
        if not np.array_equal(candidate_xy, self.bank.transmitter_xy):
            raise ValueError("BFS cost must use the same candidate coordinates/order")
        row, column = self.bank.array_index(robot_xy)
        return float(np.dot(probabilities.ravel(), self.squared_distances[:, row, column]))

    def unreachable_ids(self, robot_xy):
        row, column = self.bank.array_index(robot_xy)
        penalty = float(self.unreachable_distance_m) ** 2
        return np.flatnonzero(self.squared_distances[:, row, column] == penalty).tolist()
