"""Common utils for optimization algorithms."""

from typing import Any, Callable

import numpy as np


class BestEver:
    """Bounded record of the best-fitness unique sequences seen across a run.

    Deduplicates by sequence string (fitness is deterministic in the sequence,
    so a recurring sequence has the same score). Intended to be updated once per
    optimization step from the selected beam -- NOT inside the oracle inner loop.
    Reuses the caller's sort_key so ordering matches get_samples exactly.

    Memory is bounded by the largest `n` ever passed to `best()`, which is a
    fixed config value (proposals_per_round), i.e. on the order of a dozen nodes.
    """

    def __init__(self, sort_key: Callable[[Any], Any], capacity: int):
        assert capacity >= 1
        self._sort_key = sort_key          # node -> orderable key; the designer's own
        self._capacity = capacity          # grows lazily to the largest n requested
        self._by_seq: dict[str, Any] = {}  # sequence string -> node

    def update(self, nodes: list) -> None:
        """Merge a beam's nodes in. Called once per step."""
        for n in nodes:
            # Deterministic fitness => overwriting a recurring seq is value-identical
            # for ordering purposes (see note in get_samples about PBT fields).
            self._by_seq[n.seq] = n
        if len(self._by_seq) > self._capacity:
            kept = sorted(self._by_seq.values(), key=self._sort_key, reverse=True)
            self._by_seq = {n.seq: n for n in kept[: self._capacity]}

    def best(self, n: int) -> list:
        """Return up to `n` nodes, highest fitness first."""
        self._capacity = max(self._capacity, n)  # never prune below what's been requested
        nodes = sorted(self._by_seq.values(), key=self._sort_key, reverse=True)
        return nodes[: min(n, len(nodes))]


def get_locations_to_edit(
    positions_to_mutate: list[int],
    random_n_loc: int,
    rng: np.random.Generator,
    method: str,
) -> np.ndarray:
    """Selects locations to edit."""
    assert random_n_loc > 0
    assert random_n_loc <= len(positions_to_mutate)

    if method == "all":
        return np.array(positions_to_mutate)
    elif method == "random":
        return rng.choice(positions_to_mutate, size=random_n_loc, replace=False)
    else:
        raise ValueError("Arg not recognized.")


def generate_single_mutant_multiedits(
    base_str: str,
    locs_to_edit: list[int] | np.ndarray,
    alphabet: list[str],
    rng: np.random.Generator,
) -> str:
    """Return a mutant."""
    assert isinstance(alphabet, list)
    assert len(alphabet) > 1
    mutant = list(base_str)

    for i in locs_to_edit:
        # TODO(joelshor): This should be `rng.choice(set(alphabet) - mutant[i])`,
        # but we want to keep it for consistency with the publication.
        # Expect this behavior to change in the future.
        mutant[i] = rng.choice(alphabet)
    return "".join(mutant)
