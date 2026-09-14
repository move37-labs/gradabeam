"""Shared helpers for AdaBeam and GradaBeam beam bookkeeping.

This module owns only lifecycle-adjacent logic that is identical across
the two proposal kernels:

- candidate accumulation with documented identity keys
- beam ranking with a dedicated tie RNG
- observationally pure sample retrieval
- batched fitness
- rollout acceptance filtering
- start-sequence position resolution

Proposal generation, mutation distributions, sampler streams, TISM/PBT
state, and debug reporting stay in the optimizer modules.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, MutableMapping, Sequence
from typing import Any

import numpy as np

from gradabeam import ada_utils


def resolve_positions_to_mutate(
    start_sequence: str,
    positions_to_mutate: list[int] | None,
) -> list[int]:
    """Return mutable positions, defaulting to every index."""
    positions = positions_to_mutate or list(range(len(start_sequence)))
    assert min(positions) >= 0
    assert max(positions) < len(start_sequence)
    return positions


def accumulate_first_observed(
    store: MutableMapping[Any, Any],
    key: Any,
    node: Any,
) -> None:
    """Keep the first node seen for ``key``.

    "First" means the existing deterministic rollout / chunk expansion
    order. Later collisions are ignored.
    """
    if key not in store:
        store[key] = node


def adabeam_candidate_key(node: Any) -> tuple[str, float]:
    """AdaBeam candidate identity.

    Legacy NucleoBench / v0.1.3 pools keyed ``RolloutNode`` by
    ``(seq, fitness)``. Fitness is usually determined by sequence for a
    deterministic oracle, but the production key still keeps both fields
    so a non-deterministic or state-dependent scorer cannot silently
    collapse distinct nodes.
    """
    return (node.seq, float(node.fitness))


def gradabeam_candidate_key(node: Any) -> tuple:
    """GradaBeam candidate identity.

    Legacy ``RolloutNodeWithProbs`` compared ``seq``, ``fitness``, and
    ``edits_since_root``, while hashing those fields plus the PBT fields
    (an invalid hash contract). The replacement key keeps every scalar
    that could distinguish nodes the set-based pool treated as distinct:

    ``(seq, fitness, edits_since_root, mutations_per_sequence, exploration_alpha)``

    Gradient arrays and other transient fields are excluded.
    """
    return (
        node.seq,
        float(node.fitness),
        node.edits_since_root,
        float(node.mutations_per_sequence),
        float(node.exploration_alpha),
    )


def rank_nodes_by_fitness(
    nodes: Sequence[Any],
    limit: int,
    tie_rng: np.random.Generator,
) -> list[Any]:
    """Rank by fitness using a dedicated tie RNG.

    Insertion order is preserved until this helper shuffles a copy with
    ``tie_rng`` and then stably sorts by fitness (descending). This is
    the only place tie randomization belongs. It must not be used by
    ``get_samples()`` and must not touch the mutation RNG.
    """
    ordered = list(nodes)
    tie_rng.shuffle(ordered)
    ordered.sort(key=lambda node: node.fitness, reverse=True)
    return ordered[:limit]


def sample_sequences_from_nodes(nodes: Sequence[Any], n_samples: int) -> list[str]:
    """Return the current beam prefix. Does not consume any RNG."""
    limit = min(n_samples, len(nodes))
    return [node.seq for node in nodes[:limit]]


def filter_accepted_children(
    children: Sequence[Any],
    parents: Sequence[Any],
    cur_rollout_length: int,
) -> tuple[list[Any], list[int]]:
    """Continue a rollout when ``child.fitness >= parent.fitness``."""
    new_nodes: list[Any] = []
    terminated_lengths: list[int] = []
    for child, parent in zip(children, parents):
        if child.fitness >= parent.fitness:
            new_nodes.append(child)
        else:
            terminated_lengths.append(cur_rollout_length)
    return new_nodes, terminated_lengths


class BeamOptimizerMixin:
    """Shared fitness batching and observationally pure sample retrieval."""

    model: ada_utils.ModelWrapper
    eval_batch_size: int
    current_nodes: list[Any]
    candidate_key_fn: Callable[[Any], Hashable]

    def get_batched_fitness(self, sequences: list[str]) -> np.ndarray:
        return ada_utils.get_batched_fitness(
            model_wrapper=self.model,
            sequences=sequences,
            batch_size=self.eval_batch_size,
        )

    def get_samples(self, n_samples: int) -> list[str]:
        """Pure projection of the current beam. Consumes no RNG."""
        return sample_sequences_from_nodes(self.current_nodes, n_samples)
