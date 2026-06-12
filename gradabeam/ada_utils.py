"""Common utilities for [Gr]Ada*."""

import dataclasses
from collections import OrderedDict

import numpy as np
from scipy.stats import binom
import torch
import xxhash

from typing import Any, Callable
from gradabeam import constants


PositionsAndCharactersType = list[tuple[int, str]]
LogitsType = np.ndarray


@dataclasses.dataclass(frozen=True)
class RolloutNode:
    """Class for tracking rollout node.

    NOTE on terminology:

    a -> b -> c

    `a` is the root of `b` and `c`.
    `a` is the parent of `b`.
    `b` is the parent of `c`.

    """

    seq: str
    fitness: np.float32


class ModelWrapper:
    def __init__(
        self,
        model: Any,
        use_cache: bool = False,
        cache_limit: int = 100000,
        debug: bool = False,
        tism_cost: float | None = None,
        start_sequence: str | None = None,
    ):
        if tism_cost is not None:
            assert hasattr(model, "tism_torch"), (
                "Model must have tism_torch method. This is required for optimized get_tisms."
            )
        self.model = model
        self.cost: float = 0
        self.n_forward: int = 0
        self.n_backward: int = 0
        self.use_cache = use_cache
        self.cache_limit = cache_limit
        self.cache: OrderedDict[int, float] = OrderedDict()
        self.debug = debug
        self.tism_cost = tism_cost

        # Double check that the model is in evaluation mode.
        # TODO(joelshor): Force this to happen if the model is a PyTorch model.
        try:
            self.model.eval()
        except AttributeError:
            try:
                self.model.model.eval()
            except Exception:
                pass

        if self.tism_cost is not None:
            # Some optimizations for backprop:
            # We only need gradients for the input, so disable the rest.
            try:
                for param in self.model.parameters():
                    param.requires_grad = False
            except AttributeError:
                for param in (
                    self.model.model.parameters()
                ):  # Access the underlying torch module
                    param.requires_grad = False

        # The above is stochastic. Work around it.
        del start_sequence  # Unused.
        torch_opt_fn: Any
        if "Rinalmo" in type(self.model).__name__:
            torch_opt_fn = torch.no_grad
        else:
            torch_opt_fn = torch.inference_mode
        self.torch_opt_fn: Any = torch_opt_fn

    def get_fitness(self, m_input: list) -> list[float]:
        self.cost += len(m_input)

        if self.use_cache:
            # 1) Sift sequences into seen and unseen, keeping track of their location
            # so we can preserve order.
            # 2) Pull from the cache of the fitness of the seen sequences.
            seen_fitness, unseen_seq, unseen_hash = [], [], []
            for i, seq in enumerate(m_input):
                k = xxhash.xxh64(seq).intdigest()
                if k in self.cache:
                    self.cache.move_to_end(k)  # mark as recently used
                    seen_fitness.append((i, self.cache[k]))
                else:
                    unseen_seq.append((i, seq))
                    unseen_hash.append(k)
            m_input = [seq for _, seq in unseen_seq]

            if self.debug:
                if len(seen_fitness) > 0:
                    print(f"Cache hit: {len(seen_fitness)}")

        if len(m_input) == 0:
            results = []
        else:
            # `torch.inference_mode()` is faster than `torch.no_grad()`, but
            # doesn't work with RinAlmo's jit.compile optimization,
            # so we use the fastest we can.
            with self.torch_opt_fn():
                results = self.model(m_input)
            self.n_forward += len(m_input)

        if self.use_cache:
            # 3) Add the unseen sequences to the cache with LRU eviction.
            # 4) Interleave seen and unseen results to preserve order.
            for k, v in zip(unseen_hash, results):
                self.cache[k] = v
                if len(self.cache) > self.cache_limit:
                    evicted_key, _ = self.cache.popitem(last=False)
                    if self.debug:
                        print(
                            f"Cache limit reached. Evicting oldest entry ({evicted_key})."
                        )
            unseen_fitness = [(i, r) for (i, _), r in zip(unseen_seq, results)]
            results = [x[1] for x in sorted(seen_fitness + unseen_fitness)]

        # Ada* is formulated to maximize fitness, but we want to minimize.
        return [-float(x) for x in results]

    def get_tism(
        self,
        sequence: str,
        idxs: list[int] | None = None,
        debug: bool = False,
    ) -> tuple[PositionsAndCharactersType, LogitsType]:
        del debug  # Unused.
        assert hasattr(self.model, "tism_torch"), (
            "Model must have tism_torch method. This is required for optimized get_tisms."
        )

        if self.tism_cost is None:
            raise ValueError("Cost can't be None.")
        if self.tism_cost < 1.0:
            raise ValueError("Cost must be >= 1.0.")
        self.cost += self.tism_cost
        self.n_forward += 1
        self.n_backward += 1

        # Use fast tensor-based TISM
        pos_and_chars_to_mutate, logits = self.model.get_tism(sequence, idxs)

        logits *= -1  # Flip the sign, to conform to convention.

        return (pos_and_chars_to_mutate, logits)


def _F_inverse(mu: float, seq_len: int) -> float:
    """F_inverse = 1 - (1-mu')^l"""
    if mu >= 1.0:
        raise ValueError(
            f"_F_inverse requires mu < 1.0 (got mu={mu!r}). "
            "Check that mutations_per_sequence < len(positions_to_mutate) "
            "and that the PBT rate clamp is active."
        )
    return -np.expm1(seq_len * np.log1p(-mu))


def num_edits_likelihood_adabeam(
    num_edits: np.ndarray,
    seq_len: int,
    mu: float,
) -> float:
    """The likelihood of `num_edits` edits in the reference AdaBeam implementation.

    Thus,

    E[num locations edited] = F * mu * l

    Form:
    mu := mutation rate
    l := sequence length
    n := number of edits
    Binom(n, l, mu) := binomial distribution

    with
    F := 1 / (1 - (1-mu)^l)

    =>
    Pr[N locations edited] = 0, if N <= 0, N > l
    Pr[N locations edited] = Binom(n, l, mu) * F, otherwise

    E[num locations edited] = F * mu * l


    NOTE: For numerical accuracy, we note the following:

    (1 - mu')^l = exp( log( 1 - epsilon)^l ) )
                = exp( l * log( 1 + (-epsilon) ) ) )
                = exp( l * np.log1p(-epsilon) )
    """
    assert isinstance(num_edits, np.ndarray)
    if num_edits.min() < 0 or num_edits.max() > seq_len:
        raise ValueError("num_edits must be between 0 and seq_len, inclusive.")

    # Using the notation from above.
    F_inverse = _F_inverse(mu, seq_len)

    probs = binom.pmf(num_edits, seq_len, mu) / F_inverse

    # The Binomial distribution has support at k=0, but AdaBeam defines P(0)=0.
    # We force any element where num_edits == 0 to have probability 0.0.
    probs[num_edits == 0] = 0.0

    return probs


class NumberEditsSampler(object):
    """Vectorized samples the number of edits to make."""

    def __init__(
        self,
        sequence_len: int,
        mutation_rate: float,
        likelihood_fn: Callable[..., Any],
        rng_seed: int = 0,
    ):

        self.seq_len = sequence_len
        self.mu = mutation_rate
        self.rng = np.random.default_rng(rng_seed)

        self.num_edits = np.arange(1, self.seq_len + 1, dtype=np.uint32)

        self.probs = likelihood_fn(self.num_edits, self.seq_len, self.mu)

    def expected_num_edits(self) -> float:
        """Returns the expected number of edits."""
        return np.sum(self.num_edits * self.probs)

    def sample(self, n_samples: int) -> np.ndarray:
        # OPTIMIZATION: Use numpy array directly - faster than converting from list.
        return self.rng.choice(self.num_edits, size=n_samples, p=self.probs)


class NumberEditsSamplerAdaBeam(NumberEditsSampler):
    """Samples the number of edits to make."""

    def __init__(self, sequence_len: int, mutation_rate: float, rng_seed: int = 0):

        super().__init__(
            sequence_len=sequence_len,
            mutation_rate=mutation_rate,
            rng_seed=rng_seed,
            likelihood_fn=num_edits_likelihood_adabeam,
        )


def generate_random_mutant_tism(
    sequence: str,
    pos_and_chars_to_mutate: PositionsAndCharactersType,
    random_n_loc: int,
    rng: np.random.Generator,
    probs: np.ndarray,
    debug: bool = False,
) -> tuple[str, list[int]]:
    """
    Generate a mutant of `sequence` with exactly `random_n_loc` edits, using `tism` info.

    Args:
        sequence: Sequence that will be mutated from.
        pos_and_chars_to_mutate: (position, character) of the allowed positions to be mutated.
        random_n_loc: Number of mutations per sequence.
        alphabet: Alphabet string.
        rng: Random number generator.
        probs: XXX
        debug: If True, print debug info.

    Returns:
        Mutant sequence string and indices within the mutable positions that were mutated.

    """
    assert isinstance(pos_and_chars_to_mutate, list)

    # OPTIMIZATION: Use integer indices instead of tuples for faster rng.choice
    # NumPy's rng.choice is much faster when working with integer arrays
    n_actions = len(pos_and_chars_to_mutate)
    indices = np.arange(n_actions, dtype=np.uint32)

    selected_indices = rng.choice(indices, size=random_n_loc, replace=False, p=probs)
    assert len(selected_indices) == random_n_loc

    mutant, rel_pos_of_mutations = list(sequence), []
    for i in selected_indices:
        pos, char = pos_and_chars_to_mutate[i]
        mutant[int(pos)] = str(char)
        rel_pos_of_mutations.append(
            i
        )  # Use relative position, which is needed downstream.
    return "".join(mutant), rel_pos_of_mutations


def tism_probs_to_position_weights(
    probs_3L: np.ndarray,
    n_positions: int,
) -> np.ndarray:
    """Marginalize a 3L TISM action-value vector to a per-position weight vector.

    Action-ordering assumption (verified in tism.TISMModelClass.get_tism):
      Before the reference-base mask, the flat layout is positions-major with vocab
      tiled: [(p0,A),(p0,C),(p0,G),(p0,T), (p1,A),...].  The mask removes exactly
      one entry per position (the reference base), leaving the three non-reference
      actions for position i in the contiguous slice [3i : 3i+3] of the masked
      output.  Therefore reshape(-1, 3).sum(axis=1) correctly recovers the sum of
      each position's three action entries.

    Args:
        probs_3L: 1-D array of length 3 * n_positions containing nonnegative
            values.  No normalization is required or applied; the function returns
            the raw per-position sums of each position's three action entries.
            Pass a softmax-normalized vector to obtain P(position i is touched)
            under the current distribution; pass raw logits or counts for other
            uses.
        n_positions: number of mutable positions (L).

    Returns:
        1-D array of length n_positions containing the sum of the three action
        entries for each position.
    """
    assert len(probs_3L) == 3 * n_positions, (
        f"Expected len(probs_3L) == 3 * n_positions = {3 * n_positions}, "
        f"got {len(probs_3L)}."
    )
    return probs_3L.reshape(n_positions, 3).sum(axis=1)


def generate_random_mutant_positionspace(
    sequence: str,
    mutable_positions: list[int],
    position_weights: np.ndarray,
    n_edits: int,
    rng: np.random.Generator,
) -> tuple[str, list[int]]:
    """Generate a mutant with exactly n_edits distinct edits in position space.

    Unlike generate_random_mutant_tism (which samples in the 3L action space and
    can collide when two selected actions share a position), this function samples
    n_edits distinct positions first, then picks the new base uniformly from the
    3 non-reference bases.  This guarantees exactly n_edits distinct edits.

    Args:
        sequence: Reference sequence to mutate.
        mutable_positions: Absolute (0-based) positions that may be edited.
        position_weights: Non-negative weight for each mutable position (same
            length as mutable_positions).  Need not be normalized; normalized
            internally.
        n_edits: Number of distinct positions to edit.  Must satisfy
            1 <= n_edits <= len(mutable_positions).  Callers must handle the
            no-positions-left case (n_edits == 0) before calling; this function
            never silently returns an unedited sequence.
        rng: NumPy random Generator.

    Returns:
        (mutant_string, edited_positions) where edited_positions is the list of
        absolute (0-based) positions that were changed, in the order they were
        selected.
    """
    # Fix 4: convert to float64 array once; run all weight asserts on the array.
    weights = np.asarray(position_weights, dtype=np.float64)

    # Fix 2: enforce lower bound before upper bound.
    assert n_edits >= 1, (
        "n_edits must be >= 1; callers must handle the no-positions-left case "
        "before calling."
    )
    assert n_edits <= len(mutable_positions), (
        f"n_edits ({n_edits}) must be <= len(mutable_positions) "
        f"({len(mutable_positions)})."
    )
    assert len(weights) == len(mutable_positions), (
        f"position_weights length ({len(weights)}) must equal "
        f"len(mutable_positions) ({len(mutable_positions)})."
    )
    assert np.all(weights >= 0), "All position_weights must be nonnegative."
    assert np.any(weights > 0), "At least one position_weight must be > 0."

    weights = weights / weights.sum()

    chosen_positions = rng.choice(
        np.asarray(mutable_positions, dtype=np.int64),
        size=n_edits,
        replace=False,
        p=weights,
    )

    # Build the set of 3 non-reference bases once per chosen position.
    all_bases = constants.VOCAB  # ["A", "C", "G", "T"]
    mutant = list(sequence)
    for pos in chosen_positions:
        ref_base = sequence[int(pos)]
        alt_bases = [b for b in all_bases if b != ref_base]
        # Fix 3: str() ensures a plain Python str, not a numpy str scalar.
        new_base = str(rng.choice(alt_bases))
        mutant[int(pos)] = new_base

    return "".join(mutant), [int(p) for p in chosen_positions]


def get_batched_fitness(
    model_wrapper: ModelWrapper,
    sequences: list[str],
    batch_size: int,
) -> np.ndarray:
    """Get fitness for a list of sequences in batches."""
    if len(sequences) == 0:
        return np.array([])

    fitness = []
    for i in range(0, len(sequences), batch_size):
        batch = sequences[i : i + batch_size]
        batch_fitness = model_wrapper.get_fitness(batch)
        assert isinstance(batch_fitness, list)
        for x in batch_fitness:
            assert isinstance(x, float), (type(x), x)
        fitness.extend(batch_fitness)

    return np.array(fitness)
