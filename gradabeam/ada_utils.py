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

    def str_in_cache(self, seq: str) -> bool:
        """Check if a sequence is in the cache."""
        k = xxhash.xxh64(seq).intdigest()
        return k in self.cache

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


def build_uniform_pos_and_chars(sequence: str, positions_to_mutate: list[int]) -> list[tuple[int, str]]:
    """Build a standard 3L actions list of (position, character) pairs.

    For each position in positions_to_mutate, we generate 3 non-reference actions.
    """
    all_bases = constants.VOCAB
    pos_and_chars = []
    for pos in positions_to_mutate:
        ref = sequence[pos]
        alts = [b for b in all_bases if b != ref]
        for alt in alts:
            pos_and_chars.append((pos, alt))
    return pos_and_chars


def generate_random_mutant_actionspace(
    sequence: str,
    pos_and_chars_to_mutate: PositionsAndCharactersType,
    n_edits: int,
    rng: np.random.Generator,
    probs: np.ndarray,
) -> tuple[str, list[int], np.ndarray, list[float]]:
    """Generate a mutant with exactly n_edits distinct position edits in action space.

    Uses the Plackett-Luce / marginal+conditional identity for O(3L) cost:

      1. Marginalize 3L action probs to per-position weights w_i = Σ_a p_{i,a}.
      2. Draw N distinct POSITIONS in one rng.choice(..., replace=False) call.
      3. Draw the BASE at each chosen position from the conditional p_{i,a}/w_i
         via vectorized inverse-CDF: (cumsum < u).sum(axis=1).

    This is distribution-identical to sequential action-draw with full-position
    masking, but avoids the per-edit Python loop, sum, normalize, and arange.

    Cap policy: the per-action 0.10 cap is applied once at root-level
    (in _logits_to_gradient_probs).  Across-step masking renormalizes survivors
    upward but does NOT re-cap — the cap is root-level smoothing only, not a
    maintained invariant.  This matches the prerefactor behavior on main.

    Args:
        sequence: Reference sequence to mutate.
        pos_and_chars_to_mutate: (position, character) pairs, positions-major,
            3 per position (non-reference bases in VOCAB order).
        n_edits: Intended number of distinct position edits.
        rng: Random number generator.
        probs: 1-D array of length 3L of current (mixed) action-probabilities.
            Must be nonnegative; used for both sampling and p_final.

    Returns:
        (mutant_string, selected_action_indices, final_masked_probs, p_final_chosen_list)
    """
    assert isinstance(pos_and_chars_to_mutate, list)
    n_actions = len(pos_and_chars_to_mutate)
    assert len(probs) == n_actions, f"Expected probs of length {n_actions}, got {len(probs)}"
    assert n_actions % 3 == 0, "Action space size must be a multiple of 3."
    assert n_edits >= 1, f"n_edits ({n_edits}) must be >= 1"

    current_probs = np.asarray(probs, dtype=np.float64).copy()
    assert np.all(current_probs >= 0), "All action probabilities must be nonnegative."
    assert np.any(current_probs > 0), "At least one action probability must be > 0."

    n_positions = n_actions // 3
    probs_2d = current_probs.reshape(n_positions, 3)

    # ── Step 1: marginal position weights ────────────────────────────────────
    pos_weights = probs_2d.sum(axis=1)  # shape (L,)
    available_positions = np.where(pos_weights > 0)[0]
    n_available = len(available_positions)
    effective_n = min(n_edits, n_available)

    # ── Step 2: draw N distinct positions in ONE call ────────────────────────
    w_avail = pos_weights[available_positions]
    chosen_positions = rng.choice(
        available_positions,
        size=effective_n,
        replace=False,
        p=w_avail / w_avail.sum(),
    )

    # ── Step 3: draw bases via vectorized inverse-CDF ────────────────────────
    chosen_rows = probs_2d[chosen_positions]           # (effective_n, 3)
    row_sums = np.maximum(chosen_rows.sum(axis=1, keepdims=True), 1e-30)
    cond_probs = chosen_rows / row_sums                # (effective_n, 3)
    cum = np.cumsum(cond_probs, axis=1)                # (effective_n, 3)
    u = rng.random(effective_n)                        # (effective_n,)
    # Vectorized per-row inverse-CDF: count how many CDF entries are < u[i]
    base_offsets = np.minimum((cum < u[:, None]).sum(axis=1), 2)  # shape (effective_n,)

    # ── Build action indices and apply edits ─────────────────────────────────
    selected_action_indices_arr = chosen_positions * 3 + base_offsets
    mutant = list(sequence)
    for action_idx in selected_action_indices_arr:
        pos, char = pos_and_chars_to_mutate[int(action_idx)]
        mutant[int(pos)] = str(char)
    selected_action_indices = selected_action_indices_arr.tolist()

    # ---------------------------------------------------------------------------
    # p_final for the α-posterior: STEP-START policy probability (reading B).
    #
    # p_final[k] = probs[action_k], i.e. the probability the step-start policy
    # assigned to the chosen action. It is deliberately the *unrenormalized*
    # probs value, NOT probs[action]/remaining_mass.
    #
    # WHY (this is the subtle part):
    #   The α-update is a surprise signal — "how much did the policy up/down-weight
    #   the locations this child actually mutated, vs chance." α measures trust in
    #   the ROOT gradient policy (1-α)g + α·U. The within-child sequential masking
    #   (mask a position after editing it, then renormalize the remaining N-1 draws)
    #   is a COMPUTATIONAL amortization trick — it funds many edits from one cached
    #   gradient. It is not part of the belief model about which locations are good.
    #
    #   If we renormalized p_final per draw (call it "reading A"), the surprise for
    #   each action would pick up a remaining_mass / n_available factor that depends
    #   on draw ORDER and on which OTHER positions this child happened to hit. Two
    #   children that mutate the same actions in different orders would then get
    #   different α nudges, and a 5-edit child would get systematically different α
    #   pressure than a 1-edit child on identical landscapes — coupling α (trust in
    #   location) to N (edit count), which is the *other*, independent PBT axis (μ).
    #   That is variance injected into the trust estimate for reasons unrelated to
    #   whether the gradient was right.
    #
    #   Reading B evaluates the surprise against the step-start policy, so it is
    #   invariant to edit count and draw order. It is also strictly simpler: no
    #   per-draw state, and _compute_child_alpha's p_uniform = 1/n_avail (over the
    #   step-start available actions) is already the correctly matched reference.
    #
    #   NOTE: across-STEP masking (rollout depth) still enters — it lives in the
    #   carried, renormalized `probs` the caller passes in (with already-edited
    #   positions zeroed) and in n_avail. Reading B only declines the extra
    #   WITHIN-child renormalization across this step's N draws.
    # ---------------------------------------------------------------------------
    p_final_chosen_list = [float(current_probs[int(a)]) for a in selected_action_indices_arr]

    # ── Position masking on the carried probs ────────────────────────────────
    for pos_idx in chosen_positions:
        current_probs[3 * pos_idx : 3 * pos_idx + 3] = 0.0

    total_p = current_probs.sum()
    if total_p > 0:
        current_probs /= total_p

    return "".join(mutant), selected_action_indices, current_probs, p_final_chosen_list


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
