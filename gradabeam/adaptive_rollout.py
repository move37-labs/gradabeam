"""AdaptiveRolloutDesigner — unified designer with injected mutation strategy.

Root-step equivalence property
-------------------------------
Before any position is masked, the position-marginal of the old action-level
mixture  (1-α)·grad_action + α·unif_action  equals the new position-level
mixture  (1-α)·masked_grad + α·unif_w  — because summing a (position,base)
action distribution over bases is linear and the uniform base-draw post-hoc is
equivalent to the uniform term in the original action mixture.  So root-step
position-selection behavior is UNCHANGED from the old code; the only intended
divergence is the (previously buggy) multi-step masking.
"""

from __future__ import annotations

import dataclasses
from dataclasses import field
from functools import lru_cache
from typing import Any

import numpy as np
from scipy.special import softmax

from gradabeam import ada_utils
from gradabeam import constants
from gradabeam import opt_utils
from gradabeam import testing_utils


PositionsAndCharactersType = ada_utils.PositionsAndCharactersType


# ---------------------------------------------------------------------------
# Extended rollout-node type
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class RolloutNodeWithProbs(ada_utils.RolloutNode):
    """Rollout node that carries gradient + position-space state.

    Field notes
    -----------
    probs : np.ndarray or None
        3L mixed action-probability vector (kept for backward compatibility with
        tests that inspect the gradient-action distribution).
    pos_and_chars : list[tuple[int, str]] or None
        (position, character) pairs from the TISM call — kept for compat.
    edits_since_root : int or None
        Depth in the current rollout chain, starting at 0 for roots.
    mutations_per_sequence : float
        Current per-step edit-rate target (mutated by PBT).
    exploration_alpha : float
        Current mixing coefficient — 0 = pure gradient, 1 = pure uniform.
    position_weights : np.ndarray or None
        L-vector.  Non-zero only at still-available positions after position-
        space masking.  Renormalized over available positions after each edit.
        Becomes all-zeros when all mutable positions have been consumed in
        the current rollout chain (signals exhaustion to the rollout loop).
        None for the legacy (allow_silent_edits=True) path.
    gradient_position_weights : np.ndarray or None
        L-vector of pure-gradient position weights from the most recent TISM
        call, before any masking.  Used to recompute P_final for the α-update.
        None for the legacy path and for the corrected gradient-free path.
    """

    probs: np.ndarray | None = field(default=None, hash=False, compare=False)
    pos_and_chars: PositionsAndCharactersType | None = field(
        default=None, hash=False, compare=False
    )
    edits_since_root: int | None = None
    mutations_per_sequence: float = dataclasses.field(
        default=1.0, compare=False, hash=False
    )
    exploration_alpha: float = dataclasses.field(
        default=0.05, compare=False, hash=False
    )
    # ── position-space fields ───────────────────────────────────────────────
    position_weights: np.ndarray | None = field(default=None, hash=False, compare=False)
    gradient_position_weights: np.ndarray | None = field(
        default=None, hash=False, compare=False
    )

    @property
    def sort_key(self) -> tuple:
        """Deterministic total ordering; id(self) breaks rare float-ties."""
        return (
            self.fitness,
            self.seq,
            self.edits_since_root,
            self.mutations_per_sequence,
            self.exploration_alpha,
            id(self),
        )


# ---------------------------------------------------------------------------
# Strategy objects
# ---------------------------------------------------------------------------


class UniformPositionStrategy:
    """Uniform position weights.

    Parameters
    ----------
    allow_silent_edits : bool
        True  → legacy/reproduction path (generate_random_mutant_v2,
                 ~25% silent edits, exact RNG match with published AdaBeam).
                 Reachable only via explicit opt-in.
        False → corrected path (position-space operator, never silent,
                 uniform weights over available positions).  This is the
                 "corrected AdaBeam" referenced in Plan 01 §5.
    """

    def __init__(self, allow_silent_edits: bool = True) -> None:
        self.allow_silent_edits = allow_silent_edits

    def is_legacy(self) -> bool:
        """True when the legacy AdaBeam operator should be used verbatim."""
        return self.allow_silent_edits

    def propose_positions(
        self,
        node: RolloutNodeWithProbs,
        n_edits: int,
        rng: np.random.Generator,
        mutable_positions: list[int],
    ) -> tuple[list[int], np.ndarray]:
        """Select n_edits positions uniformly; return updated position weights.

        Only called when allow_silent_edits=False (corrected path).

        Returns
        -------
        chosen_positions : list[int]
            Absolute 0-based positions selected for mutation.
        new_position_weights : np.ndarray
            L-vector after zeroing chosen positions and renormalizing.
            All-zeros when the last available positions have been consumed;
            the rollout loop treats an all-zero vector as exhaustion.
        """
        assert not self.allow_silent_edits, (
            "propose_positions must not be called on the legacy path; "
            "check strategy.is_legacy() first."
        )
        n = len(mutable_positions)

        pw = (
            node.position_weights.copy()
            if node.position_weights is not None
            else np.ones(n, dtype=np.float64)
        )
        avail_mask = pw > 0
        n_available = int(avail_mask.sum())
        avail_positions = [p for p, m in zip(mutable_positions, avail_mask) if m]

        effective_n = min(n_edits, n_available)
        assert effective_n >= 1, (
            "propose_positions called with no available positions; "
            "the rollout must check exhaustion before calling this method."
        )

        avail_w = pw[avail_mask] / pw[avail_mask].sum()
        chosen = rng.choice(
            np.array(avail_positions, dtype=np.int64),
            size=effective_n,
            replace=False,
            p=avail_w,
        )
        chosen_list = [int(c) for c in chosen]

        pos_to_idx = {p: i for i, p in enumerate(mutable_positions)}
        new_pw = pw.copy()
        for p in chosen_list:
            new_pw[pos_to_idx[p]] = 0.0
        total = new_pw.sum()
        if total > 0:
            new_pw /= total
        else:
            # All mutable positions now consumed; return zero vector.
            # The rollout loop will detect this on the NEXT step and terminate
            # the chain rather than calling propose_positions again.
            new_pw = np.zeros(n, dtype=np.float64)

        return chosen_list, new_pw


class GradientPositionStrategy:
    """Gradient-guided position weights (Plan 01 §2).

    Marginalizes the 3L TISM distribution to a per-position weight via
    ``tism_probs_to_position_weights``, then mixes with uniform:
        P_final = (1-α)·grad_w + α·unif_w
    over the currently-available (unmasked) positions.

    This also provides the P_final values needed by the α-posterior update
    in AdaptiveRolloutDesigner._compute_child_alpha.
    """

    def is_legacy(self) -> bool:
        return False

    def propose_positions(
        self,
        node: RolloutNodeWithProbs,
        n_edits: int,
        rng: np.random.Generator,
        mutable_positions: list[int],
    ) -> tuple[list[int], np.ndarray, np.ndarray]:
        """Select positions according to gradient+uniform mixture.

        Returns
        -------
        chosen_positions : list[int]
            Absolute 0-based positions.
        new_position_weights : np.ndarray
            Updated (masked) position weights.  All-zeros when the last
            available positions are consumed.
        p_final_chosen : np.ndarray
            P_final(j) for each chosen position j — used by the α-update.
        """
        assert node.gradient_position_weights is not None, (
            "GradientPositionStrategy requires gradient_position_weights on node."
        )
        assert node.position_weights is not None, (
            "GradientPositionStrategy requires position_weights on node."
        )

        n = len(mutable_positions)
        alpha = node.exploration_alpha
        pw = node.position_weights
        grad_w_full = node.gradient_position_weights

        avail_mask = pw > 0
        n_available = int(avail_mask.sum())
        avail_positions = [p for p, m in zip(mutable_positions, avail_mask) if m]

        effective_n = min(n_edits, n_available)
        assert effective_n >= 1, "propose_positions called with no available positions."

        masked_grad: np.ndarray = grad_w_full * avail_mask.astype(np.float64)
        grad_sum = masked_grad.sum()
        if grad_sum > 0:
            masked_grad = masked_grad / grad_sum
        else:
            masked_grad = avail_mask.astype(np.float64) / n_available

        unif_w = avail_mask.astype(np.float64) / n_available
        p_final_all = (1.0 - alpha) * masked_grad + alpha * unif_w
        p_final_avail = p_final_all[avail_mask]
        p_final_avail = p_final_avail / p_final_avail.sum()

        avail_pos_arr = np.array(avail_positions, dtype=np.int64)
        chosen = rng.choice(
            avail_pos_arr, size=effective_n, replace=False, p=p_final_avail
        )
        chosen_list = [int(c) for c in chosen]

        pos_to_avail_idx = {p: i for i, p in enumerate(avail_positions)}
        p_final_chosen = np.array(
            [p_final_avail[pos_to_avail_idx[p]] for p in chosen_list],
            dtype=np.float64,
        )

        pos_to_idx = {p: i for i, p in enumerate(mutable_positions)}
        new_pw = pw.copy()
        for p in chosen_list:
            new_pw[pos_to_idx[p]] = 0.0
        total = new_pw.sum()
        if total > 0:
            new_pw /= total
        else:
            # All mutable positions now consumed; zero vector signals exhaustion.
            new_pw = np.zeros(n, dtype=np.float64)

        return chosen_list, new_pw, p_final_chosen


# ---------------------------------------------------------------------------
# AdaptiveRolloutDesigner — unified optimizer
# ---------------------------------------------------------------------------


class AdaptiveRolloutDesigner:
    """Unified beam-search sequence designer.

    Three operator/gradient paths (routed by propose_sequences):

    +-----------------------------+----------------+------------------------------+
    | strategy.is_legacy()        | use_gradients  | path                         |
    +-----------------------------+----------------+------------------------------+
    | True  (allow_silent_edits)  | False (only)   | legacy / reproduction-only   |
    | False                       | False          | corrected gradient-free      |
    | False                       | True           | gradient-guided (GradaBeam)  |
    +-----------------------------+----------------+------------------------------+

    Parameters
    ----------
    strategy : UniformPositionStrategy | GradientPositionStrategy
        Controls how candidate positions are selected at each rollout step.
    use_gradients : bool
        When True, compute TISM at each rollout root.
        When False, skip TISM (no tism_cost charged to ModelWrapper).
    allow_silent_edits : bool
        True  → legacy operator (~25% silent edits, bit-for-bit RNG match
                 with published AdaBeam).  Must use with use_gradients=False.
        False → corrected position-space operator (never silent).
    use_pbt : bool
        Enable Population Based Training for adaptive mutation rate and α.
    exploration_alpha : float
        Initial mixing coefficient (0=pure gradient, 1=pure uniform).
    skip_repeat_sequences : bool
        Legacy AdaBeam option: retry mutation until a novel sequence is found.
    """

    def __init__(
        self,
        model_fn: Any,
        start_sequence: str,
        mutations_per_sequence: float,
        beam_size: int,
        n_rollouts_per_root: int,
        strategy: UniformPositionStrategy | GradientPositionStrategy,
        use_gradients: bool,
        allow_silent_edits: bool,
        use_pbt: bool,
        exploration_alpha: float = 0.05,
        gradient_prob_cap: float = 0.10,
        max_logit: float = 3.0,
        rng_seed: int = 0,
        positions_to_mutate: list[int] | None = None,
        eval_batch_size: int = 1,
        max_rollout_len: int = 200,
        debug: bool = False,
        skip_repeat_sequences: bool = False,
    ) -> None:
        self.positions_to_mutate: list[int] = positions_to_mutate or list(
            range(len(start_sequence))
        )
        self.tism_positions: list[int] | None = (
            None
            if len(self.positions_to_mutate) == len(start_sequence)
            else self.positions_to_mutate
        )

        assert min(self.positions_to_mutate) >= 0
        assert max(self.positions_to_mutate) < len(start_sequence)
        assert mutations_per_sequence > 0
        assert mutations_per_sequence <= len(self.positions_to_mutate), (
            f"mutations_per_sequence ({mutations_per_sequence}) must be <= "
            f"len(positions_to_mutate) ({len(self.positions_to_mutate)})"
        )
        assert beam_size > 0
        assert n_rollouts_per_root > 0
        if use_gradients:
            assert 0.0 <= exploration_alpha <= 1.0

        # Strategy / gradient combination validation
        if strategy.is_legacy() and use_gradients:
            raise ValueError(
                "allow_silent_edits=True (legacy) is incompatible with "
                "use_gradients=True.  The legacy operator is reproduction-only "
                "and does not use TISM."
            )
        if isinstance(strategy, GradientPositionStrategy) and not use_gradients:
            raise ValueError("GradientPositionStrategy requires use_gradients=True.")

        self.strategy = strategy
        self.use_gradients = use_gradients
        self.allow_silent_edits = allow_silent_edits
        self.use_pbt = use_pbt
        self.exploration_alpha = exploration_alpha
        self.gradient_prob_cap = gradient_prob_cap
        self.max_logit = max_logit
        self.skip_repeat_sequences = skip_repeat_sequences

        self.model = ada_utils.ModelWrapper(
            model_fn,
            use_cache=True,
            debug=debug,
            tism_cost=1.0 if use_gradients else None,
            start_sequence=start_sequence,
        )
        self.start_sequence = start_sequence
        self.beam_size = beam_size
        self.n_rollouts_per_root = n_rollouts_per_root
        self.alphabet = "".join(constants.VOCAB)
        self.eval_batch_size = eval_batch_size
        self.rng_seed = rng_seed
        self.rng = np.random.default_rng(rng_seed)
        self.max_rollout_len = max_rollout_len
        self.debug = debug
        self.mu = float(mutations_per_sequence) / len(self.positions_to_mutate)

        # ── sampler setup ────────────────────────────────────────────────────
        # Both legacy and corrected-gradient-free paths use a single fixed-rate
        # sampler (matches AdaBeam's sampler structure).  The gradient path uses
        # the PBT-per-node get_sampler() / _get_sampler_cached() instead.
        if not use_gradients:
            self.num_mutations_sampler: ada_utils.NumberEditsSampler = (
                ada_utils.NumberEditsSamplerAdaBeam(
                    sequence_len=len(self.positions_to_mutate),
                    mutation_rate=self.mu,
                    rng_seed=rng_seed,
                )
            )

        # ── best-ever tracker ────────────────────────────────────────────────
        self.best_ever = opt_utils.BestEver(
            sort_key=lambda x: (x.fitness, x.seq),
            capacity=beam_size,
        )

        # Filled by propose_sequences; read by tests.
        self.last_all_proposals: list[dict] = []

        # ── initial beam ─────────────────────────────────────────────────────
        if strategy.is_legacy():
            self._init_beam_legacy(start_sequence, beam_size, mutations_per_sequence)
        elif use_gradients:
            self._init_beam_gradient(start_sequence, beam_size, mutations_per_sequence)
        else:
            self._init_beam_positionspace(
                start_sequence, beam_size, mutations_per_sequence
            )

        self.best_ever.update(self.current_nodes)

    # ── sampler helpers (gradient/PBT path) ─────────────────────────────────

    def get_sampler(
        self, mutations_per_sequence: float
    ) -> ada_utils.NumberEditsSampler:
        rounded = round(mutations_per_sequence, 4)
        return self._get_sampler_cached(rounded)

    @lru_cache(maxsize=256)
    def _get_sampler_cached(
        self, mutations_per_sequence: float
    ) -> ada_utils.NumberEditsSampler:
        mu = mutations_per_sequence / len(self.positions_to_mutate)
        rate_int = int(round(mutations_per_sequence * 10000))
        child_seed = int(
            np.random.SeedSequence([self.rng_seed, rate_int]).generate_state(1)[0]
        )
        return ada_utils.NumberEditsSamplerAdaBeam(
            sequence_len=len(self.positions_to_mutate),
            mutation_rate=mu,
            rng_seed=child_seed,
        )

    # ── initial beam helpers ─────────────────────────────────────────────────

    def _init_beam_legacy(
        self,
        start_sequence: str,
        beam_size: int,
        mutations_per_sequence: float,
    ) -> None:
        """AdaBeam-compatible initial beam (no TISM, uses legacy operator)."""
        seed_node = ada_utils.RolloutNode(
            seq=start_sequence, fitness=np.float32(np.nan)
        )
        num_edit_locs = self.num_mutations_sampler.sample(beam_size)
        self.current_nodes: list = []
        for i in range(0, beam_size, self.eval_batch_size):
            cur_edits = num_edit_locs[i : i + self.eval_batch_size]
            self.current_nodes.extend(
                self._mutate_legacy_nodes([seed_node] * len(cur_edits), cur_edits)
            )

    def _init_beam_gradient(
        self,
        start_sequence: str,
        beam_size: int,
        mutations_per_sequence: float,
    ) -> None:
        """GradaBeam-style initial beam with TISM gradients."""
        seed_node = RolloutNodeWithProbs(
            seq=start_sequence,
            fitness=np.float32(0.0),
            edits_since_root=0,
            probs=None,
            pos_and_chars=None,
            mutations_per_sequence=float(mutations_per_sequence),
            exploration_alpha=float(self.exploration_alpha),
            position_weights=None,
            gradient_position_weights=None,
        )
        initialized_roots = self.initialize_roots_with_gradients(
            [seed_node] * beam_size
        )
        initial_sampler = self.get_sampler(seed_node.mutations_per_sequence)
        num_edit_locs = [int(x) for x in initial_sampler.sample(beam_size)]
        self.current_nodes = []
        for i in range(0, beam_size, self.eval_batch_size):
            cur_edits = num_edit_locs[i : i + self.eval_batch_size]
            cur_roots = initialized_roots[i : i + self.eval_batch_size]
            self.current_nodes.extend(
                self._mutate_gradient_nodes(
                    cur_roots,
                    cur_edits,
                    [seed_node.mutations_per_sequence] * len(cur_edits),
                )
            )

    def _init_beam_positionspace(
        self,
        start_sequence: str,
        beam_size: int,
        mutations_per_sequence: float,
    ) -> None:
        """Corrected gradient-free initial beam (uniform position weights).

        Bug 2 (NaN fitness) analysis: seed_node carries fitness=np.float32(nan)
        — the same convention used by _init_beam_legacy.  This NaN is safe:
          * _mutate_gradient_nodes only reads node.seq, node.position_weights,
            node.exploration_alpha, and node.edits_since_root from the seed;
            children receive their fitness from get_batched_fitness(), not from
            the seed.
          * seed_node is never appended to current_nodes; only its children are.
          * No keep/reject comparison (child.fitness >= cmp_node.fitness) is
            performed against the seed_node.
        Therefore NaN cannot propagate to any comparison or tracker.
        """
        n = len(self.positions_to_mutate)
        init_pw = np.ones(n, dtype=np.float64) / n
        seed_node = RolloutNodeWithProbs(
            seq=start_sequence,
            fitness=np.float32(
                np.nan
            ),  # safe: NaN is never read after children are made
            edits_since_root=0,
            mutations_per_sequence=float(mutations_per_sequence),
            exploration_alpha=float(self.exploration_alpha),
            position_weights=init_pw.copy(),
            gradient_position_weights=None,
        )
        num_edit_locs = [int(x) for x in self.num_mutations_sampler.sample(beam_size)]
        self.current_nodes = []
        for i in range(0, beam_size, self.eval_batch_size):
            cur_edits = num_edit_locs[i : i + self.eval_batch_size]
            self.current_nodes.extend(
                self._mutate_gradient_nodes(
                    [seed_node] * len(cur_edits),
                    cur_edits,
                    [float(mutations_per_sequence)] * len(cur_edits),
                )
            )

    # ── public API ───────────────────────────────────────────────────────────

    def run(self, n_steps: int) -> None:
        for _step in range(n_steps):
            self.current_nodes = self.propose_sequences(self.current_nodes)
            self.best_ever.update(self.current_nodes)
            if self.debug and self.current_nodes:
                print(f"Step {_step} top score: {self.current_nodes[0].fitness}")

    def get_samples(self, n_samples: int) -> list[str]:
        return [x.seq for x in self.best_ever.best(n_samples)]

    def get_batched_fitness(self, sequences: list[str]) -> np.ndarray:
        return ada_utils.get_batched_fitness(
            model_wrapper=self.model,
            sequences=sequences,
            batch_size=self.eval_batch_size,
        )

    def propose_sequences(self, root_nodes: list) -> list:
        """Route to the correct operator path based on strategy and gradient flag.

        Routing table:
          strategy.is_legacy()=True              → _propose_sequences_legacy
          strategy.is_legacy()=False, no grads   → _propose_sequences_positionspace
          strategy.is_legacy()=False, with grads → _propose_sequences_gradient
        """
        if self.strategy.is_legacy():
            return self._propose_sequences_legacy(root_nodes)
        elif not self.use_gradients:
            return self._propose_sequences_positionspace(root_nodes)
        else:
            return self._propose_sequences_gradient(root_nodes)

    # ── Path 1: legacy / reproduction-only ──────────────────────────────────

    def _propose_sequences_legacy(self, root_nodes: list) -> list:
        """Verbatim replication of AdaBeam.propose_sequences for bit-for-bit match.

        REPRODUCTION-ONLY: exact RNG-for-RNG replication of published AdaBeam
        (generate_random_mutant_v2, 4-base sampling, ~25% silent edits).
        Reachable only via allow_silent_edits=True; never the default, never
        used on the corrected or gradient paths.  Pinned by
        test_adabeam_equivalence — do not "simplify" or remove without
        regenerating the golden fixture and the published-paper RNG analysis.

        Uses ada_utils.RolloutNode (simple seq+fitness) so the set-deduplication
        semantics match AdaBeam exactly.  Sorts by (fitness, seq) only.
        """
        sequences: set = set()
        rollout_lengths: list[int] = []
        root_nodes_effective = root_nodes * self.n_rollouts_per_root

        for i in range(0, len(root_nodes_effective), self.eval_batch_size):
            cur_root_nodes = root_nodes_effective[i : i + self.eval_batch_size]
            parent_nodes = cur_root_nodes
            cur_rollout_length = 0

            while len(parent_nodes) > 0 and cur_rollout_length < self.max_rollout_len:
                num_edit_locs = self.num_mutations_sampler.sample(len(parent_nodes))
                children = self._mutate_legacy_nodes(parent_nodes, num_edit_locs)
                sequences.update(children)
                cur_rollout_length += 1

                new_nodes = []
                for child, cmp_node in zip(children, parent_nodes):
                    if child.fitness >= cmp_node.fitness:
                        new_nodes.append(child)
                    else:
                        rollout_lengths.append(cur_rollout_length)
                parent_nodes = new_nodes

        if not sequences:
            raise ValueError("No sequences generated.")

        sorted_sequences = sorted(
            sequences, key=lambda x: (x.fitness, x.seq), reverse=True
        )
        self.last_all_proposals = [
            {"seq": n.seq, "fitness": float(n.fitness)} for n in sorted_sequences
        ]
        return sorted_sequences[: self.beam_size]

    def _mutate_legacy_nodes(
        self,
        nodes: list,
        num_edit_locs: list[int] | np.ndarray,
        max_num_tries: int = 300,
    ) -> list:
        """Mutation using generate_random_mutant_v2 — exact AdaBeam RNG pattern.

        REPRODUCTION-ONLY: preserves the 4-base (A/C/G/T) sampling that allows
        ~25% silent edits, matching the published AdaBeam exactly.  Do not
        substitute generate_random_mutant_positionspace here.
        """
        assert len(nodes) == len(num_edit_locs) <= self.eval_batch_size
        seqs = []
        for n, random_n_loc in zip(nodes, num_edit_locs):
            try_cnt = 0
            while True:
                candidate = ada_utils.generate_random_mutant_v2(
                    sequence=n.seq,
                    positions_to_mutate=self.positions_to_mutate,
                    random_n_loc=int(random_n_loc),
                    alphabet=self.alphabet,
                    rng=self.rng,
                )
                try_cnt += 1
                if not self.skip_repeat_sequences or not self.model.str_in_cache(
                    candidate
                ):
                    break
                if try_cnt > max_num_tries:
                    raise ValueError(
                        f"Couldn't find unique child after {try_cnt} tries."
                    )
                if self.debug and try_cnt % 50 == 0:
                    print(f"Couldn't find unique child after {try_cnt} tries…")
            seqs.append(candidate)

        fitnesses = self.get_batched_fitness(seqs)
        return [
            ada_utils.RolloutNode(seq=seq, fitness=np.float32(float(f)))
            for seq, f in zip(seqs, fitnesses)
        ]

    # ── Path 2: corrected gradient-free (corrected AdaBeam) ─────────────────

    def _propose_sequences_positionspace(self, root_nodes: list) -> list:
        """Corrected gradient-free rollout using position-space operator.

        This is the scientific comparison point for Plan 01: "corrected AdaBeam"
        ≡ GradaBeam with gradients off + uniform weights.  No TISM is computed.
        Positions are selected uniformly from those not yet edited in the current
        rollout chain.  Rollout chains terminate when all positions are exhausted.

        Rollout-length convention: same as _rollout.  Both exhaustion (recorded
        before the increment) and rejection (recorded after) capture
        cur_rollout_length = number of oracle calls made in the chain.
        See _rollout docstring for the full explanation.

        Sets self.last_rollout_lengths for testability.
        """
        nodes_visited: set = set()
        all_rollout_lengths: list[int] = []

        root_nodes_effective = root_nodes * self.n_rollouts_per_root
        for i in range(0, len(root_nodes_effective), self.eval_batch_size):
            cur_root_nodes = root_nodes_effective[i : i + self.eval_batch_size]
            # Attach fresh uniform position_weights; no TISM.
            parent_nodes = [
                self._attach_uniform_position_weights(n) for n in cur_root_nodes
            ]

            cur_rollout_length = 0
            while len(parent_nodes) > 0 and cur_rollout_length < self.max_rollout_len:
                # Exhaustion check BEFORE generating; cur_rollout_length = mutations made.
                parent_nodes, exhausted = self._filter_exhausted(parent_nodes)
                all_rollout_lengths.extend([cur_rollout_length] * len(exhausted))
                if not parent_nodes:
                    break

                num_edit_locs = [
                    int(x) for x in self.num_mutations_sampler.sample(len(parent_nodes))
                ]
                children = self._mutate_gradient_nodes(
                    parent_nodes,
                    num_edit_locs,
                    [n.mutations_per_sequence for n in parent_nodes],
                )
                nodes_visited.update(children)
                cur_rollout_length += 1  # incremented AFTER generating

                new_nodes = []
                for child, cmp_node in zip(children, parent_nodes):
                    if child.fitness >= cmp_node.fitness:
                        new_nodes.append(child)
                    else:
                        # Rejection AFTER increment; cur_rollout_length = mutations made.
                        all_rollout_lengths.append(cur_rollout_length)
                parent_nodes = new_nodes

        self.last_rollout_lengths = all_rollout_lengths  # exposed for tests

        if not nodes_visited:
            raise ValueError("No nodes generated.")

        sorted_nodes = sorted(nodes_visited, key=lambda x: x.sort_key, reverse=True)
        self.last_all_proposals = [
            {"seq": n.seq, "fitness": float(n.fitness)} for n in sorted_nodes
        ]
        return sorted_nodes[: self.beam_size]

    def _attach_uniform_position_weights(self, node: Any) -> RolloutNodeWithProbs:
        """Return a RolloutNodeWithProbs with fresh uniform position weights.

        Used to initialize each rollout in the corrected gradient-free path.
        The position budget is reset to L at the start of each rollout chain.
        """
        n = len(self.positions_to_mutate)
        pw = np.ones(n, dtype=np.float64) / n
        mps = getattr(node, "mutations_per_sequence", float(self.mu * n))
        alpha = getattr(node, "exploration_alpha", float(self.exploration_alpha))
        return RolloutNodeWithProbs(
            seq=node.seq,
            fitness=node.fitness,
            edits_since_root=0,
            mutations_per_sequence=mps,
            exploration_alpha=alpha,
            position_weights=pw,
            gradient_position_weights=None,
        )

    # ── Path 3: gradient-guided (GradaBeam) ─────────────────────────────────

    def _propose_sequences_gradient(self, root_nodes: list) -> list:
        """Position-space rollout with TISM gradients and masking."""
        nodes_visited: set = set()
        all_rollout_lengths: list[int] = []
        gradient_node_cache: dict[str, RolloutNodeWithProbs] = {}

        root_nodes_effective = root_nodes * self.n_rollouts_per_root
        for i in range(0, len(root_nodes_effective), self.eval_batch_size):
            cur_root_nodes = root_nodes_effective[i : i + self.eval_batch_size]
            assert len(cur_root_nodes) == 1, (
                "AdaptiveRolloutDesigner gradient path expects eval_batch_size=1."
            )
            parent_seq = cur_root_nodes[0].seq

            if parent_seq in gradient_node_cache:
                parent_nodes = [gradient_node_cache[parent_seq]]
            else:
                parent_nodes = self.initialize_roots_with_gradients(cur_root_nodes)
                gradient_node_cache[parent_seq] = parent_nodes[0]

            cur_visited, cur_lengths = self._rollout(parent_nodes)
            nodes_visited.update(cur_visited)
            all_rollout_lengths.extend(cur_lengths)

        self.last_rollout_lengths = all_rollout_lengths

        if not nodes_visited:
            raise ValueError("No nodes generated.")

        sorted_nodes = sorted(nodes_visited, key=lambda x: x.sort_key, reverse=True)
        self.last_all_proposals = [
            {"seq": n.seq, "fitness": float(n.fitness)} for n in sorted_nodes
        ]
        return sorted_nodes[: self.beam_size]

    def _rollout(
        self,
        parent_nodes: list[RolloutNodeWithProbs],
    ) -> tuple[set[RolloutNodeWithProbs], list[int]]:
        """Run one rollout chain and return visited nodes and per-chain lengths.

        Rollout-length convention (Bug 3):
          rollout_length = number of mutations generated in the chain, where
          "generated" means the oracle was called for that child.

          Exhaustion is detected BEFORE generating the next mutation; at that
          point cur_rollout_length equals the number of mutations already made.
          Rejection is detected AFTER generating and incrementing; at that
          point cur_rollout_length also equals the number of mutations made
          (including the terminal rejected one).

          In both cases we record cur_rollout_length — the two code sites look
          different (exhaustion records before the increment, rejection records
          after) but always capture the same quantity: oracle calls in this chain.
        """
        nodes_visited: set = set()
        rollout_lengths: list[int] = []
        cur_rollout_length = 0

        while len(parent_nodes) > 0 and cur_rollout_length < self.max_rollout_len:
            # Exhaustion check BEFORE generating.
            # cur_rollout_length here = mutations already made = correct length.
            parent_nodes, exhausted = self._filter_exhausted(parent_nodes)
            rollout_lengths.extend([cur_rollout_length] * len(exhausted))
            if not parent_nodes:
                break

            num_edit_locs, new_rates = [], []
            for n in parent_nodes:
                n_edits, new_rate = self._get_next_mutation_params(n)
                num_edit_locs.append(n_edits)
                new_rates.append(new_rate)

            children = self._mutate_gradient_nodes(
                parent_nodes, num_edit_locs, new_rates
            )
            nodes_visited.update(children)
            cur_rollout_length += 1  # incremented AFTER generating

            new_nodes = []
            for child, cmp_node in zip(children, parent_nodes):
                if child.fitness >= cmp_node.fitness:
                    new_nodes.append(child)
                else:
                    # Rejection AFTER increment: cur_rollout_length = mutations
                    # made including this rejected one = correct length.
                    rollout_lengths.append(cur_rollout_length)
            parent_nodes = new_nodes

        return nodes_visited, rollout_lengths

    @staticmethod
    def _filter_exhausted(
        parent_nodes: list[RolloutNodeWithProbs],
    ) -> tuple[list[RolloutNodeWithProbs], list[RolloutNodeWithProbs]]:
        """Split nodes into (active, exhausted) by their available-position count.

        A node is exhausted when position_weights is all-zero (all mutable
        positions have been edited in the current rollout chain).  Exhausted
        nodes have their chain terminated; their rollout lengths are recorded
        by the caller.  The weight vector is never reset — termination is the
        correct behaviour when the position budget runs out.
        """
        active, exhausted = [], []
        for n in parent_nodes:
            if n.position_weights is None or int((n.position_weights > 0).sum()) >= 1:
                active.append(n)
            else:
                exhausted.append(n)
        return active, exhausted

    def _get_next_mutation_params(
        self, node: RolloutNodeWithProbs
    ) -> tuple[int, float]:
        current_rate = node.mutations_per_sequence
        n_edits = int(self.get_sampler(current_rate).sample(1)[0])
        if self.use_pbt:
            new_rate = float(np.clip(n_edits, 1.0, len(self.positions_to_mutate)))
        else:
            new_rate = current_rate
        return n_edits, new_rate

    def _mutate_gradient_nodes(
        self,
        nodes: list[RolloutNodeWithProbs],
        num_edit_locs: list[int],
        new_rates: list[float],
    ) -> list[RolloutNodeWithProbs]:
        assert (
            len(nodes) == len(num_edit_locs) == len(new_rates) <= self.eval_batch_size
        )

        seqs: list[str] = []
        new_pw_list: list[np.ndarray] = []
        child_alphas: list[float] = []
        effective_edits: list[int] = []

        for node, n_edits in zip(nodes, num_edit_locs):
            assert node.position_weights is not None, (
                "_mutate_gradient_nodes requires position_weights on node."
            )
            # gradient_position_weights may be None for the corrected gradient-free
            # path; GradientPositionStrategy will assert it is present if needed.

            result = self.strategy.propose_positions(
                node=node,
                n_edits=n_edits,
                rng=self.rng,
                mutable_positions=self.positions_to_mutate,
            )

            # GradientPositionStrategy returns a 3-tuple (positions, weights, p_final);
            # UniformPositionStrategy returns a 2-tuple.
            if isinstance(result, tuple) and len(result) == 3:
                chosen_positions, new_pw, p_final_chosen = result
            else:
                chosen_positions, new_pw = result
                p_final_chosen = None

            # Apply non-reference base mutations in position space
            all_bases = constants.VOCAB
            mutant = list(node.seq)
            for pos in chosen_positions:
                ref = node.seq[int(pos)]
                alts = [b for b in all_bases if b != ref]
                mutant[int(pos)] = str(self.rng.choice(alts))
            seq = "".join(mutant)

            seqs.append(seq)
            new_pw_list.append(new_pw)
            effective_edits.append(len(chosen_positions))

            # α-posterior update (Plan 01 §4 option a).
            # NOTE: this update runs PRE-FITNESS — α reflects which positions
            # were selected, not whether the selection improved fitness.  It is
            # a selection-based signal that tracks how much the gradient steered
            # the choice versus pure uniform sampling.  Plan 02b's gradient-gate
            # must account for this when interpreting the α trajectory.
            child_alpha = self._compute_child_alpha(
                node=node,
                chosen_positions=chosen_positions,
                p_final_chosen=p_final_chosen,
            )
            child_alphas.append(child_alpha)

        fitnesses = self.get_batched_fitness(seqs)

        return [
            RolloutNodeWithProbs(
                seq=seq,
                fitness=np.float32(float(f)),
                probs=node.probs,
                pos_and_chars=node.pos_and_chars,
                edits_since_root=(node.edits_since_root or 0) + n_eff,
                mutations_per_sequence=new_rate,
                exploration_alpha=child_alpha,
                position_weights=new_pw,
                gradient_position_weights=node.gradient_position_weights,
            )
            for seq, f, node, n_eff, new_rate, child_alpha, new_pw in zip(
                seqs,
                fitnesses,
                nodes,
                effective_edits,
                new_rates,
                child_alphas,
                new_pw_list,
            )
        ]

    def _compute_child_alpha(
        self,
        node: RolloutNodeWithProbs,
        chosen_positions: list[int],
        p_final_chosen: np.ndarray | None,
    ) -> float:
        """Compute the α-posterior for the child node (Plan 01 §4 option a).

        α is updated PRE-FITNESS based on which positions were selected.
        See the comment in _mutate_gradient_nodes for Plan 02b implications.

        When use_pbt=False, alpha passes through unchanged.

        Bug 1 guard: when gradient_position_weights is None (corrected
        gradient-free path), there is no gradient signal to compare against and
        the Bayesian update reduces to a no-op (posterior ≈ α everywhere).
        Rather than performing a meaningless update, we always pass through α
        unchanged on gradient-free nodes, regardless of use_pbt.

        Formula (gradient path only):
          p_uniform = 1 / n_available_positions   (NOT 1/(3L) or 1/L)
          P_final(j) = (1-α)·grad_w(j) + α·unif_w(j)  over available positions
          posterior_j = α · p_uniform / P_final(j)
          child_alpha  = clip(mean(posterior_j), 0.01, 0.99)
        """
        # Short-circuit: no gradient signal means no meaningful α update.
        # This covers both use_pbt=False (explicitly no update) and the corrected
        # gradient-free path (gradient_position_weights is None → inert posterior).
        if not self.use_pbt or node.gradient_position_weights is None:
            return float(node.exploration_alpha)

        assert node.position_weights is not None
        avail_mask = node.position_weights > 0
        n_available = int(avail_mask.sum())
        assert n_available >= 1, "No available positions for α update."

        p_uniform = 1.0 / n_available

        if p_final_chosen is None:
            # Should not be reached on the gradient path (GradientPositionStrategy
            # always returns a 3-tuple with p_final_chosen), but kept as a fallback.
            p_final_chosen_vals = np.full(len(chosen_positions), p_uniform)
        else:
            p_final_chosen_vals = p_final_chosen

        alpha = node.exploration_alpha
        posteriors = (alpha * p_uniform) / (p_final_chosen_vals + 1e-10)
        return float(np.clip(np.mean(posteriors), 0.01, 0.99))

    # ── TISM / gradient helpers ──────────────────────────────────────────────

    def initialize_roots_with_gradients(
        self, nodes: list[RolloutNodeWithProbs]
    ) -> list[RolloutNodeWithProbs]:
        """Compute TISM for each node; attach probs and position_weights.

        Root-step property: the position-marginal of the old action-level
        mixture (1-α)·grad_action + α·unif_action equals the new position-level
        mixture (1-α)·masked_grad + α·unif_w before any masking.  So the
        first-step position-selection distribution is unchanged from the old
        code; only the multi-step masking behavior differs.
        """
        n_positions = len(self.positions_to_mutate)
        grad_nodes = []

        for node in nodes:
            pos_and_chars, logits = self.model.get_tism(
                sequence=node.seq,
                idxs=self.tism_positions,
                debug=self.debug,
            )
            assert len(pos_and_chars) == 3 * n_positions, (
                f"Expected 3×{n_positions}={3 * n_positions} actions, "
                f"got {len(pos_and_chars)}."
            )
            assert len(pos_and_chars) == len(logits)

            # Mixed 3L probs (kept for backward compat with existing tests)
            mixed_probs = self.logits_to_probs(logits, node.exploration_alpha)

            # Pure-gradient position weights (for position-space masking)
            gradient_pos_weights = self._logits_to_gradient_position_weights(
                logits, n_positions
            )

            grad_nodes.append(
                RolloutNodeWithProbs(
                    seq=node.seq,
                    fitness=node.fitness,
                    edits_since_root=0,
                    probs=mixed_probs,
                    pos_and_chars=pos_and_chars,
                    mutations_per_sequence=node.mutations_per_sequence,
                    exploration_alpha=node.exploration_alpha,
                    position_weights=gradient_pos_weights.copy(),
                    gradient_position_weights=gradient_pos_weights,
                )
            )

        return grad_nodes

    def _logits_to_gradient_position_weights(
        self, logits: np.ndarray, n_positions: int
    ) -> np.ndarray:
        """Convert 3L TISM logits → normalized L position weights (pure gradient)."""
        std_dev = np.std(logits)
        if std_dev < 1e-9:
            return np.ones(n_positions, dtype=np.float64) / n_positions

        scaled = logits / std_dev
        dyn_temp = max(1.0, np.max(scaled) / self.max_logit)
        scaled = scaled / dyn_temp

        gradient_action_probs = softmax(scaled)
        gradient_action_probs = np.minimum(
            gradient_action_probs, self.gradient_prob_cap
        )
        gradient_action_probs /= gradient_action_probs.sum()

        pos_weights = ada_utils.tism_probs_to_position_weights(
            gradient_action_probs, n_positions
        )
        total = pos_weights.sum()
        if total > 0:
            pos_weights = pos_weights / total
        else:
            pos_weights = np.ones(n_positions, dtype=np.float64) / n_positions

        return pos_weights

    def logits_to_probs(self, logits: np.ndarray, alpha: float) -> np.ndarray:
        """Convert 3L logits to mixed (gradient + uniform) action probabilities."""
        std_dev = np.std(logits)
        if std_dev < 1e-9:
            return np.ones_like(logits) / len(logits)

        scaled = logits / std_dev
        dyn_temp = max(1.0, np.max(scaled) / self.max_logit)
        scaled = scaled / dyn_temp

        gradient_probs = softmax(scaled)
        gradient_probs = np.minimum(gradient_probs, self.gradient_prob_cap)
        gradient_probs /= gradient_probs.sum()

        n_actions = len(scaled)
        uniform_probs = np.ones(n_actions) / n_actions
        final_probs = (1.0 - alpha) * gradient_probs + alpha * uniform_probs
        return final_probs / final_probs.sum()

    def probabilities_over_actions_from_tism(
        self, nodes: list[RolloutNodeWithProbs]
    ) -> tuple[list[np.ndarray], list[PositionsAndCharactersType]]:
        """Return (probs_list, pos_and_chars_list) from TISM calls."""
        probs_list, pac_list = [], []
        for n in nodes:
            pos_and_chars, logits = self.model.get_tism(
                sequence=n.seq, idxs=self.tism_positions, debug=self.debug
            )
            probs_list.append(self.logits_to_probs(logits, n.exploration_alpha))
            pac_list.append(pos_and_chars)
        return probs_list, pac_list

    @staticmethod
    def debug_init_args() -> dict:
        return {
            "model_fn": testing_utils.CountLetterModel(),
            "start_sequence": "AAAAAA",
            "beam_size": 10,
            "mutations_per_sequence": 1,
            "n_rollouts_per_root": 4,
            "eval_batch_size": 1,
            "rng_seed": 42,
            "strategy": GradientPositionStrategy(),
            "use_gradients": True,
            "allow_silent_edits": False,
            "use_pbt": True,
            "exploration_alpha": 0.05,
        }
