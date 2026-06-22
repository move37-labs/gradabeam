"""AdaptiveRolloutDesigner — unified designer with injected mutation strategy.

Root-step equivalence property
-------------------------------
At the first step (no positions yet masked) the position-marginal of the
action-level mixture  (1-α)·grad_action + α·unif_action  is exactly:

    w_i = (1-α)·Σ_a grad_{i,a} + α·(3/3L) = (1-α)·grad_pos_i + α·(1/L)

because summing the action mixture over the 3 bases at position i is linear.
The base is then drawn from the CONDITIONAL distribution of `probs` at the chosen
position — NOT uniformly.  Root-step position-selection behavior is therefore
identical to drawing positions by the marginal weight; the only intended
divergence from the old code is the (previously buggy) multi-step positional
masking, which now correctly zeros all 3 actions of an edited position.

Strategy representations
------------------------
UniformActionStrategy  — position-space (L-vector).
    Carries np.ones(L)/L position weights; draws position uniformly, then base
    uniformly from the 3 non-reference bases.  No 3L tuple construction.
    Distribution-identical to the old action-space uniform path.

GradientActionStrategy — action-space (3L-vector).
    Carries TISM-derived mixed action probabilities.  The gradient picks position
    AND base jointly, which requires the joint 3L action space.
    Implementation lives on AdaptiveRolloutDesigner (TISM init, _mutate_gradient_nodes,
    _rollout, gradient cache); this strategy is a thin delegation wrapper that
    gives both strategies the same interface without relocating gradient logic.
"""

from __future__ import annotations

import collections
import dataclasses
from dataclasses import field
from functools import lru_cache
from typing import Any

import numpy as np
from scipy.special import softmax

from gradabeam import ada_utils
from gradabeam import constants
from gradabeam import testing_utils


PositionsAndCharactersType = ada_utils.PositionsAndCharactersType


# ---------------------------------------------------------------------------
# Proposal result — returned by strategy.propose()
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class ProposalResult:
    """Output of strategy.propose() for a single node.

    Universal fields (both strategies):
        mutant_seq, n_edits_effective, n_positions_edited, child_alpha

    Strategy-specific state (each strategy populates its fields and leaves
    the other strategy's fields as None).  This is an inherent tradeoff of
    two representations (L-vector vs 3L) sharing one node type:
        probs, pos_and_chars, gradient_probs — used by GradientActionStrategy
        position_weights                     — used by UniformActionStrategy
    """

    mutant_seq: str
    n_edits_effective: int
    n_positions_edited: int
    child_alpha: float
    # GradientActionStrategy state (None for UniformActionStrategy):
    probs: np.ndarray | None
    pos_and_chars: PositionsAndCharactersType | None
    gradient_probs: np.ndarray | None
    # UniformActionStrategy state (None for GradientActionStrategy):
    position_weights: np.ndarray | None


# ---------------------------------------------------------------------------
# Extended rollout-node type
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class RolloutNodeWithProbs(ada_utils.RolloutNode):
    """Rollout node that carries mutation strategy state.

    Field notes
    -----------
    probs : np.ndarray or None
        GradientActionStrategy: 3L mixed action-probability vector.
        UniformActionStrategy: None (uses position_weights instead).
    pos_and_chars : list[tuple[int, str]] or None
        GradientActionStrategy: (position, character) pairs of the actions (3L).
        UniformActionStrategy: None.
    position_weights : np.ndarray or None
        UniformActionStrategy: L-vector of position weights (uniform initially,
        zeroed at edited positions across rollout steps).
        GradientActionStrategy: None (uses probs instead).
    edits_since_root : int or None
        Depth in the current rollout chain, starting at 0 for roots.
    mutations_per_sequence : float
        Current per-step edit-rate target (mutated by PBT).
    exploration_alpha : float
        Current mixing coefficient — 0 = pure gradient, 1 = pure uniform.
    gradient_probs : np.ndarray or None
        GradientActionStrategy: 3L vector of pure-gradient action probabilities
        from the most recent TISM call, before any masking or mixing.
        None for the uniform path.
    n_positions_remaining : int
        Number of mutable positions still available (not yet masked in this chain).
        O(1) counter used by _filter_exhausted and _compute_child_alpha.
        Invariant (gradient path):  n_positions_remaining * 3 == (probs > 0).sum()
        Invariant (uniform path):   n_positions_remaining == (position_weights > 0).sum()
        Set by attach_initial_state; never None after initialization.
    """

    probs: np.ndarray | None = field(default=None, hash=False, compare=False)
    pos_and_chars: PositionsAndCharactersType | None = field(
        default=None, hash=False, compare=False
    )
    position_weights: np.ndarray | None = field(default=None, hash=False, compare=False)
    edits_since_root: int | None = None
    mutations_per_sequence: float = dataclasses.field(
        default=1.0, compare=False, hash=False
    )
    exploration_alpha: float = dataclasses.field(default=0.5, compare=False, hash=False)
    gradient_probs: np.ndarray | None = field(default=None, hash=False, compare=False)
    n_positions_remaining: int | None = dataclasses.field(
        default=None, compare=False, hash=False
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


class UniformActionStrategy:
    """Position-space (L-vector) strategy for AdaBeam (gradient-free).

    Representation: np.ones(L)/L position-weight vector.  No 3L tuple
    construction, no 3L array operations.  The base at each chosen position
    is drawn uniformly from the 3 non-reference VOCAB bases, which is
    distribution-identical to the old action-space uniform path:
        uniform over 3L actions == uniform position × uniform non-ref base.

    attach_initial_state  — creates the L-vector; costs O(L), not O(3L).
    propose               — calls generate_random_mutant_positionspace.
    get_edit_params       — draws n_edits from the designer's fixed sampler;
                            rate is unchanged (no PBT on this path).
    """

    def attach_initial_state(
        self,
        node: "RolloutNodeWithProbs",
        designer: "AdaptiveRolloutDesigner",
    ) -> "RolloutNodeWithProbs":
        """Return a fresh node with uniform L-vector position weights."""
        L = len(designer.positions_to_mutate)
        mps = getattr(node, "mutations_per_sequence", float(designer.mu * L))
        alpha = getattr(node, "exploration_alpha", float(designer.exploration_alpha))
        return RolloutNodeWithProbs(
            seq=node.seq,
            fitness=node.fitness,
            edits_since_root=0,
            mutations_per_sequence=mps,
            exploration_alpha=alpha,
            probs=None,
            pos_and_chars=None,
            position_weights=np.ones(L, dtype=np.float64) / L,
            gradient_probs=None,
            n_positions_remaining=L,
        )

    def propose(
        self,
        node: "RolloutNodeWithProbs",
        rng: np.random.Generator,
        n_edits: int,
        positions_to_mutate: list[int],
    ) -> "ProposalResult":
        """Mutate node using position-space sampling; return ProposalResult."""
        assert node.position_weights is not None
        mutant_seq, _, masked_weights, n_positions_edited = (
            ada_utils.generate_random_mutant_positionspace(
                sequence=node.seq,
                positions_to_mutate=positions_to_mutate,
                position_weights=node.position_weights,
                n_edits=n_edits,
                rng=rng,
            )
        )
        return ProposalResult(
            mutant_seq=mutant_seq,
            n_edits_effective=n_positions_edited,
            n_positions_edited=n_positions_edited,
            child_alpha=float(node.exploration_alpha),  # unchanged on uniform path
            probs=None,
            pos_and_chars=None,
            gradient_probs=None,
            position_weights=masked_weights,
        )

    def get_edit_params(
        self,
        node: "RolloutNodeWithProbs",
        designer: "AdaptiveRolloutDesigner",
    ) -> tuple[int, float]:
        """Draw n_edits from the fixed sampler; rate is unchanged."""
        n_edits = int(designer.num_mutations_sampler.sample(1)[0])
        return n_edits, float(node.mutations_per_sequence)


class GradientActionStrategy:
    """Action-space (3L-vector) strategy for GrAdaBeam (gradient-guided).

    Representation: TISM-derived mixed action-probability vector of length 3L.
    The gradient picks position AND base jointly, which requires the joint 3L
    action space.

    This class is a thin delegation wrapper: all implementation stays on
    AdaptiveRolloutDesigner (TISM init, gradient cache, _mutate_gradient_nodes,
    _rollout, _compute_child_alpha).  The strategy methods call into those
    existing designer methods without relocating any gradient logic.

    attach_initial_state  — delegates to designer.initialize_roots_with_gradients.
    propose               — delegates to ada_utils.generate_random_mutant_actionspace
                            + designer._compute_child_alpha.
    get_edit_params       — delegates to designer._get_next_mutation_params.
    """

    def attach_initial_state(
        self,
        node: "RolloutNodeWithProbs",
        designer: "AdaptiveRolloutDesigner",
    ) -> "RolloutNodeWithProbs":
        """Compute TISM and attach mixed 3L probs; delegates to designer."""
        return designer.initialize_roots_with_gradients([node])[0]

    def propose(
        self,
        node: "RolloutNodeWithProbs",
        rng: np.random.Generator,
        n_edits: int,
        positions_to_mutate: list[int],
    ) -> "ProposalResult":
        """Mutate node using action-space sampling; delegates to designer helpers."""
        assert node.probs is not None
        assert node.pos_and_chars is not None
        (
            mutant_seq,
            _selected_idx,
            masked_probs,
            p_final_chosen_list,
            n_positions_edited,
        ) = ada_utils.generate_random_mutant_actionspace(
            sequence=node.seq,
            pos_and_chars_to_mutate=node.pos_and_chars,
            n_edits=n_edits,
            rng=rng,
            probs=node.probs,
        )
        # _compute_child_alpha lives on the designer (tests call it directly).
        # We need a reference; it will be set by AdaptiveRolloutDesigner.__init__.
        child_alpha = self._designer._compute_child_alpha(
            node=node,
            p_final_chosen_list=p_final_chosen_list,
        )
        return ProposalResult(
            mutant_seq=mutant_seq,
            n_edits_effective=n_positions_edited,
            n_positions_edited=n_positions_edited,
            child_alpha=child_alpha,
            probs=masked_probs,
            pos_and_chars=node.pos_and_chars,
            gradient_probs=node.gradient_probs,
            position_weights=None,
        )

    def get_edit_params(
        self,
        node: "RolloutNodeWithProbs",
        designer: "AdaptiveRolloutDesigner",
    ) -> tuple[int, float]:
        """Draw n_edits and PBT-adapted rate; delegates to designer."""
        return designer._get_next_mutation_params(node)

    def _set_designer(self, designer: "AdaptiveRolloutDesigner") -> None:
        """Called once by AdaptiveRolloutDesigner.__init__ to give back-reference."""
        self._designer = designer


# ---------------------------------------------------------------------------
# AdaptiveRolloutDesigner — unified optimizer
# ---------------------------------------------------------------------------


class AdaptiveRolloutDesigner:
    """Unified beam-search sequence designer.

    Two operator/gradient paths (routed by propose_sequences):

    +----------------+------------------------------+
    | use_gradients  | path                         |
    +----------------+------------------------------+
    | False          | corrected gradient-free      |
    | True           | gradient-guided (GradaBeam)  |
    +----------------+------------------------------+

    Parameters
    ----------
    strategy : UniformActionStrategy | GradientActionStrategy
        Controls how candidate positions are selected at each rollout step.
    use_gradients : bool
        When True, compute TISM at each rollout root.
        When False, skip TISM (no tism_cost charged to ModelWrapper).
    use_pbt : bool
        Enable Population Based Training for adaptive mutation rate and α.
    exploration_alpha : float
        Initial mixing coefficient (0=pure gradient, 1=pure uniform).
    """

    def __init__(
        self,
        model_fn: Any,
        start_sequence: str,
        mutations_per_sequence: float,
        beam_size: int,
        n_rollouts_per_root: int,
        strategy: UniformActionStrategy | GradientActionStrategy,
        use_gradients: bool,
        use_pbt: bool,
        pbt_rate_rule: str = "snap",
        exploration_alpha: float = 0.5,
        gradient_prob_cap: float = 0.10,
        max_logit: float = 3.0,
        rng_seed: int = 0,
        positions_to_mutate: list[int] | None = None,
        eval_batch_size: int = 1,
        max_rollout_len: int = 200,
        debug: bool = False,
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
        assert mutations_per_sequence < len(self.positions_to_mutate), (
            f"mutations_per_sequence ({mutations_per_sequence}) must be < "
            f"len(positions_to_mutate) ({len(self.positions_to_mutate)}) so mu < 1"
        )
        assert beam_size > 0
        assert n_rollouts_per_root > 0
        if use_gradients:
            assert 0.0 <= exploration_alpha <= 1.0

        # Strategy / gradient combination validation
        if isinstance(strategy, GradientActionStrategy) and not use_gradients:
            raise ValueError("GradientActionStrategy requires use_gradients=True.")

        self.strategy = strategy
        # Give GradientActionStrategy a back-reference so its propose() can call
        # designer methods (_compute_child_alpha) without relocating that logic.
        if isinstance(strategy, GradientActionStrategy):
            strategy._set_designer(self)
        if pbt_rate_rule not in ("snap", "perturb"):
            raise ValueError(
                f"pbt_rate_rule must be 'snap' or 'perturb', got {pbt_rate_rule!r}"
            )
        self.use_gradients = use_gradients
        self.use_pbt = use_pbt
        self.pbt_rate_rule = pbt_rate_rule
        self.exploration_alpha = exploration_alpha
        self.gradient_prob_cap = gradient_prob_cap
        self.max_logit = max_logit

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
        self._edit_count_log: list[dict] = []

        # ── sampler setup ────────────────────────────────────────────────────
        # The corrected gradient-free path uses a single fixed-rate sampler
        # (matches AdaBeam's sampler structure).  The gradient path uses
        # the PBT-per-node get_sampler() / _get_sampler_cached() instead.
        if not use_gradients:
            self.num_mutations_sampler: ada_utils.NumberEditsSampler = (
                ada_utils.NumberEditsSamplerAdaBeam(
                    sequence_len=len(self.positions_to_mutate),
                    mutation_rate=self.mu,
                    rng_seed=rng_seed,
                )
            )

        # Filled by propose_sequences; read by tests.
        self.last_all_proposals: list[dict] = []

        # ── initial beam ─────────────────────────────────────────────────────
        if use_gradients:
            self._init_beam_gradient(start_sequence, beam_size, mutations_per_sequence)
        else:
            self._init_beam_uniform(start_sequence, beam_size, mutations_per_sequence)

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
            gradient_probs=None,
            n_positions_remaining=None,
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

    def _init_beam_uniform(
        self,
        start_sequence: str,
        beam_size: int,
        mutations_per_sequence: float,
    ) -> None:
        """Gradient-free initial beam using the uniform position-space strategy.

        Bug 2 (NaN fitness) analysis: seed_node carries fitness=np.float32(nan).
        This NaN is safe:
          * attach_initial_state reads node.seq, node.mutations_per_sequence, and
            node.exploration_alpha from the seed; children receive their fitness
            from get_batched_fitness(), not from the seed.
          * seed_node is never appended to current_nodes; only its children are.
          * No keep/reject comparison is performed against the seed_node.
        Therefore NaN cannot propagate to any comparison or tracker.
        """
        seed_node = RolloutNodeWithProbs(
            seq=start_sequence,
            fitness=np.float32(
                np.nan
            ),  # safe: NaN is never read after children are made
            edits_since_root=0,
            mutations_per_sequence=float(mutations_per_sequence),
            exploration_alpha=float(self.exploration_alpha),
            probs=None,
            pos_and_chars=None,
            position_weights=None,  # attach_initial_state will set this
            gradient_probs=None,
            n_positions_remaining=None,  # set by attach_initial_state
        )
        # Initialize: attach position-space state to the seed.
        initialized_seed = self.strategy.attach_initial_state(seed_node, self)
        num_edit_locs = [int(x) for x in self.num_mutations_sampler.sample(beam_size)]

        seqs: list[str] = []
        child_state_list: list[ProposalResult] = []
        for i in range(0, beam_size, self.eval_batch_size):
            cur_edits = num_edit_locs[i : i + self.eval_batch_size]
            for n_edits in cur_edits:
                proposal = self.strategy.propose(
                    initialized_seed,
                    self.rng,
                    n_edits,
                    self.positions_to_mutate,
                )
                seqs.append(proposal.mutant_seq)
                child_state_list.append(proposal)

        fitnesses = self.get_batched_fitness(seqs)
        self.current_nodes = [
            RolloutNodeWithProbs(
                seq=p.mutant_seq,
                fitness=np.float32(float(f)),
                probs=p.probs,
                pos_and_chars=p.pos_and_chars,
                position_weights=p.position_weights,
                edits_since_root=p.n_edits_effective,
                mutations_per_sequence=float(mutations_per_sequence),
                exploration_alpha=p.child_alpha,
                gradient_probs=p.gradient_probs,
                n_positions_remaining=(
                    len(self.positions_to_mutate) - p.n_positions_edited
                ),
            )
            for p, f in zip(child_state_list, fitnesses)
        ]

    # ── public API ───────────────────────────────────────────────────────────

    def run(self, n_steps: int) -> None:
        if self.debug:
            self._edit_count_log = []
        for _step in range(n_steps):
            self.current_nodes = self.propose_sequences(self.current_nodes)
            if self.debug and self.current_nodes:
                print(f"Step {_step} top score: {self.current_nodes[0].fitness}")
        if self.debug and self._edit_count_log:
            self._print_edit_count_report()

    def _print_edit_count_report(self) -> None:
        """Histogram of per-step N_drawn and N_changed; called at end of run() when debug=True.

        Logs every proposed child (pre-acceptance) to reveal whether N is drawn
        fresh each step or pinned.  No new RNG draws — reads only values already
        computed in the rollout loops.
        """
        n_drawn_arr = np.array([d["n_drawn"] for d in self._edit_count_log])
        n_changed_arr = np.array([d["n_changed"] for d in self._edit_count_log])

        L = len(self.positions_to_mutate)
        mps = self.mu * L
        theoretical_mean = self.num_mutations_sampler.expected_num_edits()

        def _hist(counts: dict, max_width: int = 50) -> None:
            if not counts:
                return
            max_count = max(counts.values())
            scale = max_count / max_width if max_count > max_width else 1
            for k in sorted(counts):
                bar = "#" * int(counts[k] / scale)
                print(f"  {k:4d} | {bar:<{max_width}} {counts[k]}")

        print()
        print("=" * 62)
        print("Edit-count log  (every proposed child, pre-acceptance)")
        print("=" * 62)
        print(f"  L (positions_to_mutate)    : {L}")
        print(f"  mutations_per_sequence     : {mps:.4f}")
        print(f"  mu = mps / L               : {self.mu:.6f}")
        print(f"  TruncBinom(L, mu) mean     : {theoretical_mean:.4f}")
        print()
        freq_drawn = collections.Counter(n_drawn_arr.tolist())
        print(f"N_drawn  (n={len(n_drawn_arr):,})")
        print("-" * 40)
        _hist(freq_drawn)
        print()
        print(f"  min    : {n_drawn_arr.min()}")
        print(f"  max    : {n_drawn_arr.max()}")
        print(f"  mean   : {n_drawn_arr.mean():.4f}")
        print(f"  std    : {n_drawn_arr.std():.4f}")
        print()
        freq_changed = collections.Counter(n_changed_arr.tolist())
        print(f"N_changed  (n={len(n_changed_arr):,})")
        print("-" * 40)
        _hist(freq_changed)
        print()
        print(f"  mean   : {n_changed_arr.mean():.4f}")
        print("=" * 62)

    def get_samples(self, n_samples: int) -> list[str]:
        sorted_nodes = sorted(
            self.current_nodes, key=lambda x: (x.fitness, x.seq), reverse=True
        )
        return [x.seq for x in sorted_nodes[:n_samples]]

    def get_batched_fitness(self, sequences: list[str]) -> np.ndarray:
        return ada_utils.get_batched_fitness(
            model_wrapper=self.model,
            sequences=sequences,
            batch_size=self.eval_batch_size,
        )

    def propose_sequences(self, root_nodes: list) -> list:
        """Route to the correct rollout path based on strategy type.

        Routing table:
          UniformActionStrategy  → _propose_sequences_uniform
          GradientActionStrategy → _propose_sequences_gradient

        The isinstance dispatch lives here at the entry point only; neither
        inner rollout loop contains space/isinstance branching.
        """
        if isinstance(self.strategy, GradientActionStrategy):
            return self._propose_sequences_gradient(root_nodes)
        else:
            return self._propose_sequences_uniform(root_nodes)

    # ── Path 2: uniform position-space (AdaBeam) ────────────────────────────

    def _propose_sequences_uniform(self, root_nodes: list) -> list:
        """Position-space rollout using UniformActionStrategy.

        This is the corrected AdaBeam path.  No TISM is computed.  Positions are
        selected uniformly via the strategy's L-vector; bases are drawn uniformly
        from the 3 non-reference bases at each chosen position.  Rollout chains
        terminate when all mutable positions have been edited (exhaustion) or a
        child is rejected.

        Rollout-length convention: same as _rollout.  Exhaustion (recorded before
        the increment) and rejection (recorded after) both capture
        cur_rollout_length = number of oracle calls made in the chain.
        See _rollout docstring for the full explanation.

        Sets self.last_rollout_lengths for testability.
        """
        nodes_visited: set = set()
        all_rollout_lengths: list[int] = []

        root_nodes_effective = root_nodes * self.n_rollouts_per_root
        for i in range(0, len(root_nodes_effective), self.eval_batch_size):
            cur_root_nodes = root_nodes_effective[i : i + self.eval_batch_size]
            # Attach fresh L-vector position weights; no TISM, no 3L tuple construction.
            parent_nodes = [
                self.strategy.attach_initial_state(n, self) for n in cur_root_nodes
            ]

            cur_rollout_length = 0
            while len(parent_nodes) > 0 and cur_rollout_length < self.max_rollout_len:
                # Exhaustion check BEFORE generating; cur_rollout_length = mutations made.
                parent_nodes, exhausted = self._filter_exhausted(parent_nodes)
                all_rollout_lengths.extend([cur_rollout_length] * len(exhausted))
                if not parent_nodes:
                    break

                proposals: list[ProposalResult] = []
                for node in parent_nodes:
                    n_edits, new_rate = self.strategy.get_edit_params(node, self)
                    proposal = self.strategy.propose(
                        node, self.rng, n_edits, self.positions_to_mutate
                    )
                    proposals.append(proposal)

                seqs = [p.mutant_seq for p in proposals]
                fitnesses = self.get_batched_fitness(seqs)

                children = [
                    RolloutNodeWithProbs(
                        seq=p.mutant_seq,
                        fitness=np.float32(float(f)),
                        probs=p.probs,
                        pos_and_chars=p.pos_and_chars,
                        position_weights=p.position_weights,
                        edits_since_root=(node.edits_since_root or 0)
                        + p.n_edits_effective,
                        mutations_per_sequence=node.mutations_per_sequence,
                        exploration_alpha=p.child_alpha,
                        gradient_probs=p.gradient_probs,
                        n_positions_remaining=(
                            (
                                node.n_positions_remaining
                                or len(self.positions_to_mutate)
                            )
                            - p.n_positions_edited
                        ),
                    )
                    for p, f, node in zip(proposals, fitnesses, parent_nodes)
                ]

                if self.debug:
                    for _child, _par in zip(children, parent_nodes):
                        self._edit_count_log.append(
                            {
                                "n_drawn": (_child.edits_since_root or 0)
                                - (_par.edits_since_root or 0),
                                "n_changed": sum(
                                    a != b for a, b in zip(_child.seq, _par.seq)
                                ),
                            }
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

    # ── Path 3: gradient-guided (GradaBeam) ─────────────────────────────────

    def _propose_sequences_gradient(self, root_nodes: list) -> list:
        """Action-space rollout with TISM gradient-guided sampling and position-level masking."""
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

        A node is exhausted when all mutable positions have been edited in the
        current rollout chain (n_positions_remaining == 0).  The counter is O(1)
        to read; it is decremented by the number of distinct positions edited each
        step (for both L-vector and 3L representations).

        Invariant (verified pre-refactor): every node reaching this method has
        n_positions_remaining set (never None) — the None case from gradient seed
        nodes is transient within initialize_roots_with_gradients and is resolved
        before any node enters a rollout loop.

        The debug assertion validates the counter against whichever representation
        the node carries:
          GradientActionStrategy: n_positions_remaining * 3 == (probs > 0).sum()
          UniformActionStrategy:  n_positions_remaining == (position_weights > 0).sum()
        """
        active, exhausted = [], []
        for n in parent_nodes:
            assert n.n_positions_remaining is not None, (
                "n_positions_remaining must be set before _filter_exhausted is called. "
                f"Node: seq={n.seq!r}, edits_since_root={n.edits_since_root}"
            )
            if n.probs is not None:
                assert n.n_positions_remaining * 3 == int((n.probs > 0).sum()), (
                    f"n_positions_remaining invariant broken (gradient path): "
                    f"n_positions_remaining={n.n_positions_remaining}, "
                    f"(probs > 0).sum()={int((n.probs > 0).sum())}"
                )
            elif n.position_weights is not None:
                assert n.n_positions_remaining == int((n.position_weights > 0).sum()), (
                    f"n_positions_remaining invariant broken (uniform path): "
                    f"n_positions_remaining={n.n_positions_remaining}, "
                    f"(position_weights > 0).sum()={int((n.position_weights > 0).sum())}"
                )
            if n.n_positions_remaining >= 1:
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
            # max(1.0, L-1): construction asserts L >= 2, so this always equals L-1;
            # the max guards the L<=2 edge should that assertion ever relax.
            _max_rate = max(1.0, len(self.positions_to_mutate) - 1)
            if self.pbt_rate_rule == "snap":
                # Current shipped behavior: child rate snaps to the sampled edit count.
                new_rate = float(np.clip(n_edits, 1.0, _max_rate))
            else:  # "perturb" — paper §4.3.1
                # Perturb the INHERITED rate multiplicatively. RNG is consumed ONLY
                # here so the snap path's stream is bit-for-bit identical to HEAD.
                # mutations_per_sequence is the expected edit count (rate);
                # mu = rate / L. The ×{0.8, 1.2} scales rate, equivalently µ.
                p_perturb = 0.20
                r = self.rng.random()
                if r < p_perturb / 2:
                    new_rate = 0.8 * current_rate
                elif r < p_perturb:
                    new_rate = 1.2 * current_rate
                else:
                    new_rate = current_rate
                new_rate = float(np.clip(new_rate, 1.0, _max_rate))
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
        new_probs_list: list[np.ndarray] = []
        child_alphas: list[float] = []
        effective_edits: list[int] = []
        n_positions_edited_list: list[int] = []

        for node, n_edits in zip(nodes, num_edit_locs):
            assert node.probs is not None, (
                "_mutate_gradient_nodes requires probs on node."
            )
            assert node.pos_and_chars is not None, (
                "_mutate_gradient_nodes requires pos_and_chars on node."
            )

            # Sequential action-space mutation with position-level masking
            (
                mutant_seq,
                selected_idx,
                masked_probs,
                p_final_chosen_list,
                n_positions_edited,
            ) = ada_utils.generate_random_mutant_actionspace(
                sequence=node.seq,
                pos_and_chars_to_mutate=node.pos_and_chars,
                n_edits=n_edits,
                rng=self.rng,
                probs=node.probs,
            )

            seqs.append(mutant_seq)
            new_probs_list.append(masked_probs)
            effective_edits.append(len(selected_idx))
            n_positions_edited_list.append(n_positions_edited)

            # α-posterior update (measures joint surprise over action space)
            child_alpha = self._compute_child_alpha(
                node=node,
                p_final_chosen_list=p_final_chosen_list,
            )
            child_alphas.append(child_alpha)

        fitnesses = self.get_batched_fitness(seqs)

        return [
            RolloutNodeWithProbs(
                seq=seq,
                fitness=np.float32(float(f)),
                probs=new_probs,
                pos_and_chars=node.pos_and_chars,
                edits_since_root=(node.edits_since_root or 0) + n_eff,
                mutations_per_sequence=new_rate,
                exploration_alpha=child_alpha,
                gradient_probs=node.gradient_probs,
                n_positions_remaining=(
                    node.n_positions_remaining or len(self.positions_to_mutate)
                )
                - n_pos_edited,
            )
            for seq, f, node, n_eff, new_rate, child_alpha, new_probs, n_pos_edited in zip(
                seqs,
                fitnesses,
                nodes,
                effective_edits,
                new_rates,
                child_alphas,
                new_probs_list,
                n_positions_edited_list,
            )
        ]

    def _compute_child_alpha(
        self,
        node: RolloutNodeWithProbs,
        p_final_chosen_list: list[float],
    ) -> float:
        """Compute the α-posterior for the child node in action space.

        The posterior measures joint (position AND base) surprise.  p_uniform is
        1/n_avail over the step-start available actions, matching the reading-B
        p_final = probs[action] convention (see generate_random_mutant_actionspace).

        When use_pbt=False or p_final_chosen_list is empty, alpha passes through
        unchanged.  gradient_probs is used only as a path gate (None == gradient-free
        path), not to compute p_final; p_final comes from the sampler.
        """
        if not self.use_pbt or node.gradient_probs is None or not p_final_chosen_list:
            return float(node.exploration_alpha)

        assert node.probs is not None
        assert (
            node.n_positions_remaining is not None and node.n_positions_remaining >= 1
        ), (
            f"n_positions_remaining must be a positive integer here, "
            f"got {node.n_positions_remaining!r}"
        )
        n_avail = node.n_positions_remaining * 3  # actions = 3 per position

        p_uniform = 1.0 / n_avail
        alpha = node.exploration_alpha

        posteriors = []
        for p_final in p_final_chosen_list:
            posteriors.append((alpha * p_uniform) / (p_final + 1e-10))

        return float(np.clip(np.mean(posteriors), 0.01, 0.99))

    # ── TISM / gradient helpers ──────────────────────────────────────────────

    def initialize_roots_with_gradients(
        self, nodes: list[RolloutNodeWithProbs]
    ) -> list[RolloutNodeWithProbs]:
        """Compute TISM for each node; attach probs and gradient_probs."""
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

            # Pure-gradient action probabilities
            gradient_probs = self._logits_to_gradient_probs(logits)

            # Mixed 3L probs (for action selection)
            mixed_probs = self.mix_gradient_with_uniform(
                gradient_probs, node.exploration_alpha
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
                    gradient_probs=gradient_probs,
                    n_positions_remaining=len(self.positions_to_mutate),
                )
            )

        return grad_nodes

    def _logits_to_gradient_probs(self, logits: np.ndarray) -> np.ndarray:
        """Convert TISM logits -> normalized capped 3L pure-gradient action probabilities.

        Cap granularity: per-ACTION at self.gradient_prob_cap (default 0.10).
        The paper describes a per-position cap, but the prerefactor implementation
        on main caps per-action.  We follow main.  With 3 actions per position the
        per-action cap is up to 3x looser per site (a position can hold up to 0.30
        of total mass).

        Re-cap policy: the cap is applied ONCE here at root level.  Across-step
        masking (in generate_random_mutant_actionspace) renormalizes survivors and
        may push individual actions above the cap.  We do NOT re-cap — the cap is
        root-level smoothing, not a maintained invariant.  This matches the
        prerefactor behavior on main.
        """
        std_dev = np.std(logits)
        if std_dev < 1e-9:
            return np.ones_like(logits) / len(logits)

        scaled = logits / std_dev
        dyn_temp = max(1.0, np.max(scaled) / self.max_logit)
        scaled = scaled / dyn_temp

        gradient_probs = softmax(scaled)
        gradient_probs = np.minimum(gradient_probs, self.gradient_prob_cap)
        total = gradient_probs.sum()
        if total > 0:
            gradient_probs /= total
        else:
            gradient_probs = np.ones_like(gradient_probs) / len(gradient_probs)
        return gradient_probs

    def mix_gradient_with_uniform(
        self, gradient_probs: np.ndarray, alpha: float
    ) -> np.ndarray:
        n_actions = len(gradient_probs)
        uniform_probs = np.ones(n_actions) / n_actions
        final_probs = (1.0 - alpha) * gradient_probs + alpha * uniform_probs
        return final_probs / final_probs.sum()

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
            "strategy": GradientActionStrategy(),
            "use_gradients": True,
            "use_pbt": True,
            "exploration_alpha": 0.5,
        }
