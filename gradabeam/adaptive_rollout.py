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
# Extended rollout-node type
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class RolloutNodeWithProbs(ada_utils.RolloutNode):
    """Rollout node that carries gradient + action-space state.

    Field notes
    -----------
    probs : np.ndarray or None
        3L mixed action-probability vector.
    pos_and_chars : list[tuple[int, str]] or None
        (position, character) pairs of the actions.
    edits_since_root : int or None
        Depth in the current rollout chain, starting at 0 for roots.
    mutations_per_sequence : float
        Current per-step edit-rate target (mutated by PBT).
    exploration_alpha : float
        Current mixing coefficient — 0 = pure gradient, 1 = pure uniform.
    gradient_probs : np.ndarray or None
        3L vector of pure-gradient action probabilities from the most recent TISM
        call, before any masking or mixing. Used to recompute P_final for the α-update.
        None for the corrected gradient-free path.
    """

    probs: np.ndarray | None = field(default=None, hash=False, compare=False)
    pos_and_chars: PositionsAndCharactersType | None = field(
        default=None, hash=False, compare=False
    )
    edits_since_root: int | None = None
    mutations_per_sequence: float = dataclasses.field(
        default=1.0, compare=False, hash=False
    )
    exploration_alpha: float = dataclasses.field(default=0.5, compare=False, hash=False)
    gradient_probs: np.ndarray | None = field(default=None, hash=False, compare=False)

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
    """Uniform action-space strategy (corrected AdaBeam/gradients-off)."""


class GradientActionStrategy:
    """Gradient-guided action-space strategy (GrAdaBeam)."""


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
        self.use_gradients = use_gradients
        self.use_pbt = use_pbt
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
            self._init_beam_actionspace(
                start_sequence, beam_size, mutations_per_sequence
            )

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

    def _init_beam_actionspace(
        self,
        start_sequence: str,
        beam_size: int,
        mutations_per_sequence: float,
    ) -> None:
        """Corrected gradient-free initial beam (uniform action weights).

        Bug 2 (NaN fitness) analysis: seed_node carries fitness=np.float32(nan).
        This NaN is safe:
          * _mutate_gradient_nodes only reads node.seq, node.probs,
            node.exploration_alpha, and node.edits_since_root from the seed;
            children receive their fitness from get_batched_fitness(), not from
            the seed.
          * seed_node is never appended to current_nodes; only its children are.
          * No keep/reject comparison (child.fitness >= cmp_node.fitness) is
            performed against the seed_node.
        Therefore NaN cannot propagate to any comparison or tracker.
        """
        pos_and_chars = ada_utils.build_uniform_pos_and_chars(
            start_sequence, self.positions_to_mutate
        )
        n_actions = len(pos_and_chars)
        init_probs = np.ones(n_actions, dtype=np.float64) / n_actions
        seed_node = RolloutNodeWithProbs(
            seq=start_sequence,
            fitness=np.float32(
                np.nan
            ),  # safe: NaN is never read after children are made
            edits_since_root=0,
            mutations_per_sequence=float(mutations_per_sequence),
            exploration_alpha=float(self.exploration_alpha),
            probs=init_probs,
            pos_and_chars=pos_and_chars,
            gradient_probs=None,
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
        """Route to the correct operator path based on strategy and gradient flag.

        Routing table:
          no grads   → _propose_sequences_actionspace
          with grads → _propose_sequences_gradient
        """
        if not self.use_gradients:
            return self._propose_sequences_actionspace(root_nodes)
        else:
            return self._propose_sequences_gradient(root_nodes)

    # ── Path 2: corrected gradient-free (corrected AdaBeam) ─────────────────

    def _propose_sequences_actionspace(self, root_nodes: list) -> list:
        """Corrected gradient-free rollout using action-space operator.

        This is the scientific comparison point: "corrected AdaBeam"
        ≡ GradaBeam with gradients off + uniform weights.  No TISM is computed.
        Actions are selected uniformly from those whose positions have not yet
        been edited in the current rollout chain. Rollout chains terminate when all
        positions are exhausted.

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
            # Attach fresh uniform action-space probs; no TISM.
            parent_nodes = [self._attach_uniform_probs(n) for n in cur_root_nodes]

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
                if self.debug:
                    for _n_d, _child, _par in zip(
                        num_edit_locs, children, parent_nodes
                    ):
                        self._edit_count_log.append(
                            {
                                "n_drawn": int(_n_d),
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

    def _attach_uniform_probs(self, node: Any) -> RolloutNodeWithProbs:
        """Return a RolloutNodeWithProbs with fresh uniform action probabilities.

        Used to initialize each rollout in the corrected gradient-free path.
        The action space and position budget are reset to full at the start of each
        rollout chain.
        """
        pos_and_chars = ada_utils.build_uniform_pos_and_chars(
            node.seq, self.positions_to_mutate
        )
        n_actions = len(pos_and_chars)
        init_probs = np.ones(n_actions, dtype=np.float64) / n_actions
        mps = getattr(
            node,
            "mutations_per_sequence",
            float(self.mu * len(self.positions_to_mutate)),
        )
        alpha = getattr(node, "exploration_alpha", float(self.exploration_alpha))
        return RolloutNodeWithProbs(
            seq=node.seq,
            fitness=node.fitness,
            edits_since_root=0,
            mutations_per_sequence=mps,
            exploration_alpha=alpha,
            probs=init_probs,
            pos_and_chars=pos_and_chars,
            gradient_probs=None,
        )

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
        current rollout chain (all action probabilities are zero).
        """
        active, exhausted = [], []
        for n in parent_nodes:
            if n.probs is None or int((n.probs > 0).sum()) >= 1:
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
            # Cap strictly below L so mu = new_rate/L < 1, preventing _F_inverse
            # blow-up.  L-1 is exact (no float fudge) and L >= 2 is guaranteed by
            # the construction assert (mutations_per_sequence < L, and
            # mutations_per_sequence >= 1 implies L >= 2).
            _max_rate = len(self.positions_to_mutate) - 1
            new_rate = float(np.clip(n_edits, 1.0, _max_rate))
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

        for node, n_edits in zip(nodes, num_edit_locs):
            assert node.probs is not None, (
                "_mutate_gradient_nodes requires probs on node."
            )
            assert node.pos_and_chars is not None, (
                "_mutate_gradient_nodes requires pos_and_chars on node."
            )

            # Sequential action-space mutation with position-level masking
            mutant_seq, selected_idx, masked_probs, p_final_chosen_list = (
                ada_utils.generate_random_mutant_actionspace(
                    sequence=node.seq,
                    pos_and_chars_to_mutate=node.pos_and_chars,
                    n_edits=n_edits,
                    rng=self.rng,
                    probs=node.probs,
                )
            )

            seqs.append(mutant_seq)
            new_probs_list.append(masked_probs)
            effective_edits.append(len(selected_idx))

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
            )
            for seq, f, node, n_eff, new_rate, child_alpha, new_probs in zip(
                seqs,
                fitnesses,
                nodes,
                effective_edits,
                new_rates,
                child_alphas,
                new_probs_list,
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
        n_avail = int((node.probs > 0).sum())
        assert n_avail >= 1, "No available actions for α update."

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
