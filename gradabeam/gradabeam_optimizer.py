"""GradaBeamPBT.

Gradient-guided adaptive beam, adaptive mutation rate (PBT), adaptive directed evolution.
"""

import dataclasses
from dataclasses import field
from typing import Any

import numpy as np
from scipy.special import softmax

from gradabeam import ada_utils, beam_common, constants, testing_utils

PositionsAndCharactersType = ada_utils.PositionsAndCharactersType
TismActionData = tuple[PositionsAndCharactersType, np.ndarray]


@dataclasses.dataclass(frozen=True)
class RolloutNodeWithProbs(ada_utils.RolloutNode):
    """Class for tracking rollout node with probabilities.

    Equality and hashing use the documented candidate identity:

    ``(seq, fitness, edits_since_root, mutations_per_sequence, exploration_alpha)``

    Gradient arrays are excluded. This matches
    ``beam_common.gradabeam_candidate_key``.
    """

    probs: np.ndarray | None = field(default=None, hash=False, compare=False)
    pos_and_chars: PositionsAndCharactersType | None = field(
        default=None, hash=False, compare=False
    )
    edits_since_root: int | None = None
    mutations_per_sequence: float = 1.0
    exploration_alpha: float = 0.05


RolloutNode = RolloutNodeWithProbs


class GradaBeam(beam_common.BeamOptimizerMixin):
    """GradaBeam nucleic acid sequence designer with PBT."""

    def __init__(
        self,
        model_fn: Any,
        start_sequence: str,
        mutations_per_sequence: float,
        beam_size: int,
        n_rollouts_per_root: int,
        exploration_alpha: float,
        use_pbt: bool,
        max_rollout_len: int = 200,
        gradient_prob_cap: float = 0.10,
        max_logit: float = 3.0,
        rng_seed: int = 0,
        positions_to_mutate: list[int] | None = None,
        eval_batch_size: int = 1,
        debug: bool = False,
    ):
        if eval_batch_size != 1:
            raise ValueError(
                "GradaBeam only supports eval_batch_size=1. "
                "Larger values are rejected at construction because rollout "
                "batching is not implemented."
            )

        self.positions_to_mutate = beam_common.resolve_positions_to_mutate(
            start_sequence, positions_to_mutate
        )
        self.tism_positions = (
            None
            if len(self.positions_to_mutate) == len(start_sequence)
            else self.positions_to_mutate
        )

        assert mutations_per_sequence > 0
        assert beam_size > 0
        assert n_rollouts_per_root > 0
        assert exploration_alpha >= 0 and exploration_alpha <= 1

        self.exploration_alpha = exploration_alpha
        self.use_pbt = use_pbt

        self.model = ada_utils.ModelWrapper(
            model_fn,
            use_cache=True,
            debug=debug,
            tism_cost=1.0,
            start_sequence=start_sequence,
        )
        self.start_sequence = start_sequence
        self.beam_size = beam_size
        self.n_rollouts_per_root = n_rollouts_per_root
        self.alphabet = "".join(constants.VOCAB)
        self.positions_to_mutate_set = set(self.positions_to_mutate)
        self.eval_batch_size = eval_batch_size
        self.rng_seed = rng_seed
        self.rng = np.random.default_rng(rng_seed)
        # Dedicated stream for beam-ranking ties only. Same seed, independent
        # Generator, so mutation draws are unaffected.
        self.tie_rng = np.random.default_rng(rng_seed)
        self.candidate_key_fn = beam_common.gradabeam_candidate_key

        self.max_rollout_len = max_rollout_len
        self.gradient_prob_cap = gradient_prob_cap
        self.max_logit = max_logit
        self.debug = debug
        self._sampler_cache: dict[float, ada_utils.NumberEditsSampler] = {}

        assert isinstance(start_sequence, str)
        seed_node = RolloutNode(
            seq=start_sequence,
            fitness=0.0,
            edits_since_root=0,
            probs=None,
            pos_and_chars=None,
            mutations_per_sequence=float(mutations_per_sequence),
            exploration_alpha=float(exploration_alpha),
        )

        # Initialize with gradient-based mutations. Cache TISM action data by
        # sequence so the seed is scored once, not beam_size times.
        init_tism_cache: dict[str, TismActionData] = {}
        initialized_roots = self.initialize_roots_with_gradients(
            [seed_node] * beam_size, tism_cache=init_tism_cache
        )

        # Setup initial PBT sampling
        initial_sampler = self.get_sampler(seed_node.mutations_per_sequence)
        num_edit_locs = [int(x) for x in initial_sampler.sample(beam_size)]

        self.current_nodes = []
        for i in range(0, beam_size, self.eval_batch_size):
            cur_num_edits = num_edit_locs[i : i + self.eval_batch_size]
            cur_roots = initialized_roots[i : i + self.eval_batch_size]
            self.current_nodes.extend(
                self.mutate_nodes_gradabeam(
                    cur_roots,
                    cur_num_edits,
                    [seed_node.mutations_per_sequence] * len(cur_num_edits),
                )
            )
        self.current_nodes = beam_common.rank_nodes_by_fitness(
            self.current_nodes, self.beam_size, self.tie_rng
        )

    def get_sampler(
        self, mutations_per_sequence: float
    ) -> ada_utils.NumberEditsSampler:
        rounded_rate = round(mutations_per_sequence, 4)
        sampler = self._sampler_cache.get(rounded_rate)
        if sampler is None:
            mu = rounded_rate / len(self.positions_to_mutate)
            sampler = ada_utils.NumberEditsSamplerAdaBeam(
                sequence_len=len(self.positions_to_mutate),
                mutation_rate=mu,
                rng_seed=self.rng_seed,
            )
            self._sampler_cache[rounded_rate] = sampler
        return sampler

    def _get_next_mutation_params(self, node: RolloutNode) -> tuple[int, float]:
        """Calculates n_edits, new mutation rate, and target alpha for the child node."""
        current_rate = node.mutations_per_sequence

        if self.use_pbt:
            # Direct snap mode: sample edits using current rate, then snap rate to observation
            n_edits = int(self.get_sampler(current_rate).sample(1)[0])
            new_rate = float(max(1.0, n_edits))

            return n_edits, new_rate
        else:
            # PBT disabled: keep everything constant
            n_edits = int(self.get_sampler(current_rate).sample(1)[0])
            return n_edits, current_rate

    @staticmethod
    def debug_init_args():
        return {
            "model_fn": testing_utils.CountLetterModel(),
            "start_sequence": "AAAAAA",
            "beam_size": 10,
            "mutations_per_sequence": 1,
            "n_rollouts_per_root": 4,
            "eval_batch_size": 1,
            "rng_seed": 42,
            "exploration_alpha": 0.05,
            "use_pbt": True,
        }

    def run(self, n_steps: int):
        for _step in range(n_steps):
            self.current_nodes = self.propose_sequences(self.current_nodes)
            if self.debug and len(self.current_nodes) > 0:
                print(f"Step {_step} top score: {self.current_nodes[0].fitness}")
                rates = [n.mutations_per_sequence for n in self.current_nodes]
                print(f"[PBT] Mutation Rates of top candidates: {rates}")
                alphas = [n.exploration_alpha for n in self.current_nodes]
                print(
                    f"[PBT] Exploration Alphas of top candidates (high is uniform): {alphas}"
                )

    def propose_sequences(self, root_nodes: list[RolloutNode]) -> list[RolloutNode]:
        """Propose top `beam_size` sequences for evaluation."""
        candidates: dict[tuple, RolloutNode] = {}
        rollout_lengths: list[int] = []
        # Sequence -> immutable TISM action data (pos/char map, raw logits).
        # Never cache adaptive PBT node state here.
        tism_cache: dict[str, TismActionData] = {}

        root_nodes_effective = root_nodes * self.n_rollouts_per_root
        for i in range(0, len(root_nodes_effective), self.eval_batch_size):
            cur_root_nodes = root_nodes_effective[i : i + self.eval_batch_size]
            parent_nodes = cur_root_nodes

            assert len(parent_nodes) == 1, (
                "GradaBeam propose_sequences expects exactly one parent node."
            )
            parent_nodes = self.initialize_roots_with_gradients(
                parent_nodes, tism_cache=tism_cache
            )

            cur_nodes_visited, cur_rollout_lengths = self.rollout(
                parent_nodes=parent_nodes
            )
            for node in cur_nodes_visited:
                beam_common.accumulate_first_observed(
                    candidates, self.candidate_key_fn(node), node
                )
            rollout_lengths.extend(cur_rollout_lengths)

        if len(candidates) == 0:
            raise ValueError("No nodes generated.")

        self._last_candidates = list(candidates.values())
        return beam_common.rank_nodes_by_fitness(
            self._last_candidates, self.beam_size, self.tie_rng
        )

    def _get_tism_action_data(
        self, sequence: str, tism_cache: dict[str, TismActionData]
    ) -> TismActionData:
        cached = tism_cache.get(sequence)
        if cached is not None:
            return cached
        pos_and_chars, logits = self.model.get_tism(
            sequence=sequence, idxs=self.tism_positions, debug=self.debug
        )
        assert len(pos_and_chars) == 3 * len(self.positions_to_mutate), (
            len(pos_and_chars),
            len(self.positions_to_mutate),
            self.tism_positions,
        )
        assert len(pos_and_chars) == len(logits)
        tism_cache[sequence] = (pos_and_chars, logits)
        return pos_and_chars, logits

    def initialize_roots_with_gradients(
        self,
        nodes: list[RolloutNode],
        tism_cache: dict[str, TismActionData] | None = None,
    ) -> list[RolloutNode]:
        """Attach per-node probabilities to roots from cached TISM logits."""
        if tism_cache is None:
            tism_cache = {}

        grad_nodes = []
        for node in nodes:
            pos_and_chars, logits = self._get_tism_action_data(node.seq, tism_cache)
            grad_nodes.append(
                RolloutNode(
                    seq=node.seq,
                    fitness=node.fitness,
                    edits_since_root=0,
                    probs=self.logits_to_probs(logits, node.exploration_alpha),
                    pos_and_chars=pos_and_chars,
                    mutations_per_sequence=node.mutations_per_sequence,
                    exploration_alpha=node.exploration_alpha,
                )
            )
        return grad_nodes

    def rollout(
        self, parent_nodes: list[RolloutNode]
    ) -> tuple[list[RolloutNode], list[int]]:
        """Rollout with PBT."""
        candidates: dict[tuple, RolloutNode] = {}
        rollout_lengths: list[int] = []

        cur_rollout_length = 0
        while len(parent_nodes) > 0 and cur_rollout_length < self.max_rollout_len:
            # [PBT Modification]: Calculate dynamic rates/edits/alpha per parent
            num_edit_locs, new_rates = [], []
            for n in parent_nodes:
                n_edits, new_rate = self._get_next_mutation_params(n)
                num_edit_locs.append(n_edits)
                new_rates.append(new_rate)

            # [PBT Modification]: Pass new rates and target alphas to mutate
            children = self.mutate_nodes_gradabeam(
                parent_nodes, num_edit_locs, new_rates
            )

            for child in children:
                beam_common.accumulate_first_observed(
                    candidates, self.candidate_key_fn(child), child
                )

            cur_rollout_length += 1
            parent_nodes, terminated = beam_common.filter_accepted_children(
                children, parent_nodes, cur_rollout_length
            )
            rollout_lengths.extend(terminated)

        return list(candidates.values()), rollout_lengths

    def mutate_nodes_gradabeam(
        self,
        nodes: list[RolloutNode],
        num_edit_locs: list[int],
        new_rates: list[float],
    ) -> list[RolloutNode]:

        # [PBT Modification]: Validation
        assert (
            len(nodes) == len(num_edit_locs) == len(new_rates) <= self.eval_batch_size
        )

        seqs, new_probs, num_edits_effective, child_alphas = [], [], [], []
        for node, num_edits in zip(nodes, num_edit_locs):
            assert node.probs is not None
            assert node.pos_and_chars is not None
            num_available = (node.probs > 0).sum()
            effective_num_edits = min(num_edits, num_available)
            assert effective_num_edits > 0
            num_edits_effective.append(effective_num_edits)

            candidate, rel_pos_of_mutations = ada_utils.generate_random_mutant_tism(
                sequence=node.seq,
                pos_and_chars_to_mutate=node.pos_and_chars,
                random_n_loc=effective_num_edits,
                rng=self.rng,
                probs=node.probs,
                debug=self.debug,
            )
            seqs.append(candidate)

            # --- MASKING LOGIC ---
            # Zero out the positions we just changed
            # (We cannot trust the old gradient at these new chars)
            child_probs = node.probs.copy()
            child_probs[rel_pos_of_mutations] = 0.0
            total_p = child_probs.sum()
            if total_p > 0:
                child_probs /= total_p
            else:
                # Fallback: If we exhausted all probability mass,
                # revert to uniform or stop mutating.
                child_probs = np.ones_like(child_probs) / len(child_probs)
            new_probs.append(child_probs)

            # --- DIRECT SNAP FOR ALPHA ---
            if self.use_pbt:
                # Calculate posterior probability that mutations were uniform
                p_uniform = 1.0 / len(node.probs)
                # Get the P_final (from node.probs) for the chosen indices
                P_final_values = node.probs[rel_pos_of_mutations]
                # Calculate posterior per mutation: P(uniform | observed) = (alpha * p_uniform) / P_final
                # Add small epsilon to avoid division by zero
                posteriors = (node.exploration_alpha * p_uniform) / (
                    P_final_values + 1e-10
                )
                # Average this posterior over the number of edits made
                avg_posterior = float(np.mean(posteriors))
                child_alpha = float(np.clip(avg_posterior, 0.01, 0.99))
            else:
                # PBT disabled: keep alpha constant
                child_alpha = node.exploration_alpha

            child_alphas.append(child_alpha)

        fitnesses = self.get_batched_fitness(seqs)

        return [
            RolloutNode(
                seq=seq,
                fitness=float(f),
                probs=probs,
                edits_since_root=n.edits_since_root + int(num_edits),
                pos_and_chars=n.pos_and_chars,
                # [PBT Modification]: Child inherits new rate and alpha
                mutations_per_sequence=new_rate,
                exploration_alpha=child_alpha,
            )
            for seq, f, probs, n, num_edits, new_rate, child_alpha in zip(
                seqs,
                fitnesses,
                new_probs,
                nodes,
                num_edits_effective,
                new_rates,
                child_alphas,
            )
        ]

    # ... [Rest of file: probabilities_over_actions_from_tism, logits_to_probs] ...
    # (Functions below can remain identical to the original)
    def probabilities_over_actions_from_tism(
        self, nodes: list[RolloutNode]
    ) -> tuple[list[np.ndarray], list[PositionsAndCharactersType]]:
        tism_cache: dict[str, TismActionData] = {}
        probs_list, pos_and_chars_list = [], []
        for n in nodes:
            pos_and_chars, logits = self._get_tism_action_data(n.seq, tism_cache)
            probs_list.append(self.logits_to_probs(logits, n.exploration_alpha))
            pos_and_chars_list.append(pos_and_chars)
        return probs_list, pos_and_chars_list

    def logits_to_probs(self, logits: np.ndarray, alpha: float) -> np.ndarray:
        # Normalize logit standard deviation.
        std_dev = np.std(logits)
        if std_dev < 1e-9:
            return np.ones_like(logits) / len(logits)
        scaled_logits = logits / std_dev

        # Scale logits by a dynamic temperature.
        dynamic_temp = max(1.0, np.max(scaled_logits) / self.max_logit)
        scaled_logits = scaled_logits / dynamic_temp

        gradient_probs = softmax(scaled_logits)

        # Somewhat normalize probabilities to a cap.
        gradient_probs = np.minimum(gradient_probs, self.gradient_prob_cap)
        gradient_probs /= np.sum(gradient_probs)

        # Mix in uniform exploration.
        n_actions = len(scaled_logits)
        uniform_probs = np.ones(n_actions) / n_actions
        final_probs = ((1.0 - alpha) * gradient_probs) + (alpha * uniform_probs)

        return final_probs / np.sum(final_probs)
