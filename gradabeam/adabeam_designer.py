"""AdaBeam — thin constructor over AdaptiveRolloutDesigner.

Public constructor signature is unchanged; existing callers and the CLI
continue to work without modification.

Default behavior as of Plan 01 part 1b:
  allow_silent_edits=False (corrected path, never the silent legacy operator).
  Pass allow_silent_edits=True only to reproduce pre-refactor published numbers.
"""

from typing import Any

import numpy as np

from gradabeam import ada_utils
from gradabeam import testing_utils
from gradabeam.adaptive_rollout import AdaptiveRolloutDesigner, UniformActionStrategy


class AdaBeam(AdaptiveRolloutDesigner):
    """AdaBeam nucleic acid sequence designer.

    Defaults to the corrected action-space path (allow_silent_edits=False):
      strategy = UniformActionStrategy(allow_silent_edits=False)
      use_gradients = False
      use_pbt = False

    Pass allow_silent_edits=True to obtain the reproduction-only legacy path,
    which matches the pre-refactor AdaBeam RNG stream bit-for-bit and is pinned
    by test_adabeam_equivalence.  This path is permanent but NOT the default.
    """

    def __init__(
        self,
        model_fn: Any,
        start_sequence: str,
        mutations_per_sequence: float,
        beam_size: int,
        n_rollouts_per_root: int,
        eval_batch_size: int,
        skip_repeat_sequences: bool,
        rng_seed: int = 0,
        positions_to_mutate: list[int] | None = None,
        max_rollout_len: int = 200,
        debug: bool = False,
        allow_silent_edits: bool = False,
    ) -> None:
        """AdaBeam nucleic acid sequence designer.

        Args:
            model_fn: Oracle / scoring model.
            start_sequence: Seed sequence for the search.
            mutations_per_sequence: Expected mutations per rollout step.
            beam_size: Candidates to keep between rounds.
            n_rollouts_per_root: Rollouts launched per beam candidate per round.
            eval_batch_size: Oracle calls per batch.
            skip_repeat_sequences: Retry mutation until a novel sequence is found.
            rng_seed: Pseudo-random seed.
            positions_to_mutate: 0-based positions that may be mutated; None = all.
            max_rollout_len: Maximum rollout depth before terminating.
            debug: Print diagnostic information.
            allow_silent_edits: When False (default), use the corrected
                position-space operator — every edit changes a base.  When True,
                use the legacy reproduction operator (generate_random_mutant_v2,
                ~25% silent edits); required to reproduce published paper numbers.
        """
        super().__init__(
            model_fn=model_fn,
            start_sequence=start_sequence,
            mutations_per_sequence=mutations_per_sequence,
            beam_size=beam_size,
            n_rollouts_per_root=n_rollouts_per_root,
            eval_batch_size=eval_batch_size,
            rng_seed=rng_seed,
            positions_to_mutate=positions_to_mutate,
            max_rollout_len=max_rollout_len,
            debug=debug,
            strategy=UniformActionStrategy(allow_silent_edits=allow_silent_edits),
            use_gradients=False,
            allow_silent_edits=allow_silent_edits,
            use_pbt=False,
            skip_repeat_sequences=skip_repeat_sequences,
        )

    @staticmethod
    def debug_init_args() -> dict:
        return {
            "model_fn": testing_utils.CountLetterModel(),
            "start_sequence": "AAAAAA",
            "beam_size": 10,
            "mutations_per_sequence": 1,
            "n_rollouts_per_root": 4,
            "eval_batch_size": 1,
            "skip_repeat_sequences": False,
            "rng_seed": 42,
            # allow_silent_edits intentionally absent; defaults to False (corrected).
            # Tests that need the legacy path must pass allow_silent_edits=True explicitly.
        }

    # generate_mutations kept for any external callers that relied on it
    def generate_mutations(self, sequence: str, random_n_locs: int) -> str:
        """Convenience wrapper around generate_random_mutant_v2."""
        return ada_utils.generate_random_mutant_v2(
            sequence=sequence,
            positions_to_mutate=self.positions_to_mutate,
            random_n_loc=random_n_locs,
            alphabet=self.alphabet,
            rng=self.rng,
        )

    def mutate_nodes(
        self,
        nodes: list,
        num_edit_locs: list | np.ndarray,
        max_num_tries: int = 300,
    ) -> list:
        """Public alias kept for external callers (delegates to _mutate_legacy_nodes)."""
        return self._mutate_legacy_nodes(nodes, num_edit_locs, max_num_tries)
