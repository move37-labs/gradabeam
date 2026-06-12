"""AdaBeam — thin constructor over AdaptiveRolloutDesigner.

Public constructor signature is unchanged; existing callers and the CLI
continue to work without modification.
"""

from typing import Any

import numpy as np

from gradabeam import ada_utils
from gradabeam import testing_utils
from gradabeam.adaptive_rollout import AdaptiveRolloutDesigner, UniformPositionStrategy


class AdaBeam(AdaptiveRolloutDesigner):
    """AdaBeam nucleic acid sequence designer.

    Defaults to :
      strategy = UniformPositionStrategy()
      use_gradients = False
      use_pbt = False
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
            strategy=UniformPositionStrategy(),
            use_gradients=False,
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
        }
