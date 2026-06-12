"""GradaBeam -- thin constructor over AdaptiveRolloutDesigner.

Mutation is in position space (generate_random_mutant_positionspace).
Masking zeros edited positions in the L-vector and renormalizes, preventing
double-edits within a rollout.  alpha-posterior uses
p_uniform = 1/n_available_positions; P_final is recomputed at each step as
the (1-alpha) * grad + alpha * unif mixture over available positions.
"""

from typing import Any

from gradabeam import testing_utils
from gradabeam.adaptive_rollout import (
    AdaptiveRolloutDesigner,
    GradientPositionStrategy,
)


class GradaBeam(AdaptiveRolloutDesigner):
    """GradaBeam nucleic acid sequence designer with PBT.

    Delegates to AdaptiveRolloutDesigner with:
      strategy = GradientPositionStrategy()
      use_gradients = True
      use_pbt = <from constructor>

    The gradient information is marginalized to position space via
    tism_probs_to_position_weights, mixed with uniform exploration at weight α,
    and used to sample N distinct positions for each rollout step.
    """

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
    ) -> None:
        """GradaBeam nucleic acid sequence designer.

        Args:
            model_fn: Oracle / scoring model (must implement tism_torch).
            start_sequence: Seed sequence for the search.
            mutations_per_sequence: Expected mutations per rollout step.
            beam_size: Candidates to keep between rounds.
            n_rollouts_per_root: Rollouts launched per beam candidate per round.
            exploration_alpha: Initial α (0=pure gradient, 1=pure uniform).
            use_pbt: Enable Population Based Training for adaptive mutation
                rate and α.
            max_rollout_len: Maximum rollout depth.
            gradient_prob_cap: Per-action probability cap after softmax.
            max_logit: Dynamic temperature ceiling for logit scaling.
            rng_seed: Pseudo-random seed.
            positions_to_mutate: 0-based positions that may be mutated; None = all.
            eval_batch_size: Oracle calls per batch.
            debug: Print diagnostic information.
        """
        assert exploration_alpha >= 0 and exploration_alpha <= 1
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
            strategy=GradientPositionStrategy(),
            use_gradients=True,
            use_pbt=use_pbt,
            exploration_alpha=exploration_alpha,
            gradient_prob_cap=gradient_prob_cap,
            max_logit=max_logit,
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
            "rng_seed": 42,
            "exploration_alpha": 0.5,
            "use_pbt": True,
        }
