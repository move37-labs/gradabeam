"""RED regression guard — AdaBeam per-step overhead in action_space_mutation.

Measured context (Step 3, dummy oracle, L=3000, beam_size=2, n_rollouts=4):
  build_uniform_pos_and_chars: 8 calls/step × 1154 µs/call ≈ 9.2 ms/step
  The call count (8) = beam_size × n_rollouts_per_root — one rebuild of the
  O(L) action list per rollout root, even when the root sequence is unchanged.

Fix target: cache the result per (sequence, positions_to_mutate) so at most
  beam_size=2 calls/step are needed (once per UNIQUE beam candidate).

This test FAILS on action_space_mutation (unfixed) and will PASS after caching.
It does NOT test timing (too flaky) — it tests the call count, which is the
deterministic proxy for the overhead.

To run:
    conda run -n gradabeam python -m pytest gradabeam/adabeam_perf_regression_test.py -v
"""

import unittest
from unittest.mock import patch

import gradabeam.ada_utils as ada_utils
from gradabeam import AdaBeam
from gradabeam import testing_utils


class TestBuildUniformCallCountPerStep(unittest.TestCase):
    """Regression guard: build_uniform_pos_and_chars call count per step.

    On action_space_mutation (unfixed):
      _attach_uniform_probs is called beam_size × n_rollouts_per_root = 8
      times per step, each calling build_uniform_pos_and_chars(node.seq, ...)
      to rebuild the full 3L list-of-tuples from scratch.  When the beam has
      not changed (same 2 sequences in both roots), 6 of those 8 rebuilds are
      pure redundancy.

    After fix (caching per sequence):
      At most beam_size = 2 calls per step (one per unique beam candidate).
    """

    BEAM_SIZE = 2
    N_ROLLOUTS = 4
    # Upper bound AFTER fix: one call per unique beam candidate per step.
    MAX_CALLS_AFTER_FIX = BEAM_SIZE  # = 2

    def _make_designer(self, seq_len: int = 50) -> AdaBeam:
        return AdaBeam(
            model_fn=testing_utils.CountLetterModel(),
            start_sequence="A" * seq_len,
            beam_size=self.BEAM_SIZE,
            mutations_per_sequence=1.0,
            n_rollouts_per_root=self.N_ROLLOUTS,
            skip_repeat_sequences=False,
            eval_batch_size=1,
            rng_seed=42,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
