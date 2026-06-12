"""Tests for gradabeam_designer.py

To test:
```zsh
pytest gradabeam/gradabeam_designer_test.py
```
"""

import numpy as np
import pytest

from gradabeam import testing_utils

from gradabeam.gradabeam_designer import GradaBeam
from gradabeam.adaptive_rollout import RolloutNodeWithProbs


def test_gradabeam_sanity():
    kwargs = GradaBeam.debug_init_args()
    kwargs["debug"] = True

    gradabeam = GradaBeam(**kwargs)

    gradabeam.run(n_steps=2)

    out_seqs = gradabeam.get_samples(kwargs["beam_size"])
    del out_seqs


def test_gradabeam_convergence():
    kwargs = GradaBeam.debug_init_args()

    model_fn = kwargs["model_fn"]
    # Be greedy so this definitely improves.
    kwargs["exploration_alpha"] = 0.0001

    start_seq = "A" * 100

    start_score = model_fn([start_seq])[0]

    kwargs["start_sequence"] = start_seq
    gradabeam = GradaBeam(**kwargs)

    gradabeam.run(n_steps=2)
    out_seqs = gradabeam.get_samples(kwargs["beam_size"])
    out_seq_scores = np.array([model_fn([s])[0] for s in out_seqs])
    # GradaBeam should improve (lower is better).
    assert out_seq_scores[0] < start_score


def test_gradabeam_positions_to_mutate():
    """No matter how many iterations, positions outside `positions_to_mutate` shouldn't change."""

    start_seq = "A" * 100

    beam_size = 2
    kwargs = GradaBeam.debug_init_args()
    kwargs["start_sequence"] = start_seq
    kwargs["beam_size"] = beam_size
    gradabeam = GradaBeam(**kwargs, positions_to_mutate=list(range(20)))

    for i in range(4):
        gradabeam.run(n_steps=1)

        out_seqs = gradabeam.get_samples(beam_size)
        for seq in out_seqs:
            for s in seq[20:]:
                assert s == "A", seq


@pytest.mark.skip(reason="Disable multi-batch for now.")
@pytest.mark.parametrize("eval_batch_size", [1, 2, 4])
def test_gradabeam_eval_batch_size_sanity(eval_batch_size):
    """Test that `eval_batch_size` works."""
    kwargs = GradaBeam.debug_init_args()
    kwargs["eval_batch_size"] = eval_batch_size
    gradabeam = GradaBeam(**kwargs)

    gradabeam.run(n_steps=2)

    # TODO(joelshor):
    # Add correctness checks.


def test_gradabeam_eval_batch_size_consistency():
    """Test that `eval_batch_size` is consistent."""
    model_fn = testing_utils.CountLetterModel()

    seqs = [
        "".join(np.random.choice(["A", "G", "T", "C"], size=100)) for _ in range(10)
    ]

    kwargs = GradaBeam.debug_init_args()
    kwargs["model_fn"] = model_fn
    kwargs["start_sequence"] = "A" * 100

    kwargs["eval_batch_size"] = 1
    gradabeam_1 = GradaBeam(**kwargs)

    kwargs["eval_batch_size"] = 2
    gradabeam_2 = GradaBeam(**kwargs)

    kwargs["eval_batch_size"] = 4
    gradabeam_4 = GradaBeam(**kwargs)

    scores1 = gradabeam_1.get_batched_fitness(seqs)
    scores2 = gradabeam_2.get_batched_fitness(seqs)
    scores4 = gradabeam_4.get_batched_fitness(seqs)

    assert np.array_equal(scores1, scores2)
    assert np.array_equal(scores1, scores4)
    assert np.array_equal(scores2, scores4)


class TestGradientAlignment:
    def test_gradient_map_matches_territory(self):
        """
        Verifies that when CountLetterModel says 'C is good',
        GradaBeam picks 'C' and fitness improves.

        Uses start_sequence="AC": position 0 is A (one beneficial C-mutation exists)
        and position 1 is already C.  This gives a highly-concentrated gradient at
        pos0→C (prob ≈ 0.57 with gradient_prob_cap=1.0 and alpha=0.0), which is
        reliable regardless of the initial-beam RNG path.

        An explicit root node with exploration_alpha=0.0 is passed to
        initialize_roots_with_gradients so the test does not depend on which
        sequence happened to land in current_nodes[0].
        """
        start_sequence = "AC"

        # We want Positive Gradients (+1) for 'C'.
        target_char = "C"
        model = testing_utils.CountLetterModel(target_char=target_char)

        # Initialize GradaBeam
        gb = GradaBeam(
            model_fn=model,
            start_sequence=start_sequence,
            mutations_per_sequence=1,
            beam_size=5,
            n_rollouts_per_root=1,
            eval_batch_size=1,
            exploration_alpha=0.0,
            use_pbt=True,
            gradient_prob_cap=1.0,  # No cap.
            rng_seed=42,
            debug=True,
        )

        print("\n[Test] Calculating gradients on root...")
        # Create an explicit root node on the START SEQUENCE with alpha=0.0 so
        # the test is independent of which sequence ended up in current_nodes[0].
        explicit_root = RolloutNodeWithProbs(
            seq=start_sequence,
            fitness=np.float32(0.0),
            edits_since_root=0,
            mutations_per_sequence=1.0,
            exploration_alpha=0.0,
            probs=None,
            gradient_probs=None,
            pos_and_chars=None,
        )
        nodes = gb.initialize_roots_with_gradients([explicit_root])
        root = nodes[0]

        # Find the max probability action
        best_flat_idx = np.argmax(root.probs)
        best_pos, best_char = root.pos_and_chars[best_flat_idx]
        best_prob = root.probs[best_flat_idx]

        print(
            f"[Test] Algorithm chose: Pos {best_pos} -> '{best_char}' with prob {best_prob:.4f}"
        )

        # ASSERTION 1: Did it pick 'C'?
        assert best_char == target_char, (
            f"Vocab Mismatch! Model wanted '{target_char}', but GradaBeam picked '{best_char}'."
        )

        # ASSERTION 2: Is it confident?
        # With start_sequence="AC", the only beneficial mutation is pos0 → C
        # (prob ≈ 0.57 with no cap and alpha=0.0); other actions are neutral or harmful.
        assert best_prob >= 0.5, (
            f"Entropy Death! Expected high confidence (≥0.5) for '{target_char}', got {best_prob:.4f}."
        )

        # 3. Verify Actual Fitness Gain
        original_seq = start_sequence
        mut_list = list(original_seq)
        mut_list[int(best_pos)] = best_char
        mutant_seq = "".join(mut_list)

        base_score = gb.get_batched_fitness([original_seq])[0]
        new_score = gb.get_batched_fitness([mutant_seq])[0]
        delta = new_score - base_score

        print(f"[Test] Fitness: {base_score} -> {new_score} (Delta: {delta})")

        # ASSERTION 3: Maximization check
        # Score should increase (Delta > 0)
        assert delta > 0, (
            f"Optimization Failure! Mutating to '{best_char}' should improve score, but delta was {delta}."
        )
