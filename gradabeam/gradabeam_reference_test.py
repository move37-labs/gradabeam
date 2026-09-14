"""Tests for gradabeam_reference.py.

To test:
```zsh
pytest gradabeam/gradabeam_reference_test.py
```
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import pytest

from gradabeam import testing_utils
from gradabeam.gradabeam_reference import GradaBeamReference

# Frozen NucleoBench paper-result snapshots. Generated from
# nucleobench/optimizations/ada/gradabeam/gradabeam.py
# (blob 569ea54af6331d07edcd4bb294cc09186b451824) with PYTHONHASHSEED=0.
_GRADABEAM_SHARED = {
    "start_sequence": "A" * 20,
    "beam_size": 4,
    "mutations_per_sequence": 1.0,
    "n_rollouts_per_root": 2,
    "eval_batch_size": 1,
    "rng_seed": 42,
    "exploration_alpha": 0.05,
    "debug": False,
}

_NUCLEOBENCH_GRADABEAM_SNAPSHOTS = {
    "gradabeam_pbt": {
        "initial": [
            {
                "seq": "AAAAAAAACAAAAAACAAAA",
                "fitness": 2.0,
                "fitness_type": "float",
                "edits_since_root": 2,
                "mutations_per_sequence": 1.0,
                "exploration_alpha": 0.021286948052531715,
            },
            {
                "seq": "AAAAAAAAAAAAAAAAACAA",
                "fitness": 1.0,
                "fitness_type": "float",
                "edits_since_root": 1,
                "mutations_per_sequence": 1.0,
                "exploration_alpha": 0.021286948052531715,
            },
            {
                "seq": "AGAAAAAAAAAAATAAAAAA",
                "fitness": 0.0,
                "fitness_type": "float",
                "edits_since_root": 2,
                "mutations_per_sequence": 1.0,
                "exploration_alpha": 0.1535761258264875,
            },
            {
                "seq": "AAAAAAAAAAAAAAACAAAC",
                "fitness": 2.0,
                "fitness_type": "float",
                "edits_since_root": 2,
                "mutations_per_sequence": 1.0,
                "exploration_alpha": 0.021286948052531715,
            },
        ],
        "samples_init": [
            "AAAAAAAACAAAAAACAAAA",
            "AAAAAAAAAAAAAAACAAAC",
            "AAAAAAAAAAAAAAAAACAA",
            "AGAAAAAAAAAAATAAAAAA",
        ],
        "after_3": [
            {
                "seq": "CCCCCCCCCCCCCCCCCACA",
                "fitness": 18.0,
                "fitness_type": "float",
                "edits_since_root": 5,
                "mutations_per_sequence": 5.0,
                "exploration_alpha": 0.011639341319611026,
            },
            {
                "seq": "CCCACCCCCCCCCCCCCCTC",
                "fitness": 18.0,
                "fitness_type": "float",
                "edits_since_root": 3,
                "mutations_per_sequence": 3.0,
                "exploration_alpha": 0.01,
            },
            {
                "seq": "CCCCCCCCCCCCACCCCCTC",
                "fitness": 18.0,
                "fitness_type": "float",
                "edits_since_root": 5,
                "mutations_per_sequence": 2.0,
                "exploration_alpha": 0.011373372766479156,
            },
            {
                "seq": "CCCCCTCCCCCTCCCCCCCC",
                "fitness": 18.0,
                "fitness_type": "float",
                "edits_since_root": 4,
                "mutations_per_sequence": 4.0,
                "exploration_alpha": 0.01,
            },
        ],
        "samples": [
            "CCCCCCCCCCCCCCCCCACA",
            "CCCACCCCCCCCCCCCCCTC",
            "CCCCCCCCCCCCACCCCCTC",
            "CCCCCTCCCCCTCCCCCCCC",
        ],
    },
    "gradabeam_no_pbt": {
        "initial": [
            {
                "seq": "AAAAAAAACAAAAAACAAAA",
                "fitness": 2.0,
                "fitness_type": "float",
                "edits_since_root": 2,
                "mutations_per_sequence": 1.0,
                "exploration_alpha": 0.05,
            },
            {
                "seq": "AAAAAAAAAAAAAAAAACAA",
                "fitness": 1.0,
                "fitness_type": "float",
                "edits_since_root": 1,
                "mutations_per_sequence": 1.0,
                "exploration_alpha": 0.05,
            },
            {
                "seq": "AGAAAAAAAAAAATAAAAAA",
                "fitness": 0.0,
                "fitness_type": "float",
                "edits_since_root": 2,
                "mutations_per_sequence": 1.0,
                "exploration_alpha": 0.05,
            },
            {
                "seq": "AAAAAAAAAAAAAAACAAAC",
                "fitness": 2.0,
                "fitness_type": "float",
                "edits_since_root": 2,
                "mutations_per_sequence": 1.0,
                "exploration_alpha": 0.05,
            },
        ],
        "samples_init": [
            "AAAAAAAACAAAAAACAAAA",
            "AAAAAAAAAAAAAAACAAAC",
            "AAAAAAAAAAAAAAAAACAA",
            "AGAAAAAAAAAAATAAAAAA",
        ],
        "after_3": [
            {
                "seq": "CCCCCCCCCCCCCCCCCCCC",
                "fitness": 20.0,
                "fitness_type": "float",
                "edits_since_root": 1,
                "mutations_per_sequence": 1.0,
                "exploration_alpha": 0.05,
            },
            {
                "seq": "CCCCCCCCCCCCCCCCCCAC",
                "fitness": 19.0,
                "fitness_type": "float",
                "edits_since_root": 2,
                "mutations_per_sequence": 1.0,
                "exploration_alpha": 0.05,
            },
            {
                "seq": "CCTCCCCCCCCCCCCCCCCC",
                "fitness": 19.0,
                "fitness_type": "float",
                "edits_since_root": 2,
                "mutations_per_sequence": 1.0,
                "exploration_alpha": 0.05,
            },
            {
                "seq": "CCACCCCGCCCCCCCCCCCC",
                "fitness": 18.0,
                "fitness_type": "float",
                "edits_since_root": 4,
                "mutations_per_sequence": 1.0,
                "exploration_alpha": 0.05,
            },
        ],
        "samples": [
            "CCCCCCCCCCCCCCCCCCCC",
            "CCCCCCCCCCCCCCCCCCAC",
            "CCTCCCCCCCCCCCCCCCCC",
            "CCACCCCGCCCCCCCCCCCC",
        ],
    },
    "gradabeam_positions": {
        "initial": [
            {
                "seq": "AAACAACAAAAAAAAAAAAA",
                "fitness": 2.0,
                "fitness_type": "float",
                "edits_since_root": 2,
                "mutations_per_sequence": 1.0,
                "exploration_alpha": 0.0213202690211942,
            },
            {
                "seq": "AAAAAAGAAAAAAAAAAAAA",
                "fitness": 0.0,
                "fitness_type": "float",
                "edits_since_root": 1,
                "mutations_per_sequence": 1.0,
                "exploration_alpha": 0.15271515958688567,
            },
            {
                "seq": "CAAAACAAAAAAAAAAAAAA",
                "fitness": 2.0,
                "fitness_type": "float",
                "edits_since_root": 2,
                "mutations_per_sequence": 1.0,
                "exploration_alpha": 0.0213202690211942,
            },
            {
                "seq": "AAAAAACGAAAAAAAAAAAA",
                "fitness": 1.0,
                "fitness_type": "float",
                "edits_since_root": 2,
                "mutations_per_sequence": 1.0,
                "exploration_alpha": 0.08701771430403993,
            },
        ],
        "samples_init": [
            "AAACAACAAAAAAAAAAAAA",
            "CAAAACAAAAAAAAAAAAAA",
            "AAAAAACGAAAAAAAAAAAA",
            "AAAAAAGAAAAAAAAAAAAA",
        ],
        "after_3": [
            {
                "seq": "CCCCCCGCAAAAAAAAAAAA",
                "fitness": 7.0,
                "fitness_type": "float",
                "edits_since_root": 2,
                "mutations_per_sequence": 2.0,
                "exploration_alpha": 0.010932120901498525,
            },
            {
                "seq": "CCCCCCACAAAAAAAAAAAA",
                "fitness": 7.0,
                "fitness_type": "float",
                "edits_since_root": 4,
                "mutations_per_sequence": 4.0,
                "exploration_alpha": 0.01,
            },
            {
                "seq": "CCCCCGCAAAAAAAAAAAAA",
                "fitness": 6.0,
                "fitness_type": "float",
                "edits_since_root": 2,
                "mutations_per_sequence": 2.0,
                "exploration_alpha": 0.012711622753910865,
            },
            {
                "seq": "CCCTTCCCAAAAAAAAAAAA",
                "fitness": 6.0,
                "fitness_type": "float",
                "edits_since_root": 2,
                "mutations_per_sequence": 2.0,
                "exploration_alpha": 0.012711622753910865,
            },
        ],
        "samples": [
            "CCCCCCGCAAAAAAAAAAAA",
            "CCCCCCACAAAAAAAAAAAA",
            "CCCCCGCAAAAAAAAAAAAA",
            "CCCTTCCCAAAAAAAAAAAA",
        ],
    },
}


def _dump_nodes(nodes) -> list[dict]:
    return [
        {
            "seq": n.seq,
            "fitness": None if n.fitness is None else float(n.fitness),
            "fitness_type": type(n.fitness).__name__,
            "edits_since_root": n.edits_since_root,
            "mutations_per_sequence": float(n.mutations_per_sequence),
            "exploration_alpha": float(n.exploration_alpha),
        }
        for n in nodes
    ]


def collect_gradabeam_reference_snapshots() -> dict:
    """Collect characterization snapshots. Call with PYTHONHASHSEED=0."""
    model = testing_utils.CountLetterModel()

    def run(**kwargs):
        gb = GradaBeamReference(model_fn=model, **kwargs)
        initial = _dump_nodes(gb.current_nodes)
        samples_init = gb.get_samples(kwargs["beam_size"])
        gb.run(n_steps=3)
        return {
            "initial": initial,
            "samples_init": samples_init,
            "after_3": _dump_nodes(gb.current_nodes),
            "samples": gb.get_samples(kwargs["beam_size"]),
        }

    return {
        "gradabeam_pbt": run(**_GRADABEAM_SHARED, use_pbt=True),
        "gradabeam_no_pbt": run(**_GRADABEAM_SHARED, use_pbt=False),
        "gradabeam_positions": run(
            **_GRADABEAM_SHARED,
            use_pbt=True,
            positions_to_mutate=list(range(8)),
        ),
    }


def test_gradabeam_reference_sanity():
    kwargs = GradaBeamReference.debug_init_args()
    kwargs["debug"] = True

    gradabeam = GradaBeamReference(**kwargs)

    gradabeam.run(n_steps=2)

    out_seqs = gradabeam.get_samples(kwargs["beam_size"])
    del out_seqs


def test_gradabeam_reference_convergence():
    kwargs = GradaBeamReference.debug_init_args()

    model_fn = kwargs["model_fn"]
    # Be greedy so this definitely improves.
    kwargs["exploration_alpha"] = 0.0001

    start_seq = "A" * 100

    start_score = model_fn([start_seq])[0]

    kwargs["start_sequence"] = start_seq
    gradabeam = GradaBeamReference(**kwargs)

    gradabeam.run(n_steps=2)
    out_seqs = gradabeam.get_samples(kwargs["beam_size"])
    out_seq_scores = np.array([model_fn([s])[0] for s in out_seqs])
    # GradaBeam should improve (lower is better).
    assert out_seq_scores[0] < start_score


def test_gradabeam_reference_positions_to_mutate():
    """No matter how many iterations, positions outside `positions_to_mutate` shouldn't change."""

    start_seq = "A" * 100

    beam_size = 2
    kwargs = GradaBeamReference.debug_init_args()
    kwargs["start_sequence"] = start_seq
    kwargs["beam_size"] = beam_size
    gradabeam = GradaBeamReference(**kwargs, positions_to_mutate=list(range(20)))

    for _ in range(4):
        gradabeam.run(n_steps=1)

        out_seqs = gradabeam.get_samples(beam_size)
        for seq in out_seqs:
            for s in seq[20:]:
                assert s == "A", seq


@pytest.mark.skip(reason="Disable multi-batch for now.")
@pytest.mark.parametrize("eval_batch_size", [1, 2, 4])
def test_gradabeam_reference_eval_batch_size_sanity(eval_batch_size):
    """Test that `eval_batch_size` works."""
    kwargs = GradaBeamReference.debug_init_args()
    kwargs["eval_batch_size"] = eval_batch_size
    gradabeam = GradaBeamReference(**kwargs)

    gradabeam.run(n_steps=2)


def test_gradabeam_reference_eval_batch_size_consistency():
    """Test that `eval_batch_size` is consistent."""
    model_fn = testing_utils.CountLetterModel()

    seqs = [
        "".join(np.random.choice(["A", "G", "T", "C"], size=100)) for _ in range(10)
    ]

    kwargs = GradaBeamReference.debug_init_args()
    kwargs["model_fn"] = model_fn
    kwargs["start_sequence"] = "A" * 100

    kwargs["eval_batch_size"] = 1
    gradabeam_1 = GradaBeamReference(**kwargs)

    kwargs["eval_batch_size"] = 2
    gradabeam_2 = GradaBeamReference(**kwargs)

    kwargs["eval_batch_size"] = 4
    gradabeam_4 = GradaBeamReference(**kwargs)

    scores1 = gradabeam_1.get_batched_fitness(seqs)
    scores2 = gradabeam_2.get_batched_fitness(seqs)
    scores4 = gradabeam_4.get_batched_fitness(seqs)

    assert np.array_equal(scores1, scores2)
    assert np.array_equal(scores1, scores4)
    assert np.array_equal(scores2, scores4)


def test_gradabeam_reference_child_fitness_is_python_float():
    kwargs = GradaBeamReference.debug_init_args()
    gb = GradaBeamReference(**kwargs)
    assert all(type(n.fitness) is float for n in gb.current_nodes)


class TestGradientAlignment:
    def test_gradient_map_matches_territory(self):
        """
        Verifies that when CountLetterModel says 'C is good',
        GradaBeam picks 'C' and fitness improves.
        """
        start_sequence = "AA"

        # We want Positive Gradients (+1) for 'C'.
        target_char = "C"
        model = testing_utils.CountLetterModel(target_char=target_char)

        # Initialize GradaBeam
        gb = GradaBeamReference(
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
        nodes = gb.initialize_roots_with_gradients([gb.current_nodes[0]])
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
        assert best_prob >= 0.5, (
            f"Entropy Death! Expected high confidence (>0.8) for '{target_char}', got {best_prob:.4f}."
        )

        # 3. Verify Actual Fitness Gain
        original_seq = root.seq
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


def test_gradabeam_reference_matches_nucleobench_snapshots():
    """Lock paper-result trajectories, including PYTHONHASHSEED-sensitive set order."""
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = "0"
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import json; "
                "from gradabeam.gradabeam_reference_test import "
                "collect_gradabeam_reference_snapshots; "
                "print(json.dumps(collect_gradabeam_reference_snapshots()))"
            ),
        ],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    got = json.loads(proc.stdout)
    assert got == _NUCLEOBENCH_GRADABEAM_SNAPSHOTS
