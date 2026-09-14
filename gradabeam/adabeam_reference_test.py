"""Tests for adabeam_reference.py.

To test:
```zsh
pytest gradabeam/adabeam_reference_test.py
```
"""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys

import numpy as np
import pytest

from gradabeam import testing_utils
from gradabeam.adabeam_reference import AdaBeamReference

# Frozen NucleoBench paper-result snapshots. Generated from
# nucleobench/optimizations/ada/adabeam/adabeam.py
# (blob 767402c810da5b8df45f81b3238897d7bb194af0) with PYTHONHASHSEED=0.
_ADABEAM_SHARED = {
    "start_sequence": "A" * 20,
    "beam_size": 4,
    "mutations_per_sequence": 1.0,
    "n_rollouts_per_root": 2,
    "eval_batch_size": 1,
    "rng_seed": 42,
    "debug": False,
}

_NUCLEOBENCH_ADABEAM_SNAPSHOTS = {
    "adabeam_default": {
        "initial": [
            {"seq": "ACAAAAAAAAAAAAACAAAA", "fitness": 2.0, "fitness_type": "float64"},
            {"seq": "AAAAAAAAAAAAAAAAAAAA", "fitness": 0.0, "fitness_type": "float64"},
            {"seq": "AAAAGAAAAAAAATAAAAAA", "fitness": 0.0, "fitness_type": "float64"},
            {"seq": "AAAAAAAAAAAAATAGAAAA", "fitness": 0.0, "fitness_type": "float64"},
        ],
        "samples_init": [
            "ACAAAAAAAAAAAAACAAAA",
            "AAAAAAAAAAAAAAAAAAAA",
            "AAAAGAAAAAAAATAAAAAA",
            "AAAAAAAAAAAAATAGAAAA",
        ],
        "after_3": [
            {"seq": "AACCTTCCTAAGCCCCCCCA", "fitness": 11.0, "fitness_type": "float64"},
            {"seq": "GACCTCCCTAAGCCCCCACA", "fitness": 11.0, "fitness_type": "float64"},
            {"seq": "GCCCTAACCACCCCCAAACA", "fitness": 11.0, "fitness_type": "float64"},
            {"seq": "TCCCTAACCACCCCCAAAAA", "fitness": 10.0, "fitness_type": "float64"},
        ],
        "samples": [
            "AACCTTCCTAAGCCCCCCCA",
            "GACCTCCCTAAGCCCCCACA",
            "GCCCTAACCACCCCCAAACA",
            "TCCCTAACCACCCCCAAAAA",
        ],
    },
    "adabeam_skip_repeat": {
        "initial": [
            {"seq": "ACAAAAAAAAAAAAACAAAA", "fitness": 2.0, "fitness_type": "float64"},
            {"seq": "AAAAAAAAAAAAAAAAAAAA", "fitness": 0.0, "fitness_type": "float64"},
            {"seq": "AAAAGAAAAAAAATAAAAAA", "fitness": 0.0, "fitness_type": "float64"},
            {"seq": "AAAAAAAAAAAAATAGAAAA", "fitness": 0.0, "fitness_type": "float64"},
        ],
        "samples_init": [
            "ACAAAAAAAAAAAAACAAAA",
            "AAAAAAAAAAAAAAAAAAAA",
            "AAAAGAAAAAAAATAAAAAA",
            "AAAAAAAAAAAAATAGAAAA",
        ],
        "after_3": [
            {"seq": "GCCATAAACAAACCCACACC", "fitness": 9.0, "fitness_type": "float64"},
            {"seq": "GCCAAAATCAAACCCACGCA", "fitness": 8.0, "fitness_type": "float64"},
            {"seq": "GCGAAAAACAAACCCACACC", "fitness": 8.0, "fitness_type": "float64"},
            {"seq": "ACTAGCAACAACGCCACACT", "fitness": 8.0, "fitness_type": "float64"},
        ],
        "samples": [
            "GCCATAAACAAACCCACACC",
            "GCCAAAATCAAACCCACGCA",
            "GCGAAAAACAAACCCACACC",
            "ACTAGCAACAACGCCACACT",
        ],
    },
    "adabeam_positions": {
        "initial": [
            {"seq": "CAAAAACAAAAAAAAAAAAA", "fitness": 2.0, "fitness_type": "float64"},
            {"seq": "AAAAAAAAAAAAAAAAAAAA", "fitness": 0.0, "fitness_type": "float64"},
            {"seq": "AGAATAAAAAAAAAAAAAAA", "fitness": 0.0, "fitness_type": "float64"},
            {"seq": "AAAAATGAAAAAAAAAAAAA", "fitness": 0.0, "fitness_type": "float64"},
        ],
        "samples_init": [
            "CAAAAACAAAAAAAAAAAAA",
            "AAAAAAAAAAAAAAAAAAAA",
            "AGAATAAAAAAAAAAAAAAA",
            "AAAAATGAAAAAAAAAAAAA",
        ],
        "after_3": [
            {"seq": "CACCTCACAAAAAAAAAAAA", "fitness": 5.0, "fitness_type": "float64"},
            {"seq": "CGCTCCCTAAAAAAAAAAAA", "fitness": 5.0, "fitness_type": "float64"},
            {"seq": "CTCTCCCTAAAAAAAAAAAA", "fitness": 5.0, "fitness_type": "float64"},
            {"seq": "CTCGCACTAAAAAAAAAAAA", "fitness": 4.0, "fitness_type": "float64"},
        ],
        "samples": [
            "CACCTCACAAAAAAAAAAAA",
            "CGCTCCCTAAAAAAAAAAAA",
            "CTCTCCCTAAAAAAAAAAAA",
            "CTCGCACTAAAAAAAAAAAA",
        ],
    },
}


def _dump_nodes(nodes) -> list[dict]:
    return [
        {
            "seq": n.seq,
            "fitness": None if n.fitness is None else float(n.fitness),
            "fitness_type": type(n.fitness).__name__,
        }
        for n in nodes
    ]


def collect_adabeam_reference_snapshots() -> dict:
    """Collect characterization snapshots. Call with PYTHONHASHSEED=0."""
    model = testing_utils.CountLetterModel()

    def run(**kwargs):
        ab = AdaBeamReference(model_fn=model, **kwargs)
        initial = _dump_nodes(ab.current_nodes)
        samples_init = ab.get_samples(kwargs["beam_size"])
        ab.run(n_steps=3)
        return {
            "initial": initial,
            "samples_init": samples_init,
            "after_3": _dump_nodes(ab.current_nodes),
            "samples": ab.get_samples(kwargs["beam_size"]),
        }

    return {
        "adabeam_default": run(**_ADABEAM_SHARED, skip_repeat_sequences=False),
        "adabeam_skip_repeat": run(**_ADABEAM_SHARED, skip_repeat_sequences=True),
        "adabeam_positions": run(
            **_ADABEAM_SHARED,
            skip_repeat_sequences=False,
            positions_to_mutate=list(range(8)),
        ),
    }


@pytest.mark.parametrize("skip_repeat_sequences", [True, False])
def test_adabeam_reference_sanity(skip_repeat_sequences):
    model_fn = testing_utils.CountLetterModel()

    start_seq = "A" * 100
    start_score = model_fn([start_seq])[0]
    assert start_score == 0

    beam_size = 20
    kwargs = AdaBeamReference.debug_init_args()
    kwargs["model_fn"] = model_fn
    kwargs["start_sequence"] = start_seq
    kwargs["beam_size"] = beam_size
    kwargs["skip_repeat_sequences"] = skip_repeat_sequences
    adabeam = AdaBeamReference(**kwargs)

    adabeam.run(n_steps=2)

    out_seqs = adabeam.get_samples(beam_size)
    del out_seqs


def test_adabeam_reference_convergence():
    model_fn = testing_utils.CountLetterModel()

    start_seq = "A" * 100
    start_score = model_fn([start_seq])[0]
    assert start_score == 0

    kwargs = AdaBeamReference.debug_init_args()
    kwargs["model_fn"] = model_fn
    kwargs["start_sequence"] = start_seq
    kwargs["skip_repeat_sequences"] = True
    adabeam = AdaBeamReference(**kwargs)

    adabeam.run(n_steps=2)

    out_seqs = adabeam.get_samples(kwargs["beam_size"])
    out_seq_scores = np.array([model_fn([s])[0] for s in out_seqs])

    assert out_seq_scores[0] < start_score


def test_adabeam_reference_positions_to_mutate():
    """No matter how many iterations, positions outside `positions_to_mutate` shouldn't change."""
    model_fn = testing_utils.CountLetterModel()

    start_seq = "A" * 100
    start_score = model_fn([start_seq])[0]
    assert start_score == 0

    beam_size = 2
    kwargs = AdaBeamReference.debug_init_args()
    kwargs["model_fn"] = model_fn
    kwargs["start_sequence"] = start_seq
    kwargs["skip_repeat_sequences"] = True
    kwargs["beam_size"] = beam_size
    adabeam = AdaBeamReference(**kwargs, positions_to_mutate=list(range(20)))

    for _ in range(4):
        adabeam.run(n_steps=1)

        out_seqs = adabeam.get_samples(beam_size)
        for seq in out_seqs:
            for s in seq[20:]:
                assert s == "A", seq


@pytest.mark.parametrize("eval_batch_size", [1, 2, 4])
def test_adabeam_reference_eval_batch_size_sanity(eval_batch_size):
    """Test that `eval_batch_size` works."""
    model_fn = testing_utils.CountLetterModel()

    start_seq = "A" * 100
    start_score = model_fn([start_seq])[0]
    assert start_score == 0

    kwargs = AdaBeamReference.debug_init_args()
    kwargs["model_fn"] = model_fn
    kwargs["start_sequence"] = start_seq
    kwargs["eval_batch_size"] = eval_batch_size
    adabeam = AdaBeamReference(**kwargs)

    for _ in range(4):
        adabeam.run(n_steps=4)


def test_adabeam_reference_eval_batch_size_consistency():
    """Test that `eval_batch_size` is consistent."""
    model_fn = testing_utils.CountLetterModel()

    seqs = ["".join(random.choices(["A", "G", "T", "C"], k=100)) for _ in range(10)]

    kwargs = AdaBeamReference.debug_init_args()
    kwargs["model_fn"] = model_fn
    kwargs["start_sequence"] = "A" * 100

    kwargs["eval_batch_size"] = 1
    adabeam1 = AdaBeamReference(**kwargs)

    kwargs["eval_batch_size"] = 2
    adabeam2 = AdaBeamReference(**kwargs)

    kwargs["eval_batch_size"] = 4
    adabeam4 = AdaBeamReference(**kwargs)

    scores1 = adabeam1.get_batched_fitness(seqs)
    scores2 = adabeam2.get_batched_fitness(seqs)
    scores4 = adabeam4.get_batched_fitness(seqs)

    assert np.array_equal(scores1, scores2)
    assert np.array_equal(scores1, scores4)
    assert np.array_equal(scores2, scores4)


def test_adabeam_reference_matches_nucleobench_snapshots():
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
                "from gradabeam.adabeam_reference_test import "
                "collect_adabeam_reference_snapshots; "
                "print(json.dumps(collect_adabeam_reference_snapshots()))"
            ),
        ],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    got = json.loads(proc.stdout)
    assert got == _NUCLEOBENCH_ADABEAM_SNAPSHOTS


def test_reference_designers_are_exported_and_registered():
    import gradabeam
    from gradabeam.__main__ import _OPTIMIZERS

    assert gradabeam.AdaBeamReference is _OPTIMIZERS["adabeam-reference"]
    assert gradabeam.GradaBeamReference is _OPTIMIZERS["gradabeam-reference"]
    assert _OPTIMIZERS["adabeam"].__name__ == "AdaBeam"
    assert _OPTIMIZERS["gradabeam"].__name__ == "GradaBeam"
