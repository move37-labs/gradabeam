"""Tests for adabeam.py

To test:
```zsh
pytest gradabeam/adabeam_optimizer_test.py
```
"""

import json
import os
import random
import subprocess
import sys

import numpy as np
import pytest

from gradabeam import testing_utils
from gradabeam.adabeam_optimizer import AdaBeam


def _make_adabeam(**overrides):
    kwargs = AdaBeam.debug_init_args()
    kwargs.update(
        {
            "model_fn": testing_utils.CountLetterModel(),
            "start_sequence": "A" * 20,
            "beam_size": 4,
            "n_rollouts_per_root": 2,
            "mutations_per_sequence": 2.0,
            "skip_repeat_sequences": False,
            "rng_seed": 42,
        }
    )
    kwargs.update(overrides)
    return AdaBeam(**kwargs)


def _beam_rows(nodes):
    return [(n.seq, float(n.fitness)) for n in nodes]


@pytest.mark.parametrize("skip_repeat_sequences", [True, False])
def test_adabeam_sanity(skip_repeat_sequences):
    model_fn = testing_utils.CountLetterModel()

    start_seq = "A" * 100
    start_score = model_fn([start_seq])[0]
    assert start_score == 0

    beam_size = 20
    kwargs = AdaBeam.debug_init_args()
    kwargs["model_fn"] = model_fn
    kwargs["start_sequence"] = start_seq
    kwargs["beam_size"] = beam_size
    kwargs["skip_repeat_sequences"] = skip_repeat_sequences
    adabeam = AdaBeam(**kwargs)

    adabeam.run(n_steps=2)

    out_seqs = adabeam.get_samples(beam_size)
    del out_seqs


def test_adabeam_convergence():
    model_fn = testing_utils.CountLetterModel()

    start_seq = "A" * 100
    start_score = model_fn([start_seq])[0]
    assert start_score == 0

    kwargs = AdaBeam.debug_init_args()
    kwargs["model_fn"] = model_fn
    kwargs["start_sequence"] = start_seq
    kwargs["skip_repeat_sequences"] = True
    adabeam = AdaBeam(**kwargs)

    adabeam.run(n_steps=2)

    out_seqs = adabeam.get_samples(kwargs["beam_size"])
    out_seq_scores = np.array([model_fn([s])[0] for s in out_seqs])

    assert out_seq_scores[0] < start_score


def test_positions_to_mutate():
    """No matter how many iterations, positions outside `positions_to_mutate` shouldn't change."""
    model_fn = testing_utils.CountLetterModel()

    start_seq = "A" * 100
    start_score = model_fn([start_seq])[0]
    assert start_score == 0

    beam_size = 2
    kwargs = AdaBeam.debug_init_args()
    kwargs["model_fn"] = model_fn
    kwargs["start_sequence"] = start_seq
    kwargs["skip_repeat_sequences"] = True
    kwargs["beam_size"] = beam_size
    adabeam = AdaBeam(**kwargs, positions_to_mutate=list(range(20)))

    for _ in range(4):
        adabeam.run(n_steps=1)

        out_seqs = adabeam.get_samples(beam_size)
        for seq in out_seqs:
            for s in seq[20:]:
                assert s == "A", seq


@pytest.mark.parametrize("eval_batch_size", [1, 2, 4])
def test_eval_batch_size_sanity(eval_batch_size):
    """Test that `eval_batch_size` works."""
    model_fn = testing_utils.CountLetterModel()

    start_seq = "A" * 100
    start_score = model_fn([start_seq])[0]
    assert start_score == 0

    kwargs = AdaBeam.debug_init_args()
    kwargs["model_fn"] = model_fn
    kwargs["start_sequence"] = start_seq
    kwargs["eval_batch_size"] = eval_batch_size
    adabeam = AdaBeam(**kwargs)

    for _ in range(4):
        adabeam.run(n_steps=4)

        # TODO(joelshor):
        # Add correctness checks.


def test_eval_batch_size_consistency():
    """Test that `eval_batch_size` is consistent."""
    model_fn = testing_utils.CountLetterModel()

    seqs = ["".join(random.choices(["A", "G", "T", "C"], k=100)) for _ in range(10)]

    kwargs = AdaBeam.debug_init_args()
    kwargs["model_fn"] = model_fn
    kwargs["start_sequence"] = "A" * 100

    kwargs["eval_batch_size"] = 1
    adabeam1 = AdaBeam(**kwargs)

    kwargs["eval_batch_size"] = 2
    adabeam2 = AdaBeam(**kwargs)

    kwargs["eval_batch_size"] = 4
    adabeam4 = AdaBeam(**kwargs)

    scores1 = adabeam1.get_batched_fitness(seqs)
    scores2 = adabeam2.get_batched_fitness(seqs)
    scores4 = adabeam4.get_batched_fitness(seqs)

    assert np.array_equal(scores1, scores2)
    assert np.array_equal(scores1, scores4)
    assert np.array_equal(scores2, scores4)


def test_adabeam_determinism_and_diversity():
    """Test that runs with the same seed are deterministic and different seeds produce different diversity."""
    model_fn = testing_utils.CountLetterModel()
    start_seq = "A" * 100

    kwargs1 = AdaBeam.debug_init_args()
    kwargs1["model_fn"] = model_fn
    kwargs1["start_sequence"] = start_seq
    kwargs1["rng_seed"] = 42
    adabeam1 = AdaBeam(**kwargs1)
    adabeam1.run(n_steps=3)
    out1 = adabeam1.get_samples(10)

    kwargs2 = AdaBeam.debug_init_args()
    kwargs2["model_fn"] = model_fn
    kwargs2["start_sequence"] = start_seq
    kwargs2["rng_seed"] = 42
    adabeam2 = AdaBeam(**kwargs2)
    adabeam2.run(n_steps=3)
    out2 = adabeam2.get_samples(10)

    assert out1 == out2

    kwargs3 = AdaBeam.debug_init_args()
    kwargs3["model_fn"] = model_fn
    kwargs3["start_sequence"] = start_seq
    kwargs3["rng_seed"] = 43
    adabeam3 = AdaBeam(**kwargs3)
    adabeam3.run(n_steps=3)
    out3 = adabeam3.get_samples(10)

    assert out1 != out3


def test_skip_repeat_retries_until_uncached():
    opt = _make_adabeam(skip_repeat_sequences=True, beam_size=2)
    parent = opt.current_nodes[0]
    cached = parent.seq
    calls = {"n": 0}

    def fake_generate(_sequence: str, _n: int) -> str:
        calls["n"] += 1
        return cached if calls["n"] == 1 else "T" * len(cached)

    opt.generate_mutations = fake_generate  # type: ignore[method-assign]
    child = opt.mutate_nodes([parent], [1])[0]
    assert calls["n"] == 2
    assert child.seq == "T" * len(cached)


def test_init_beam_is_ranked_and_get_samples_is_pure():
    opt = _make_adabeam()
    fitnesses = [float(n.fitness) for n in opt.current_nodes]
    assert fitnesses == sorted(fitnesses, reverse=True)
    mut_before = opt.rng.bit_generator.state
    tie_before = opt.tie_rng.bit_generator.state
    assert opt.get_samples(len(opt.current_nodes)) == [n.seq for n in opt.current_nodes]
    assert opt.rng.bit_generator.state == mut_before
    assert opt.tie_rng.bit_generator.state == tie_before


def test_get_samples_does_not_change_later_search():
    opt = _make_adabeam()
    opt.run(n_steps=1)
    mut_before = opt.rng.bit_generator.state
    tie_before = opt.tie_rng.bit_generator.state
    first = opt.get_samples(3)
    second = opt.get_samples(3)
    assert first == second
    assert opt.rng.bit_generator.state == mut_before
    assert opt.tie_rng.bit_generator.state == tie_before

    after_observe = _make_adabeam()
    after_observe.run(n_steps=1)
    after_observe.get_samples(3)
    after_observe.run(n_steps=1)
    control = _make_adabeam()
    control.run(n_steps=2)
    assert _beam_rows(after_observe.current_nodes) == _beam_rows(control.current_nodes)


_HASHSEED_SCRIPT = r"""
import json
from gradabeam.adabeam_optimizer import AdaBeam

opt = AdaBeam(**AdaBeam.debug_init_args())
opt.run(n_steps=2)
print(json.dumps(opt.get_samples(4)))
"""


def _run_adabeam_with_hashseed(hashseed: str) -> list[str]:
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = hashseed
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    env["PYTHONPATH"] = os.pathsep.join(
        [
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
            env.get("PYTHONPATH", ""),
        ]
    )
    proc = subprocess.run(
        [sys.executable, "-c", _HASHSEED_SCRIPT],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_seeded_run_matches_across_pythonhashseed():
    first = _run_adabeam_with_hashseed("0")
    second = _run_adabeam_with_hashseed("1")
    third = _run_adabeam_with_hashseed("random")
    assert first == second == third
