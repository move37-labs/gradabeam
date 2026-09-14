"""Tests for gradabeam_optimizer.py

To test:
```zsh
pytest gradabeam/gradabeam_optimizer_test.py
```
"""

import json
import os
import subprocess
import sys

import numpy as np
import pytest

from gradabeam import ada_utils, testing_utils
from gradabeam.gradabeam_optimizer import GradaBeam, RolloutNodeWithProbs


def _make_gradabeam(**overrides):
    kwargs = GradaBeam.debug_init_args()
    kwargs.update(
        {
            "model_fn": testing_utils.CountLetterModel(),
            "start_sequence": "A" * 16,
            "beam_size": 3,
            "n_rollouts_per_root": 2,
            "mutations_per_sequence": 1.0,
            "exploration_alpha": 0.05,
            "use_pbt": True,
            "rng_seed": 42,
        }
    )
    kwargs.update(overrides)
    return GradaBeam(**kwargs)


def _beam_rows(nodes):
    return [
        (
            n.seq,
            float(n.fitness),
            n.edits_since_root,
            float(n.mutations_per_sequence),
            float(n.exploration_alpha),
        )
        for n in nodes
    ]


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


def test_gradabeam_rejects_eval_batch_size_gt_1():
    kwargs = GradaBeam.debug_init_args()
    kwargs["eval_batch_size"] = 2
    with pytest.raises(ValueError, match="eval_batch_size=1"):
        GradaBeam(**kwargs)


def test_gradabeam_eval_batch_size_consistency():
    """Batched fitness scoring is independent of the helper chunk size."""
    model_fn = testing_utils.CountLetterModel()
    seqs = [
        "".join(np.random.choice(["A", "G", "T", "C"], size=100)) for _ in range(10)
    ]
    kwargs = GradaBeam.debug_init_args()
    kwargs["model_fn"] = model_fn
    kwargs["start_sequence"] = "A" * 100
    opt = GradaBeam(**kwargs)
    scores = [
        ada_utils.get_batched_fitness(opt.model, seqs, batch_size=b) for b in (1, 2, 4)
    ]
    assert np.array_equal(scores[0], scores[1])
    assert np.array_equal(scores[0], scores[2])


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
        seed = RolloutNodeWithProbs(
            seq=start_sequence,
            fitness=0.0,
            edits_since_root=0,
            mutations_per_sequence=1.0,
            exploration_alpha=0.0,
        )
        nodes = gb.initialize_roots_with_gradients([seed])
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

        # ASSERTION 2: Is C-mass concentrated? On "AA" both positions are
        # equally good, so a single action may be below 0.5.
        c_mass = float(
            sum(
                p
                for (_pos, ch), p in zip(root.pos_and_chars, root.probs)
                if ch == target_char
            )
        )
        assert c_mass >= 0.5, (
            f"Entropy Death! Expected high C-mass (>0.5), got {c_mass:.4f} "
            f"(top action {best_prob:.4f})."
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


def test_cli_rejects_gradabeam_eval_batch_size_gt_1():
    from pathlib import Path

    from gradabeam.__main__ import main

    oracle = Path(__file__).resolve().parents[1] / "oracles" / "count_letter.py"
    with pytest.raises(SystemExit):
        main(
            [
                "--optimizer",
                "gradabeam",
                "--oracle_script",
                str(oracle),
                "--start_sequence",
                "AAAA",
                "--n_steps",
                "1",
                "--beam_size",
                "1",
                "--mutations_per_sequence",
                "1",
                "--n_rollouts_per_root",
                "1",
                "--use_pbt",
                "False",
                "--eval_batch_size",
                "2",
            ]
        )


def test_oracle_template_exposes_tism_torch():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "oracles" / "template.py"
    spec = importlib.util.spec_from_file_location("oracle_template", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    oracle = module.make_oracle()
    assert hasattr(oracle, "tism_torch")
    scores = oracle(["AAAA", "CCCC"])
    assert scores[1] < scores[0]
    opt = GradaBeam(
        model_fn=oracle,
        start_sequence="AAAA",
        mutations_per_sequence=1,
        beam_size=2,
        n_rollouts_per_root=1,
        exploration_alpha=0.0,
        use_pbt=False,
        rng_seed=0,
    )
    opt.run(n_steps=1)
    assert opt.get_samples(1)


def test_selected_action_masking_leaves_sibling_actions():
    opt = _make_gradabeam(use_pbt=False, n_rollouts_per_root=1)
    parent = opt.initialize_roots_with_gradients([opt.current_nodes[0]])[0]
    child = opt.mutate_nodes_gradabeam([parent], [1], [parent.mutations_per_sequence])[
        0
    ]
    zeroed = np.where(child.probs == 0.0)[0]
    assert len(zeroed) == 1
    zeroed_pos = parent.pos_and_chars[int(zeroed[0])][0]
    sibling_live = [
        i
        for i, (pos, _ch) in enumerate(parent.pos_and_chars)
        if pos == zeroed_pos and child.probs[i] > 0
    ]
    assert sibling_live


def test_alpha_posterior_uses_uniform_over_action_space():
    opt = _make_gradabeam(use_pbt=True, n_rollouts_per_root=1)
    parent = opt.initialize_roots_with_gradients([opt.current_nodes[0]])[0]
    child = opt.mutate_nodes_gradabeam([parent], [1], [2.0])[0]
    selected = np.where(child.probs == 0.0)[0]
    p_uniform = 1.0 / len(parent.probs)
    expected = float(
        np.clip(
            np.mean(
                (parent.exploration_alpha * p_uniform)
                / (parent.probs[selected] + 1e-10)
            ),
            0.01,
            0.99,
        )
    )
    assert child.exploration_alpha == pytest.approx(expected)


def test_exhausted_action_mass_resets_to_uniform():
    opt = _make_gradabeam(use_pbt=False, n_rollouts_per_root=1)
    parent = opt.initialize_roots_with_gradients([opt.current_nodes[0]])[0]
    exhausted = np.zeros_like(parent.probs)
    exhausted[0] = 1.0
    parent = RolloutNodeWithProbs(
        seq=parent.seq,
        fitness=parent.fitness,
        edits_since_root=parent.edits_since_root,
        probs=exhausted,
        pos_and_chars=parent.pos_and_chars,
        mutations_per_sequence=parent.mutations_per_sequence,
        exploration_alpha=parent.exploration_alpha,
    )
    child = opt.mutate_nodes_gradabeam([parent], [1], [1.0])[0]
    np.testing.assert_allclose(
        child.probs, np.full_like(child.probs, 1.0 / len(child.probs))
    )


def test_get_sampler_is_cached_per_rate():
    opt = _make_gradabeam()
    first = opt.get_sampler(1.0)
    assert opt.get_sampler(1.0) is first
    assert opt.get_sampler(2.0) is not first


def test_tism_cache_preserves_distinct_pbt_state():
    opt = _make_gradabeam(n_rollouts_per_root=1)
    seq = opt.current_nodes[0].seq
    left = RolloutNodeWithProbs(
        seq=seq,
        fitness=1.0,
        edits_since_root=0,
        mutations_per_sequence=1.0,
        exploration_alpha=0.05,
    )
    right = RolloutNodeWithProbs(
        seq=seq,
        fitness=1.0,
        edits_since_root=0,
        mutations_per_sequence=3.0,
        exploration_alpha=0.9,
    )
    cache = {}
    tism_before = opt.model.tism_calls
    out = opt.initialize_roots_with_gradients([left, right], tism_cache=cache)
    assert opt.model.tism_calls == tism_before + 1
    assert out[0].mutations_per_sequence == 1.0
    assert out[1].mutations_per_sequence == 3.0
    assert out[0].exploration_alpha == 0.05
    assert out[1].exploration_alpha == 0.9
    assert not np.array_equal(out[0].probs, out[1].probs)


def test_fitness_is_python_float():
    opt = _make_gradabeam()
    assert type(opt.current_nodes[0].fitness) is float


def test_rollout_node_identity_includes_pbt_fields():
    a = RolloutNodeWithProbs(
        seq="AAAA",
        fitness=1.0,
        edits_since_root=1,
        mutations_per_sequence=1.0,
        exploration_alpha=0.05,
    )
    b = RolloutNodeWithProbs(
        seq="AAAA",
        fitness=1.0,
        edits_since_root=1,
        mutations_per_sequence=2.0,
        exploration_alpha=0.05,
    )
    c = RolloutNodeWithProbs(
        seq="AAAA",
        fitness=1.0,
        edits_since_root=2,
        mutations_per_sequence=1.0,
        exploration_alpha=0.05,
    )
    assert a != b
    assert hash(a) != hash(b)
    assert a != c


def test_init_ranks_beam_and_scores_tism_once():
    opt = _make_gradabeam()
    fitnesses = [float(n.fitness) for n in opt.current_nodes]
    assert fitnesses == sorted(fitnesses, reverse=True)
    assert opt.model.tism_calls == 1
    mut_before = opt.rng.bit_generator.state
    tie_before = opt.tie_rng.bit_generator.state
    assert opt.get_samples(len(opt.current_nodes)) == [n.seq for n in opt.current_nodes]
    assert opt.rng.bit_generator.state == mut_before
    assert opt.tie_rng.bit_generator.state == tie_before


def test_get_samples_does_not_advance_rngs():
    opt = _make_gradabeam()
    opt.run(n_steps=1)
    mut_before = opt.rng.bit_generator.state
    tie_before = opt.tie_rng.bit_generator.state
    first = opt.get_samples(2)
    second = opt.get_samples(2)
    assert first == second
    assert opt.rng.bit_generator.state == mut_before
    assert opt.tie_rng.bit_generator.state == tie_before

    observed = _make_gradabeam()
    observed.run(n_steps=1)
    observed.get_samples(2)
    observed.run(n_steps=1)
    control = _make_gradabeam()
    control.run(n_steps=2)
    assert _beam_rows(observed.current_nodes) == _beam_rows(control.current_nodes)


_HASHSEED_SCRIPT = r"""
import json
from gradabeam.gradabeam_optimizer import GradaBeam

opt = GradaBeam(**GradaBeam.debug_init_args())
opt.run(n_steps=2)
print(json.dumps(opt.get_samples(4)))
"""


def _run_gradabeam_with_hashseed(hashseed: str) -> list[str]:
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
    first = _run_gradabeam_with_hashseed("0")
    second = _run_gradabeam_with_hashseed("1")
    third = _run_gradabeam_with_hashseed("random")
    assert first == second == third
