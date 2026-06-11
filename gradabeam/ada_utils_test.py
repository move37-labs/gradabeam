"""Tests for [Gr]AdaBeam utils.

To test:
```zsh
pytest gradabeam/ada_utils_test.py
```
"""
# TODO(joelshor): Write test for `get_tism_edits_and_probs`.

import numpy as np
import pytest

from gradabeam import constants
from gradabeam import testing_utils
from gradabeam import ada_utils


# (sequence length, mutation rate)
PARAMS_TO_TEST_ = [
    (10, 0.1),
    (10, 0.5),
    (10, 0.7),
    (100, 0.1),
    (10, 0.5),
    (10, 0.7),
]

LIKELIHOOD_FNS_ = [ada_utils.num_edits_likelihood_adabeam]


@pytest.mark.parametrize("likelihood_fn", LIKELIHOOD_FNS_)
def test_num_edits_likelihood_legacy_prob_dist(likelihood_fn):
    for sequence_length, mutation_rate in PARAMS_TO_TEST_:
        actual_sum = np.sum(
            likelihood_fn(
                np.arange(sequence_length + 1), sequence_length, mutation_rate
            )
        )
        expected_sum = 1.0
        np.testing.assert_allclose(actual_sum, expected_sum)


@pytest.mark.parametrize("likelihood_fn", LIKELIHOOD_FNS_)
def test_num_edits_sampler(likelihood_fn, num_samples=150000, atol=0.002):
    for sequence_length, mutation_rate in PARAMS_TO_TEST_:
        num_edits_sampler = ada_utils.NumberEditsSampler(
            sequence_length, mutation_rate, likelihood_fn=likelihood_fn, rng_seed=1
        )

        num_edits = num_edits_sampler.sample(num_samples)
        possible_num_edits = np.arange(1, sequence_length + 1)

        actual_probs = [
            float(np.count_nonzero(num_edits == n)) / len(num_edits)
            for n in possible_num_edits
        ]
        expected_probs = likelihood_fn(
            possible_num_edits, sequence_length, mutation_rate
        )

        np.testing.assert_allclose(actual_probs, expected_probs, atol=atol)


def expected_num_edits(sequence_len: int, mutation_rate: float) -> float:
    """A conceptually simpler expectation calculation for testing."""
    F_inverse = ada_utils._F_inverse(mutation_rate, sequence_len)
    return sequence_len * mutation_rate / F_inverse


def test_expected_num_edits(num_samples=150000, atol=0.002):
    """Tests that the expected number of edits is correct."""
    likelihood_fn = ada_utils.num_edits_likelihood_adabeam
    expected_num_edits_fn = expected_num_edits
    for sequence_length, mutation_rate in PARAMS_TO_TEST_:
        num_edits_sampler = ada_utils.NumberEditsSampler(
            sequence_length, mutation_rate, likelihood_fn=likelihood_fn, rng_seed=1
        )
        actual = np.mean(num_edits_sampler.sample(num_samples))
        expected = expected_num_edits_fn(sequence_length, mutation_rate)

        np.testing.assert_allclose(actual, expected, atol=atol)


def test_no_tism_cost_fail():
    model_fn = testing_utils.CountLetterModel(
        target_char="A",
    )

    model = ada_utils.ModelWrapper(model_fn)
    with pytest.raises(ValueError):
        model.get_tism("ACAAA", idxs=None)


@pytest.mark.parametrize("idx_option", [None, "all", "skipC", "includeC"])
def test_get_tisms_basic(idx_option):
    """Test basic functionality of get_tism."""
    model_fn = testing_utils.CountLetterModel(
        target_char="A",  # counts 'A's, so more A's = higher score
    )
    idxs = {
        None: None,
        "all": list(range(5)),
        "skipC": [0, 3, 4],
        "includeC": [0, 1, 2],
    }[idx_option]

    model = ada_utils.ModelWrapper(model_fn, tism_cost=1.0)
    sequence = "ACAAA"

    pos_and_chars, logits = model.get_tism(
        sequence=sequence,
        idxs=idxs,
    )

    # Check return types
    assert isinstance(pos_and_chars, list)
    assert isinstance(logits, np.ndarray)
    assert logits.dtype == np.float32

    # Check structure
    assert len(pos_and_chars) == len(logits)
    for pos, char in pos_and_chars:
        assert isinstance(pos, int)
        assert isinstance(char, str)
        assert char in ["A", "C", "G", "T"]

    # Check that we don't include mutations to the same character
    if idxs is None:
        positions_to_check = list(range(len(sequence)))
    else:
        positions_to_check = idxs

    for pos in positions_to_check:
        base_char = sequence[pos]
        # Should not have (pos, base_char) in results
        assert (pos, base_char) not in pos_and_chars, (
            f"Position {pos} should not have mutation to its own base '{base_char}'"
        )

    # Check that we get expected number of mutations
    # For each position, we should have 3 mutations (4 vocab - 1 base)
    expected_num_mutations = len(positions_to_check) * 3
    assert len(pos_and_chars) == expected_num_mutations, (
        f"Expected {expected_num_mutations} mutations, got {len(pos_and_chars)}"
    )


def test_get_tisms_with_idxs():
    """Test get_tisms with specific indices."""
    model_fn = testing_utils.CountLetterModel(
        target_char="A",
    )

    model = ada_utils.ModelWrapper(model_fn, tism_cost=1.0)
    sequence = "ACAAA"
    idxs = [0, 2, 4]  # Only check positions 0, 2, 4

    pos_and_chars, logits = model.get_tism(
        sequence=sequence,
        idxs=idxs,
    )

    # Check that all positions in results are from idxs
    result_positions = {pos for pos, _ in pos_and_chars}
    assert result_positions.issubset(set(idxs)), (
        f"All result positions should be in {idxs}, but got {result_positions}"
    )

    # Check that we have mutations for all specified positions
    for pos in idxs:
        # Should have 3 mutations per position (4 vocab - 1 base)
        mutations_at_pos = [p for p, _ in pos_and_chars if p == pos]
        assert len(mutations_at_pos) == 3, (
            f"Position {pos} should have 3 mutations, got {len(mutations_at_pos)}"
        )


# ---------------------------------------------------------------------------
# Item 1 — Real get_tism ordering (breaks the circularity of hand-built tests)
# ---------------------------------------------------------------------------


def test_get_tism_real_action_ordering():
    """Verify the real get_tism emits actions in positions-major, VOCAB-order, reference-removed order.

    This test calls the actual ModelWrapper.get_tism (backed by TISMModelClass)
    on a short sequence and asserts the structural invariants that the action-space
    sampler's reshape(-1, 3) depends on:

      1. Total length == 3 * L  (one reference base removed per position).
      2. Actions for position i occupy the contiguous slice [3i : 3i+3]
         (positions-major layout).
      3. Within each group, bases appear in the expected VOCAB order with the
         reference base skipped.  The expected values are HARDCODED below (not
         derived from constants.VOCAB) so a reordering of VOCAB would move the
         implementation and the derived expectation in lockstep but would still
         fail this test.

    If either 2 or 3 fails, stop — the reshape(-1, 3) in
    generate_random_mutant_actionspace is wrong.
    """
    model_fn = testing_utils.CountLetterModel(target_char="A")
    model = ada_utils.ModelWrapper(model_fn, tism_cost=1.0)

    # Use a sequence where every position has a different reference base so
    # the masking logic at each position is exercised independently.
    sequence = "ACGT"

    pos_and_chars, _ = model.get_tism(sequence=sequence, idxs=None)

    n_positions = len(sequence)
    assert len(pos_and_chars) == 3 * n_positions, (
        f"Expected 3*{n_positions}={3 * n_positions} actions, got {len(pos_and_chars)}. "
        "action-space reshape(-1, 3) would fail."
    )

    # Hardcoded per-position expected bases in VOCAB order ["A","C","G","T"]
    # with each position's reference base removed.  These are literal constants,
    # not derived from constants.VOCAB, so they catch a VOCAB reordering.
    #   pos 0: ref='A' → non-ref in VOCAB order = C, G, T
    #   pos 1: ref='C' → non-ref in VOCAB order = A, G, T
    #   pos 2: ref='G' → non-ref in VOCAB order = A, C, T
    #   pos 3: ref='T' → non-ref in VOCAB order = A, C, G
    expected_bases_per_position = {
        0: ["C", "G", "T"],
        1: ["A", "G", "T"],
        2: ["A", "C", "T"],
        3: ["A", "C", "G"],
    }

    for i in range(n_positions):
        abs_pos = i  # mutable_positions = range(L) when idxs=None
        expected_bases = expected_bases_per_position[i]

        group = pos_and_chars[3 * i : 3 * i + 3]

        # --- Invariant 2: positions-major ---
        actual_positions = [p for p, _ in group]
        assert all(p == abs_pos for p in actual_positions), (
            f"Position group {i}: expected all entries to be position {abs_pos}, "
            f"got positions {actual_positions}. "
            "Ordering is NOT positions-major — reshape(-1, 3) is WRONG. Stop."
        )

        # --- Invariant 3: VOCAB order, reference removed (hardcoded expectation) ---
        actual_bases = [c for _, c in group]
        assert actual_bases == expected_bases, (
            f"Position {abs_pos}: expected bases {expected_bases} "
            f"in VOCAB order, got {actual_bases}. "
            "Within-group ordering assumption is WRONG — stop, do not proceed with integration."
        )


# ---------------------------------------------------------------------------
# Tests for generate_random_mutant_actionspace and build_uniform_pos_and_chars
# ---------------------------------------------------------------------------

# Shared fixture data
_SEQUENCE = "ACGTACGT"  # length 8
_POS_AND_CHARS = ada_utils.build_uniform_pos_and_chars(_SEQUENCE, list(range(len(_SEQUENCE))))
_UNIFORM_PROBS = np.ones(len(_POS_AND_CHARS), dtype=np.float64) / len(_POS_AND_CHARS)


def test_build_uniform_pos_and_chars():
    """Verify build_uniform_pos_and_chars builds correct 3L actions in positions-major order."""
    pos_and_chars = ada_utils.build_uniform_pos_and_chars("AC", list(range(2)))
    # Length of AC is 2. 3*2 = 6 actions.
    # Pos 0 (A): non-ref are C, G, T
    # Pos 1 (C): non-ref are A, G, T
    expected = [
        (0, "C"), (0, "G"), (0, "T"),
        (1, "A"), (1, "G"), (1, "T"),
    ]
    assert pos_and_chars == expected


@pytest.mark.parametrize("n_edits", [1, 2, 3, 5])
@pytest.mark.parametrize("seed", [0, 1, 42, 99, 137])
def test_actionspace_edit_count_invariant(n_edits, seed):
    """Mutant differs from input in exactly n_edits positions."""
    rng = np.random.default_rng(seed)
    mutant, edited_indices, remaining_probs, p_final_chosen_list = ada_utils.generate_random_mutant_actionspace(
        sequence=_SEQUENCE,
        pos_and_chars_to_mutate=_POS_AND_CHARS,
        n_edits=n_edits,
        rng=rng,
        probs=_UNIFORM_PROBS.copy(),
    )

    # Exactly n_edits edits.
    actual_diffs = [i for i, (a, b) in enumerate(zip(_SEQUENCE, mutant)) if a != b]
    assert len(actual_diffs) == n_edits, (
        f"Expected {n_edits} edits, got {len(actual_diffs)}: {actual_diffs}"
    )

    # No silent edit: every changed position has a different base.
    for pos in actual_diffs:
        assert mutant[pos] != _SEQUENCE[pos], (
            f"Silent edit at position {pos}: base unchanged ({_SEQUENCE[pos]})."
        )

    # All selected action indices are valid
    assert len(edited_indices) == n_edits
    for idx in edited_indices:
        assert 0 <= idx < len(_POS_AND_CHARS)

    # Positional masking was applied correctly: the remaining_probs for the selected indices' positions should be zeroed
    for idx in edited_indices:
        selected_pos, _ = _POS_AND_CHARS[idx]
        # All actions for this position should be 0.0 in remaining_probs
        for a_idx, (pos, _) in enumerate(_POS_AND_CHARS):
            if pos == selected_pos:
                assert remaining_probs[a_idx] == 0.0


def test_actionspace_collision_regression():
    """Verify that action-space masking prevents position collisions."""
    sequence = "ACGTACGT"  # length 8
    n_edits = 2

    # Put heavy mass on position 0's actions
    pos_and_chars = ada_utils.build_uniform_pos_and_chars(sequence, list(range(len(sequence))))
    n_actions = len(pos_and_chars)
    probs = np.zeros(n_actions, dtype=np.float64)
    # Position 0 occupies indices [0, 1, 2]
    probs[0] = 0.47
    probs[1] = 0.47
    probs[2] = 0.02
    # Spread remaining mass
    remaining = 1.0 - probs.sum()
    probs[3:] = remaining / (n_actions - 3)
    probs /= probs.sum()

    n_seeds = 100
    actionspace_distinct_counts = []

    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        mutant_act, _, _, _ = ada_utils.generate_random_mutant_actionspace(
            sequence=sequence,
            pos_and_chars_to_mutate=pos_and_chars,
            n_edits=n_edits,
            rng=rng,
            probs=probs.copy(),
        )
        distinct_act = sum(1 for a, b in zip(sequence, mutant_act) if a != b)
        actionspace_distinct_counts.append(distinct_act)

    # No collisions ever: always exactly n_edits edits
    assert all(c == n_edits for c in actionspace_distinct_counts), (
        f"generate_random_mutant_actionspace produced collisions/wrong edit counts: "
        f"{set(actionspace_distinct_counts)}"
    )


def test_actionspace_guard_n_edits_zero():
    """n_edits=0 must raise AssertionError."""
    with pytest.raises(AssertionError, match="n_edits"):
        ada_utils.generate_random_mutant_actionspace(
            sequence="ACGT",
            pos_and_chars_to_mutate=_POS_AND_CHARS[:12],
            n_edits=0,
            rng=np.random.default_rng(0),
            probs=np.ones(12) / 12,
        )


def test_actionspace_boundary_exceeds_positions():
    """n_edits exceeding available positions gets clamped to total positions, but doesn't crash."""
    sequence = "ACGT"  # length 4
    pos_and_chars = ada_utils.build_uniform_pos_and_chars(sequence, list(range(len(sequence))))
    # 12 actions total across 4 positions.
    # Max positions to edit is 4.
    rng = np.random.default_rng(0)
    mutant, edited_indices, remaining_probs, p_final_chosen_list = ada_utils.generate_random_mutant_actionspace(
        sequence=sequence,
        pos_and_chars_to_mutate=pos_and_chars,
        n_edits=5,  # Exceeds total positions (4)
        rng=rng,
        probs=np.ones(12) / 12,
    )
    # It should be clamped to 4 edits (one per position)
    actual_diffs = [i for i, (a, b) in enumerate(zip(sequence, mutant)) if a != b]
    assert len(actual_diffs) == 4
    assert len(edited_indices) == 4


def test_actionspace_guard_negative_probs():
    """Any negative probability must raise AssertionError."""
    with pytest.raises(AssertionError, match="nonnegative"):
        probs = np.ones(24) / 24
        probs[3] = -0.1
        ada_utils.generate_random_mutant_actionspace(
            sequence=_SEQUENCE,
            pos_and_chars_to_mutate=_POS_AND_CHARS,
            n_edits=1,
            rng=np.random.default_rng(0),
            probs=probs,
        )


def test_actionspace_guard_allzero_probs():
    """All-zero probabilities must raise AssertionError."""
    with pytest.raises(AssertionError, match="least one"):
        ada_utils.generate_random_mutant_actionspace(
            sequence=_SEQUENCE,
            pos_and_chars_to_mutate=_POS_AND_CHARS,
            n_edits=1,
            rng=np.random.default_rng(0),
            probs=np.zeros(24),
        )


def test_actionspace_weights_bias_selection():
    """Verify weights bias action selection."""
    sequence = "ACGTACGT"  # length 8
    pos_and_chars = ada_utils.build_uniform_pos_and_chars(sequence, list(range(len(sequence))))
    # Let's put 80% weight on action 5 (position 1, base 'T' or similar) and spread the rest
    hot_idx = 5
    probs = np.ones(len(pos_and_chars), dtype=np.float64)
    probs[hot_idx] = 80.0
    expected_rate = probs[hot_idx] / probs.sum()

    n_trials = 1000
    hot_selected = 0
    for seed in range(n_trials):
        rng = np.random.default_rng(seed)
        _, edited_indices, _, _ = ada_utils.generate_random_mutant_actionspace(
            sequence=sequence,
            pos_and_chars_to_mutate=pos_and_chars,
            n_edits=1,
            rng=rng,
            probs=probs.copy(),
        )
        if hot_idx in edited_indices:
            hot_selected += 1

    observed_rate = hot_selected / n_trials
    sigma = np.sqrt(expected_rate * (1 - expected_rate) / n_trials)
    atol = 4 * sigma
    assert abs(observed_rate - expected_rate) < atol, (
        f"Selection rate {observed_rate:.4f} is not within {atol:.4f} of expected {expected_rate:.4f}."
    )


# ---------------------------------------------------------------------------
# REGRESSION GUARD for the α-posterior
# ---------------------------------------------------------------------------


def test_pfinal_equals_step_start_action_probability():
    """REGRESSION GUARD for the α-posterior.

    p_final for each chosen edit must be the STEP-START policy probability of that
    action, probs[action] — NOT the decomposed factorial p_pos*p_base, and NOT a
    within-child without-replacement renormalized value. See the design comment in
    generate_random_mutant_actionspace for the rationale.
    """
    sequence = "AAAA"
    pos_and_chars = ada_utils.build_uniform_pos_and_chars(sequence, [0, 1, 2, 3])  # 12 actions
    # Non-uniform probs so the factorial reconstruction != probs[action] in the interior.
    probs = np.array([0.30, 0.02, 0.02,
                      0.05, 0.05, 0.05,
                      0.10, 0.01, 0.04,
                      0.10, 0.10, 0.16], dtype=np.float64)
    probs /= probs.sum()

    rng = np.random.default_rng(0)
    _, sel_idx, _, p_final = ada_utils.generate_random_mutant_actionspace(
        sequence=sequence,
        pos_and_chars_to_mutate=pos_and_chars,
        n_edits=3,
        rng=rng,
        probs=probs.copy(),
    )

    expected = [probs[a] for a in sel_idx]   # step-start policy prob; NO renormalization
    np.testing.assert_allclose(
        p_final, expected, rtol=1e-9, atol=0,
        err_msg="p_final is not the step-start action probability probs[action].",
    )
