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

    All other tests that rely on tism_probs_to_position_weights build their
    synthetic pos_and_chars by hand using the same assumed ordering — that is
    circular.  This test calls the actual ModelWrapper.get_tism (backed by
    TISMModelClass) on a short sequence and asserts the two structural
    invariants that tism_probs_to_position_weights.reshape(-1, 3) depends on:

      1. Total length == 3 * L  (one reference base removed per position).
      2. Actions for position i occupy the contiguous slice [3i : 3i+3]
         (positions-major layout).
      3. Within each group, bases appear in the expected VOCAB order with the
         reference base skipped.  The expected values are HARDCODED below (not
         derived from constants.VOCAB) so a reordering of VOCAB would move the
         implementation and the derived expectation in lockstep but would still
         fail this test.

    If either 2 or 3 fails, stop — the marginalizer is wrong and Plan 01 part
    1b is blocked.
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
        "tism_probs_to_position_weights length assertion would fail."
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
# Tests for generate_random_mutant_positionspace and tism_probs_to_position_weights
# ---------------------------------------------------------------------------

# Shared fixture data
_SEQUENCE = "ACGTACGT"  # length 8
_MUTABLE_POSITIONS = list(range(8))
_UNIFORM_WEIGHTS = np.ones(8, dtype=np.float64)


@pytest.mark.parametrize("n_edits", [1, 2, 3, 5])
@pytest.mark.parametrize("seed", [0, 1, 42, 99, 137])
def test_positionspace_edit_count_invariant(n_edits, seed):
    """Mutant differs from input in exactly n_edits positions, all within mutable_positions."""
    rng = np.random.default_rng(seed)
    mutant, edited_positions = ada_utils.generate_random_mutant_positionspace(
        sequence=_SEQUENCE,
        mutable_positions=_MUTABLE_POSITIONS,
        position_weights=_UNIFORM_WEIGHTS,
        n_edits=n_edits,
        rng=rng,
    )

    # Exactly n_edits edits.
    actual_diffs = [i for i, (a, b) in enumerate(zip(_SEQUENCE, mutant)) if a != b]
    assert len(actual_diffs) == n_edits, (
        f"Expected {n_edits} edits, got {len(actual_diffs)}: {actual_diffs}"
    )

    # All edited positions are within mutable_positions.
    mutable_set = set(_MUTABLE_POSITIONS)
    for pos in actual_diffs:
        assert pos in mutable_set, f"Position {pos} is not in mutable_positions."

    # No silent edit: every changed position has a different base.
    for pos in actual_diffs:
        assert mutant[pos] != _SEQUENCE[pos], (
            f"Silent edit at position {pos}: base unchanged ({_SEQUENCE[pos]})."
        )

    # edited_positions matches the actual diff set.
    assert set(edited_positions) == set(actual_diffs)


# ---------------------------------------------------------------------------
# Collision regression test
# ---------------------------------------------------------------------------


def _build_peaked_tism_inputs(sequence: str, hot_position: int, n_edits: int):
    """Build pos_and_chars + probs that concentrate mass on hot_position.

    The TISM ordering from tism.get_tism is positions-major after reference removal:
    3 consecutive entries per position.  Here we construct a synthetic version of
    that layout manually so the test does not require a real model forward pass.

    hot_position is the index within mutable_positions (0-based), not the
    absolute sequence index (they coincide when mutable_positions = range(L)).
    """
    L = len(sequence)
    vocab = list("ACGT")

    # Build pos_and_chars in the same order get_tism would produce: for each
    # position, the 3 non-reference bases in VOCAB order.
    pos_and_chars = []
    for abs_pos in range(L):
        ref = sequence[abs_pos]
        for base in vocab:
            if base != ref:
                pos_and_chars.append((abs_pos, base))

    n_actions = len(pos_and_chars)  # should be 3 * L

    # Put most mass on the two highest-ranked actions of the hot position.
    # hot_position's actions are at indices [3*hot_position, 3*hot_position+2].
    raw = np.zeros(n_actions, dtype=np.float64)
    hot_start = 3 * hot_position
    raw[hot_start] = 0.47
    raw[hot_start + 1] = 0.47
    raw[hot_start + 2] = 0.02
    # Spread remaining mass uniformly across the rest.
    remaining = 1.0 - raw.sum()
    other_count = n_actions - 3
    raw[raw == 0] = remaining / other_count if other_count > 0 else 0.0
    probs = raw / raw.sum()

    return pos_and_chars, probs


def _analytical_collision_rate(p0: float, p1: float, p2: float) -> float:
    """P(both of 2 draws without replacement land on the same position's 3 actions).

    Derivation: numpy's Generator.choice(replace=False, p=...) implements the
    Efraimidis-Spirakis exponential-race algorithm, which is equivalent to
    successive conditional draws.  For ordered pairs (i, j), i ≠ j, both in
    {0, 1, 2}:

        P(first=i, second=j) = p_i * p_j / (1 - p_i)

    Summing over all 6 ordered pairs:

        P(collision) = p0*(p1+p2)/(1-p0) + p1*(p0+p2)/(1-p1) + p2*(p0+p1)/(1-p2)

    Note: replace=False prevents picking the same *action* twice, but two
    distinct actions can still target the same *position* — that is the
    collision.  This input is the maximally-favorable case for the bug: 94% of
    mass sits on two actions that both map to position 0, so the collision is
    easy to observe (expected rate ≈ 88%).  The test confirms the bug's
    existence, not its prevalence on real TISM maps.
    """
    return (
        p0 * (p1 + p2) / (1 - p0)
        + p1 * (p0 + p2) / (1 - p1)
        + p2 * (p0 + p1) / (1 - p2)
    )


def test_choice_without_replacement_collision_formula():
    """Empirically verify that numpy's choice(replace=False, p=...) uses successive draws.

    The _analytical_collision_rate formula assumes
        P(first=i, second=j) = p_i * p_j / (1 - p_i)
    which is the successive-conditional-draw model.  This test verifies that
    assumption holds for numpy's actual implementation before trusting the
    formula in test_collision_regression.

    If the formula were wrong and the ±0.10 tolerance in test_collision_regression
    were wide enough to hide it, that would be false rigor.  This test uses a
    tight tolerance (≈ 4σ at n=500k) that would catch any significant deviation.

    Empirical result (from the implementation run):
        P(first=0, second=1) ≈ 0.41667  formula=0.41679  diff=-0.00012  ✓
        P(first=0, second=2) ≈ 0.05293  formula=0.05321  diff=-0.00028  ✓
        P(first=1, second=0) ≈ 0.41709  formula=0.41679  diff= 0.00029  ✓
        P(first=1, second=2) ≈ 0.05337  formula=0.05321  diff= 0.00017  ✓
        P(first=2, second=0) ≈ 0.02998  formula=0.03000  diff=-0.00002  ✓
        P(first=2, second=1) ≈ 0.02995  formula=0.03000  diff=-0.00004  ✓
    All within 3σ.  numpy does use successive conditional draws.
    """
    p = np.array([0.47, 0.47, 0.06])
    n_draws = 500_000
    rng = np.random.default_rng(2024)

    results = np.array(
        [rng.choice(3, size=2, replace=False, p=p) for _ in range(n_draws)]
    )
    first = results[:, 0]
    second = results[:, 1]

    # σ for a Bernoulli proportion at n=500k: σ ≤ 0.5/√n ≈ 0.000707
    # 4σ ≈ 0.0028.  Use atol=0.003 for a small safety buffer.
    atol = 0.003

    for i in range(3):
        for j in range(3):
            if i == j:
                continue
            empirical = float(np.mean((first == i) & (second == j)))
            formula = p[i] * p[j] / (1.0 - p[i])
            assert abs(empirical - formula) < atol, (
                f"Ordered-pair formula mismatch for (first={i}, second={j}): "
                f"empirical={empirical:.5f}, formula={formula:.5f}, diff={empirical - formula:.5f}. "
                "numpy does NOT use successive conditional draws — the analytic collision rate "
                "in _analytical_collision_rate is WRONG. Replace it with the empirical estimate."
            )


def test_collision_regression():
    """Collision bug in generate_random_mutant_tism vs fix in positionspace variant.

    Strengthened regression guard: asserts the *rate* of collisions (not just
    their existence) is close to the analytically-derived value.  A refactor
    that cut collisions to near-zero would be caught here even if n_collisions>0
    still passed.
    """
    sequence = "ACGTACGT"  # length 8
    mutable_positions = list(range(8))
    hot_position = 0  # absolute index 0
    n_edits = 2

    pos_and_chars, probs = _build_peaked_tism_inputs(sequence, hot_position, n_edits)
    n_positions = len(sequence)

    # The three action-probs for hot_position (before normalization they are
    # 0.47, 0.47, 0.02; the full vector sums to 1.0 already).
    p_hot = [probs[0], probs[1], probs[2]]
    expected_collision_rate = _analytical_collision_rate(*p_hot)
    # ≈ 0.888; σ ≈ 0.022 at n=200, so ±0.10 gives >4σ safety.
    collision_atol = 0.10

    n_seeds = 200
    tism_distinct_counts = []
    posspace_distinct_counts = []

    for seed in range(n_seeds):
        rng_tism = np.random.default_rng(seed)
        mutant_tism, _ = ada_utils.generate_random_mutant_tism(
            sequence=sequence,
            pos_and_chars_to_mutate=pos_and_chars,
            random_n_loc=n_edits,
            rng=rng_tism,
            probs=probs,
        )
        distinct_tism = sum(1 for a, b in zip(sequence, mutant_tism) if a != b)
        tism_distinct_counts.append(distinct_tism)

        rng_pos = np.random.default_rng(seed)
        position_weights = ada_utils.tism_probs_to_position_weights(probs, n_positions)
        mutant_pos, edited_positions = ada_utils.generate_random_mutant_positionspace(
            sequence=sequence,
            mutable_positions=mutable_positions,
            position_weights=position_weights,
            n_edits=n_edits,
            rng=rng_pos,
        )
        distinct_pos = sum(1 for a, b in zip(sequence, mutant_pos) if a != b)
        posspace_distinct_counts.append(distinct_pos)

    # --- fails-old check: existing sampler collides ---
    n_collisions = sum(1 for c in tism_distinct_counts if c < n_edits)
    assert n_collisions > 0, (
        f"generate_random_mutant_tism never produced a collision across {n_seeds} seeds. "
        "The collision-bug premise is wrong — stop and rethink Plan 01."
    )

    # --- regression guard: observed collision rate ≈ analytical expectation ---
    observed_collision_rate = n_collisions / n_seeds
    assert abs(observed_collision_rate - expected_collision_rate) < collision_atol, (
        f"Observed collision rate {observed_collision_rate:.3f} is not within "
        f"±{collision_atol} of the analytically expected {expected_collision_rate:.3f}. "
        "A near-zero value means the collision bug has been fixed in the old operator "
        "(re-evaluate Plan 01 scope); a value far above expected is an RNG anomaly."
    )

    # --- passes-new check: position-space sampler always produces exact edits ---
    assert all(c == n_edits for c in posspace_distinct_counts), (
        f"generate_random_mutant_positionspace produced wrong edit counts: "
        f"{set(posspace_distinct_counts)}"
    )


# ---------------------------------------------------------------------------
# Marginalizer round-trip test
# ---------------------------------------------------------------------------


def test_tism_probs_to_position_weights_round_trip():
    """Flattening a known per-position map and marginalizing recovers the raw row sums.

    Two checks:
      1. Raw-sum (unnormalized input): catches constant-scale errors such as
         returning row *means* instead of row *sums*, which would pass the
         normalized check but fail here.
      2. Normalized ratio (realistic normalized input): confirms relative weights
         are correct when the input is a valid probability vector.
    """
    rng = np.random.default_rng(7)
    n_positions = 6
    raw = rng.uniform(0.0, 1.0, size=(n_positions, 3))

    # --- Raw-sum check (unnormalized input) ---
    # The function does not require a probability vector; it is a pure reshape+sum.
    probs_3L_raw = raw.flatten()
    expected_raw_sums = raw.sum(axis=1)  # shape (n_positions,)

    result_raw = ada_utils.tism_probs_to_position_weights(probs_3L_raw, n_positions)
    np.testing.assert_allclose(
        result_raw,
        expected_raw_sums,
        rtol=1e-6,
        err_msg=(
            "tism_probs_to_position_weights did not return per-position row sums. "
            "Returning row means (sum/3) would fail here."
        ),
    )

    # --- Normalized ratio check (realistic probability-vector input) ---
    probs_3L_norm = probs_3L_raw / probs_3L_raw.sum()
    expected_normalized = expected_raw_sums / expected_raw_sums.sum()

    result_norm = ada_utils.tism_probs_to_position_weights(probs_3L_norm, n_positions)
    np.testing.assert_allclose(
        result_norm / result_norm.sum(),
        expected_normalized,
        rtol=1e-6,
        err_msg="tism_probs_to_position_weights did not recover the correct normalized ratios.",
    )


def test_tism_probs_to_position_weights_wrong_size():
    """Mis-sized input must raise AssertionError."""
    with pytest.raises(AssertionError):
        ada_utils.tism_probs_to_position_weights(
            probs_3L=np.ones(10, dtype=np.float64) / 10,
            n_positions=4,  # 3*4=12 ≠ 10
        )


def test_real_get_tism_through_marginalizer():
    """End-to-end composition: real get_tism output → tism_probs_to_position_weights.

    The ordering test and the round-trip test are never composed; the exact path
    that Plan 01 integration runs (real get_tism → normalize logits → marginalize
    to position weights) has no test.  This test closes that gap.

    Crucially, the expected per-position weight is computed by GROUPING the real
    pos_and_chars by their position value — not by assuming a layout.  If the
    marginalizer's reshape disagrees with the real ordering, the result will not
    match the grouped sum, and the test fails.
    """
    model_fn = testing_utils.CountLetterModel(target_char="A")
    model = ada_utils.ModelWrapper(model_fn, tism_cost=1.0)
    sequence = "ACGT"  # distinct reference base at each position
    n_positions = len(sequence)

    pos_and_chars, logits = model.get_tism(sequence=sequence, idxs=None)
    assert len(logits) == 3 * n_positions  # sanity: right length

    # Convert logits to a valid probability vector using numerically stable softmax.
    # The optimizer uses a scaled softmax; here a plain softmax is sufficient to
    # produce a nonneg vector of length 3L that feeds tism_probs_to_position_weights.
    # logits are float32; work in float64 to avoid float32 precision accumulation.
    logits_f64 = logits.astype(np.float64)
    shifted = logits_f64 - logits_f64.max()
    probs = np.exp(shifted) / np.exp(shifted).sum()
    assert probs.shape == (3 * n_positions,)
    assert abs(probs.sum() - 1.0) < 1e-9

    # Compute the expected per-position weight by grouping the REAL pos_and_chars.
    # This does not assume any ordering — it accumulates by the actual position
    # field in each (position, char) tuple.
    from collections import defaultdict

    pos_to_prob_sum: dict[int, float] = defaultdict(float)
    for (pos, _), prob in zip(pos_and_chars, probs):
        pos_to_prob_sum[pos] += float(prob)

    # The positions emitted by get_tism (idxs=None) are range(L).
    expected_weights = np.array([pos_to_prob_sum[i] for i in range(n_positions)])

    # Run the marginalizer.
    result = ada_utils.tism_probs_to_position_weights(probs, n_positions)

    assert result.shape == (n_positions,), (
        f"Expected shape ({n_positions},), got {result.shape}"
    )
    np.testing.assert_allclose(
        result,
        expected_weights,
        rtol=1e-6,
        err_msg=(
            "tism_probs_to_position_weights result does not match the per-position "
            "sums computed by grouping the real get_tism pos_and_chars. "
            "The reshape(-1, 3) assumption is inconsistent with the real ordering."
        ),
    )


# ---------------------------------------------------------------------------
# Base-choice distribution test
# ---------------------------------------------------------------------------


def test_positionspace_base_choice_distribution():
    """Over many seeds, all 3 non-reference bases appear; reference never does."""
    sequence = "ACGT"
    mutable_positions = [0]  # Only mutate position 0 (reference = 'A')
    weights = np.array([1.0])
    ref_base = sequence[0]  # 'A'
    expected_alt_bases = {"C", "G", "T"}

    observed_bases = set()
    n_trials = 300
    for seed in range(n_trials):
        rng = np.random.default_rng(seed)
        mutant, _ = ada_utils.generate_random_mutant_positionspace(
            sequence=sequence,
            mutable_positions=mutable_positions,
            position_weights=weights,
            n_edits=1,
            rng=rng,
        )
        new_base = mutant[0]
        assert new_base != ref_base, (
            f"seed {seed}: reference base '{ref_base}' was selected (silent edit)."
        )
        observed_bases.add(new_base)

    assert observed_bases == expected_alt_bases, (
        f"Not all 3 non-reference bases were observed. Got: {observed_bases}"
    )


# ---------------------------------------------------------------------------
# Item 3 — Guard / degenerate-weight paths
# (These are exercised in integration when masking renormalizes weights toward
# zero within a rollout; pin them now while isolated.)
# ---------------------------------------------------------------------------


def test_positionspace_guard_n_edits_zero():
    """n_edits=0 must raise AssertionError (silent no-op is forbidden).

    match="n_edits" verified against ada_utils.py:
      assert n_edits >= 1, "n_edits must be >= 1; ..."
    """
    with pytest.raises(AssertionError, match="n_edits"):
        ada_utils.generate_random_mutant_positionspace(
            sequence="ACGT",
            mutable_positions=[0, 1, 2, 3],
            position_weights=np.ones(4),
            n_edits=0,
            rng=np.random.default_rng(0),
        )


def test_positionspace_guard_n_edits_exceeds_positions():
    """n_edits > len(mutable_positions) must raise AssertionError.

    match="n_edits" verified against ada_utils.py:
      assert ... f"n_edits ({n_edits}) must be <= len(mutable_positions) ..."
    """
    with pytest.raises(AssertionError, match="n_edits"):
        ada_utils.generate_random_mutant_positionspace(
            sequence="ACGT",
            mutable_positions=[0, 1],
            position_weights=np.ones(2),
            n_edits=3,  # > len([0, 1]) = 2
            rng=np.random.default_rng(0),
        )


def test_positionspace_guard_negative_weights():
    """Any negative weight must raise AssertionError.

    match="nonneg" verified against ada_utils.py:
      assert ... "All position_weights must be nonnegative."
    """
    with pytest.raises(AssertionError, match="nonneg"):
        ada_utils.generate_random_mutant_positionspace(
            sequence="ACGT",
            mutable_positions=[0, 1, 2, 3],
            position_weights=np.array([1.0, -0.5, 1.0, 1.0]),
            n_edits=1,
            rng=np.random.default_rng(0),
        )


def test_positionspace_guard_allzero_weights():
    """All-zero weights must raise AssertionError.

    match="least one" verified against ada_utils.py:
      assert ... "At least one position_weight must be > 0."
    """
    with pytest.raises(AssertionError, match="least one"):
        ada_utils.generate_random_mutant_positionspace(
            sequence="ACGT",
            mutable_positions=[0, 1, 2, 3],
            position_weights=np.zeros(4),
            n_edits=1,
            rng=np.random.default_rng(0),
        )


# ---------------------------------------------------------------------------
# Item 4 — Subset mutable_positions with index-alignment check
# ---------------------------------------------------------------------------


def test_positionspace_subset_mutable_positions():
    """Edits land only within a strict non-contiguous mutable subset.

    Also verifies that position_weights is aligned to the mutable_positions
    list by index (weight[i] ↔ mutable_positions[i]), not to absolute sequence
    indices.  Confusion between these two is the indexing bug most likely to
    surface only at integration.
    """
    sequence = "ACGTACGTAC"  # length 10
    mutable_positions = [2, 5, 7]
    mutable_set = set(mutable_positions)

    # --- Structural check: all edits in the subset, exactly n_edits made ---
    uniform_weights = np.array([1.0, 1.0, 1.0])
    n_edits = 2
    for seed in range(50):
        rng = np.random.default_rng(seed)
        mutant, edited_positions = ada_utils.generate_random_mutant_positionspace(
            sequence=sequence,
            mutable_positions=mutable_positions,
            position_weights=uniform_weights,
            n_edits=n_edits,
            rng=rng,
        )
        actual_diffs = [i for i, (a, b) in enumerate(zip(sequence, mutant)) if a != b]
        assert len(actual_diffs) == n_edits, (
            f"seed {seed}: expected {n_edits} edits, got {len(actual_diffs)}"
        )
        for pos in actual_diffs:
            assert pos in mutable_set, (
                f"seed {seed}: position {pos} is not in mutable_positions {mutable_positions}"
            )

    # --- Index-alignment check: weight index i → mutable_positions[i] ---
    # Give all weight to index 1, which maps to mutable_positions[1]=5, NOT
    # absolute sequence index 1.  With n_edits=1, position 5 must always be chosen.
    biased_weights = np.array([0.0, 1.0, 0.0])
    for seed in range(30):
        rng = np.random.default_rng(seed)
        _, edited_positions = ada_utils.generate_random_mutant_positionspace(
            sequence=sequence,
            mutable_positions=mutable_positions,
            position_weights=biased_weights,
            n_edits=1,
            rng=rng,
        )
        assert edited_positions == [5], (
            f"seed {seed}: weight index 1 should map to mutable_positions[1]=5, "
            f"but got edited_positions={edited_positions}. "
            "Index-alignment bug: weights are not aligned to mutable_positions."
        )


# ---------------------------------------------------------------------------
# Item 5 — Statistical weight-bias test
# ---------------------------------------------------------------------------


def test_positionspace_weights_bias_selection():
    """Non-uniform weights bias position selection proportionally to their values.

    Every prior positionspace test uses uniform weights; this is the only test
    that verifies the gradient-mode contract — that a position with higher weight
    is chosen more often in proportion to that weight.
    """
    sequence = "ACGTACGT"  # length 8
    mutable_positions = list(range(8))
    hot_idx = 3  # absolute position 3 gets 80x more weight than the rest

    weights = np.ones(8, dtype=np.float64)
    weights[hot_idx] = 80.0
    # Expected selection rate for hot_idx with n_edits=1:
    #   P(hot) = 80 / (80 + 7*1) = 80/87 ≈ 0.9195
    expected_rate = weights[hot_idx] / weights.sum()

    n_trials = 1000
    hot_selected = sum(
        1
        for seed in range(n_trials)
        if hot_idx
        in ada_utils.generate_random_mutant_positionspace(
            sequence=sequence,
            mutable_positions=mutable_positions,
            position_weights=weights,
            n_edits=1,
            rng=np.random.default_rng(seed),
        )[1]
    )

    observed_rate = hot_selected / n_trials
    # 4σ tolerance: σ = sqrt(p*(1-p)/n)
    sigma = np.sqrt(expected_rate * (1 - expected_rate) / n_trials)
    atol = 4 * sigma
    assert abs(observed_rate - expected_rate) < atol, (
        f"Selection rate {observed_rate:.4f} is not within {atol:.4f} (4σ) of "
        f"expected {expected_rate:.4f}. Non-uniform weights are not being honored."
    )


def test_positionspace_boundary_edit_all_positions():
    """n_edits == len(mutable_positions) must work: every mutable position is edited.

    This is the masking endgame that 1b hits when intra-rollout position
    exhaustion drives n_edits up to the number of remaining positions.
    Every mutable position must appear in edited_positions, no position
    outside mutable_positions may be touched, and no edit is silent.
    """
    sequence = "ACGTACGT"
    mutable_positions = [1, 3, 5, 7]
    n_edits = len(mutable_positions)  # edit all four

    for seed in range(20):
        rng = np.random.default_rng(seed)
        mutant, edited_positions = ada_utils.generate_random_mutant_positionspace(
            sequence=sequence,
            mutable_positions=mutable_positions,
            position_weights=np.ones(len(mutable_positions)),
            n_edits=n_edits,
            rng=rng,
        )
        assert set(edited_positions) == set(mutable_positions), (
            f"seed {seed}: expected all mutable positions edited, "
            f"got {edited_positions}"
        )
        for pos in mutable_positions:
            assert mutant[pos] != sequence[pos], (
                f"seed {seed}: silent edit at position {pos}"
            )
        immutable = set(range(len(sequence))) - set(mutable_positions)
        for pos in immutable:
            assert mutant[pos] == sequence[pos], (
                f"seed {seed}: immutable position {pos} was changed"
            )


# ---------------------------------------------------------------------------
# Published behavior pin for generate_random_mutant_v2
# ---------------------------------------------------------------------------


def test_generate_random_mutant_v2_published_behavior():
    """Pin the RNG-to-sequence mapping of the legacy AdaBeam mutation operator.

    This test is a durable anchor that survives AdaBeam default changes and any
    future class refactors — it verifies the OPERATOR ITSELF, not any class or
    path routing, so it catches regressions in the fundamental published behavior.

    Two checks:
      (a) Exact sequence outputs under a fixed seed (pin the RNG stream).
      (b) Silent-edit rate ≈ 25% over many draws, within a tight tolerance.
          A "silent edit" is when a position is selected for mutation but the
          randomly chosen base from {A,C,G,T} equals the original base.
          With a 4-symbol alphabet and uniform base sampling, P(silent) = 1/4.

    If either assertion fails, the published AdaBeam RNG consumption has changed
    and the golden fixtures must be regenerated from the new operator.
    """
    alphabet = "".join(constants.VOCAB)
    seq = "AAAAAA"
    positions = list(range(len(seq)))

    # ── (a) exact output under seed 42 ─────────────────────────────────────
    rng = np.random.default_rng(42)
    out1 = ada_utils.generate_random_mutant_v2(seq, positions, 2, alphabet, rng)
    out2 = ada_utils.generate_random_mutant_v2(seq, positions, 1, alphabet, rng)
    out3 = ada_utils.generate_random_mutant_v2(seq, positions, 2, alphabet, rng)

    # Exact values captured from pre-refactor AdaBeam at commit 982c75a.
    # Do NOT update these values without regenerating the golden fixtures.
    assert out1 == "CAAACA", (
        f"generate_random_mutant_v2 output 1 changed: expected 'CAAACA', got {out1!r}.  "
        "The published AdaBeam RNG stream has diverged — regenerate the golden fixture."
    )
    assert out2 == "AAAAAA", (  # silent edit: A->A at the chosen position
        f"generate_random_mutant_v2 output 2 changed: expected 'AAAAAA', got {out2!r}."
    )
    assert out3 == "AGATAA", (
        f"generate_random_mutant_v2 output 3 changed: expected 'AGATAA', got {out3!r}."
    )

    # ── (b) silent-edit rate ≈ 25% (published behavior) ────────────────────
    # The legacy operator samples a base uniformly from {A,C,G,T} regardless of
    # the current base, so each edit has a 1/4 chance of being A→A (silent).
    # This is the defining behavioral difference from the corrected operator.
    N = 40_000
    rng_rate = np.random.default_rng(0)
    seq10 = "AAAAAAAAAA"
    pos10 = list(range(10))

    n_silent = sum(
        # n=1: a single position is chosen, then a base drawn uniformly.
        # Since seq is all-A, a silent edit leaves the sequence unchanged.
        1
        for _ in range(N)
        if ada_utils.generate_random_mutant_v2(seq10, pos10, 1, alphabet, rng_rate)
        == seq10
    )
    rate = n_silent / N
    assert abs(rate - 0.25) < 0.01, (
        f"Silent-edit rate is {rate:.4f}; expected ≈0.25 (±0.01).  "
        "If the operator was changed to non-uniform base sampling, this is expected — "
        "update the golden fixture and this bound accordingly."
    )
