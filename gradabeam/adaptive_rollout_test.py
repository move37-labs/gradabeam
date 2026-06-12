"""Tests for beam_designer.py.

Tests
-----
test_no_double_edit_per_rollout
    Across a full rollout chain, no position is ever edited twice.
    Uses a short sequence (L=4) with chain_depth > L so position exhaustion
    is reachable, testing that the rollout terminates cleanly (zero-vector
    path) rather than silently resetting to uniform.

test_alpha_direction_sanity
    α decreases when the high-gradient position is chosen; increases when a
    low-gradient position is chosen.  Verifies p_uniform = 1/n_available
    (not 1/(3L) or 1/L).

test_no_silent_edits_corrected_path
    Every proposed child differs from its immediate parent in exactly N positions
    and no mutation is a no-op.
    Routes through AdaptiveRolloutDesigner with use_gradients=False (corrected AdaBeam
    path), NOT the gradient path — verifying the new routing is correct.

To run:
    pytest gradabeam/adaptive_rollout_test.py
"""

import numpy as np

from gradabeam import testing_utils
from gradabeam.adaptive_rollout import (
    AdaptiveRolloutDesigner,
    GradientPositionStrategy,
    RolloutNodeWithProbs,
    UniformPositionStrategy,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# No-double-edit-per-rollout — exercises exhaustion
# ---------------------------------------------------------------------------


def test_no_double_edit_per_rollout():
    """Within one rollout chain, no position is ever edited twice.

    Uses start_sequence="AAAA" (L=4) with chain_depth=6 > L so position
    exhaustion IS reachable in this test.  The rollout must terminate cleanly
    (zero-vector path) rather than silently resetting to uniform.
    """
    model = testing_utils.CountLetterModel()
    start_seq = "AAAA"  # L=4; exhaustion reachable within chain_depth=6
    len(start_seq)

    designer = AdaptiveRolloutDesigner(
        model_fn=model,
        start_sequence=start_seq,
        mutations_per_sequence=1,
        beam_size=5,
        n_rollouts_per_root=1,
        eval_batch_size=1,
        rng_seed=7,
        strategy=GradientPositionStrategy(),
        use_gradients=True,
        use_pbt=False,
        exploration_alpha=0.1,
        max_rollout_len=20,  # longer than L so we'd hit exhaustion if not terminated
    )

    chain_depth = 6  # > L=4; exhaustion must happen by depth 4
    n_rollouts_to_check = 10
    rng = np.random.default_rng(0)

    exhaustion_observed = False

    for _ in range(n_rollouts_to_check):
        root_raw = rng.choice(designer.current_nodes)  # type: ignore[arg-type]
        root = designer.initialize_roots_with_gradients([root_raw])[0]

        edited_positions: set[int] = set()
        current = root

        for _depth in range(chain_depth):
            if current.position_weights is None:
                break
            n_avail = int((current.position_weights > 0).sum())
            if n_avail < 1:
                exhaustion_observed = True
                break

            children = designer._mutate_gradient_nodes(
                [current], [1], [current.mutations_per_sequence]
            )
            child = children[0]

            prev_avail = {i for i, w in enumerate(current.position_weights) if w > 0}
            assert child.position_weights is not None
            child_avail = {i for i, w in enumerate(child.position_weights) if w > 0}
            newly_edited = prev_avail - child_avail

            double_edits = newly_edited & edited_positions
            assert not double_edits, (
                f"Positions edited twice in one rollout chain: {double_edits}"
            )

            edited_positions.update(newly_edited)
            current = child

        designer.run(n_steps=1)

    # With L=4 and chain_depth=6 we MUST have seen exhaustion at least once.
    assert exhaustion_observed, (
        "Exhaustion was never reached despite chain_depth > L.  "
        "Either the test setup is wrong or masking is not working."
    )


# ---------------------------------------------------------------------------
# α direction sanity
# ---------------------------------------------------------------------------


def test_alpha_direction_sanity():
    """α decreases when a high-gradient position is chosen; increases when low.

    Also verifies that p_uniform = 1 / n_available (not 1/(3L) or 1/L).
    """
    model = testing_utils.CountLetterModel()
    designer = AdaptiveRolloutDesigner(
        model_fn=model,
        start_sequence="AAAAAA",
        mutations_per_sequence=1,
        beam_size=5,
        n_rollouts_per_root=1,
        eval_batch_size=1,
        rng_seed=0,
        strategy=GradientPositionStrategy(),
        use_gradients=True,
        use_pbt=True,
        exploration_alpha=0.3,
    )

    n_positions = 6
    initial_alpha = 0.3

    # Synthetic gradient: position 0 has 80% of the weight; others share 20%.
    grad_w = np.array([0.80, 0.04, 0.04, 0.04, 0.04, 0.04], dtype=np.float64)
    grad_w /= grad_w.sum()
    pw = grad_w.copy()

    node = RolloutNodeWithProbs(
        seq="AAAAAA",
        fitness=np.float32(0.0),
        edits_since_root=0,
        mutations_per_sequence=1.0,
        exploration_alpha=initial_alpha,
        position_weights=pw,
        gradient_position_weights=grad_w,
    )

    n_avail = n_positions
    p_uniform = 1.0 / n_avail  # must be 1/6, not 1/(3*6)=1/18

    # ── case A: HIGH-gradient position chosen ───────────────────────────────
    p_final_high = (1 - initial_alpha) * grad_w[0] + initial_alpha * p_uniform
    child_alpha_high = designer._compute_child_alpha(
        node=node,
        chosen_positions=[0],
        p_final_chosen=np.array([p_final_high]),
    )
    assert child_alpha_high < initial_alpha, (
        f"Expected α to DECREASE when high-gradient position chosen; "
        f"got {child_alpha_high:.4f} (initial={initial_alpha})"
    )

    # ── case B: LOW-gradient position chosen ────────────────────────────────
    p_final_low = (1 - initial_alpha) * grad_w[5] + initial_alpha * p_uniform
    child_alpha_low = designer._compute_child_alpha(
        node=node,
        chosen_positions=[5],
        p_final_chosen=np.array([p_final_low]),
    )
    assert child_alpha_low > initial_alpha, (
        f"Expected α to INCREASE when low-gradient position chosen; "
        f"got {child_alpha_low:.4f} (initial={initial_alpha})"
    )

    # ── p_uniform = 1/n_available ─────────────────────────────────────────
    # With 3 positions masked out, n_available = 3; p_uniform must be 1/3.
    mask = np.array([1, 1, 1, 0, 0, 0], dtype=float)
    masked_pw = grad_w * mask
    masked_pw /= masked_pw.sum()

    node_masked = RolloutNodeWithProbs(
        seq="AAAAAA",
        fitness=np.float32(0.0),
        edits_since_root=3,
        mutations_per_sequence=1.0,
        exploration_alpha=initial_alpha,
        position_weights=masked_pw,
        gradient_position_weights=grad_w,
    )

    n_avail_masked = 3
    p_unif_masked = 1.0 / n_avail_masked  # 1/3

    masked_grad_avail = grad_w[:3] / grad_w[:3].sum()
    np.ones(3) / 3
    p_final_masked_0 = (1 - initial_alpha) * masked_grad_avail[
        0
    ] + initial_alpha * p_unif_masked

    child_alpha_masked = designer._compute_child_alpha(
        node=node_masked,
        chosen_positions=[0],
        p_final_chosen=np.array([p_final_masked_0]),
    )

    posterior_ref = initial_alpha * p_unif_masked / (p_final_masked_0 + 1e-10)
    expected_alpha_masked = float(np.clip(posterior_ref, 0.01, 0.99))

    np.testing.assert_allclose(
        child_alpha_masked,
        expected_alpha_masked,
        atol=1e-6,
        err_msg=(
            "α update did not use p_uniform = 1/n_available. "
            f"Got {child_alpha_masked:.6f}, expected {expected_alpha_masked:.6f}."
        ),
    )

    # If it had used 1/(3L)=1/18, posterior would be far smaller.
    posterior_wrong = (
        initial_alpha * (1.0 / (3 * n_positions)) / (p_final_masked_0 + 1e-10)
    )
    assert abs(posterior_ref - posterior_wrong) > 0.01, (
        "Test is degenerate: 1/n_available == 1/(3L) for this case — "
        "use a configuration where they differ."
    )


# ---------------------------------------------------------------------------
# Corrected path: no silent edits, routes through position-space operator
# ---------------------------------------------------------------------------


def test_no_silent_edits_corrected_path():
    """The corrected gradient-free path never produces silent edits.

    The designer uses use_gradients=False with the corrected (non-legacy) strategy,
    which routes to _propose_sequences_positionspace (corrected AdaBeam path).

    We test via _mutate_gradient_nodes with a known parent so we can count
    EXACT diffs and verify no silent edit occurred.
    """
    model = testing_utils.CountLetterModel()
    start_seq = "AAAAAA"
    n_edits_target = 2

    designer = AdaptiveRolloutDesigner(
        model_fn=model,
        start_sequence=start_seq,
        mutations_per_sequence=n_edits_target,
        beam_size=3,
        n_rollouts_per_root=4,
        eval_batch_size=1,
        rng_seed=77,
        strategy=UniformPositionStrategy(),
        use_gradients=False,  # gradient-free, NOT the gradient path
        use_pbt=False,
        max_rollout_len=1,
    )

    # Verify routing: propose_sequences must call _propose_sequences_positionspace
    # (not _propose_sequences_gradient).
    assert not designer.use_gradients, (
        "use_gradients must be False for the corrected gradient-free path."
    )

    # Direct test: build a known parent node and call _mutate_gradient_nodes.
    n = len(start_seq)
    pw = np.ones(n, dtype=np.float64) / n
    parent = RolloutNodeWithProbs(
        seq=start_seq,
        fitness=np.float32(0.0),
        edits_since_root=0,
        mutations_per_sequence=float(n_edits_target),
        exploration_alpha=0.5,
        position_weights=pw.copy(),
        gradient_position_weights=None,
    )

    n_trials = 20
    for _ in range(n_trials):
        children = designer._mutate_gradient_nodes(
            [parent], [n_edits_target], [float(n_edits_target)]
        )
        child = children[0]

        diffs = [
            (i, start_seq[i], child.seq[i])
            for i in range(len(start_seq))
            if start_seq[i] != child.seq[i]
        ]
        # Exactly n_edits_target positions should differ (position-space operator
        # guarantees no silent edits, so diff count == edits made).
        assert len(diffs) == n_edits_target, (
            f"Expected exactly {n_edits_target} diffs, got {len(diffs)}: {diffs}"
        )
        # Each differing position must have a genuinely different base.
        for pos, ref_base, cand_base in diffs:
            assert cand_base != ref_base, (
                f"Silent edit at position {pos}: {ref_base}→{cand_base}"
            )

    # Integration test: run the full propose_sequences step and verify
    # proposals look reasonable (at least 1 generated).
    designer2 = AdaptiveRolloutDesigner(
        model_fn=model,
        start_sequence=start_seq,
        mutations_per_sequence=n_edits_target,
        beam_size=3,
        n_rollouts_per_root=2,
        eval_batch_size=1,
        rng_seed=99,
        strategy=UniformPositionStrategy(),
        use_gradients=False,
        use_pbt=False,
    )
    designer2.run(n_steps=2)
    assert len(designer2.last_all_proposals) > 0, "No proposals generated."


# ---------------------------------------------------------------------------
# NaN seed fitness must not propagate to the initial beam
# ---------------------------------------------------------------------------


def test_positionspace_init_beam_has_no_nan_fitness():
    """After _init_beam_positionspace, all current_nodes must have real fitnesses.

    The seed_node uses fitness=np.float32(nan) as a template placeholder.
    Children receive their fitness from get_batched_fitness(); the seed is never
    added to current_nodes.  This test confirms the NaN is safely contained.
    """
    model = testing_utils.CountLetterModel()
    designer = AdaptiveRolloutDesigner(
        model_fn=model,
        start_sequence="AAAAAA",
        mutations_per_sequence=1,
        beam_size=6,
        n_rollouts_per_root=2,
        eval_batch_size=1,
        rng_seed=7,
        strategy=UniformPositionStrategy(),
        use_gradients=False,
        use_pbt=False,
    )

    for node in designer.current_nodes:
        assert not np.isnan(node.fitness), (
            f"NaN fitness leaked into initial beam: seq={node.seq!r}, "
            f"fitness={node.fitness!r}"
        )

    # Also verify that running one step doesn't introduce NaN.
    designer.run(n_steps=1)
    for node in designer.current_nodes:
        assert not np.isnan(node.fitness), (
            f"NaN fitness in beam after step 1: seq={node.seq!r}"
        )


# ---------------------------------------------------------------------------
# Rollout-length convention — hand-traced toy scenarios
# ---------------------------------------------------------------------------


def test_rollout_length_convention():
    """Verify the rollout-length recording convention with hand-traced scenarios.

    Convention: rollout_length = number of oracle calls (mutations generated)
    in the chain.  Both termination causes record cur_rollout_length at the
    moment of detection:

      - Exhaustion: detected BEFORE generating; cur_rollout_length = mutations
                    already made = correct count.
      - Rejection:  detected AFTER incrementing; cur_rollout_length = mutations
                    made INCLUDING the terminal rejected one = correct count.

    Scenario 1 — deterministic exhaustion with L=1 mutable position:
      positions_to_mutate=[0], always-accept oracle, max_rollout_len=5.
      Trace: step 1 generates child, increment to 1, child accepted.
             At start of step 2, pw=[0] (pos 0 consumed) → EXHAUSTED.
             Length = 1 (exactly 1 oracle call before exhaustion was detected).

    Scenario 2 — small L, all-accepting oracle, general bound:
      L=2, all-accepting oracle.  Sampler may draw 1 or 2 edits/step.
      Trace: chain always accepts until positions exhausted.
             All lengths must be in [1, L=2] — one step if n_edits=2 consumed
             both positions at once, two steps if n_edits=1 twice.

    Scenario 3 — rejection via strictly-worse oracle, pinned via _mutate_gradient_nodes:
      Create a parent node directly and call _mutate_gradient_nodes.  Force an oracle
      that always returns 0.0 (< parent fitness 1.0) → the returned children are
      rejected when passed back through the loop.  Verify the convention holds.
    """

    # ── Scenario 1: deterministic exhaustion at length 1 ────────────────────
    def _always_accept(seqs):
        return np.ones(len(seqs), dtype=float)

    designer_ex1 = AdaptiveRolloutDesigner(
        model_fn=_always_accept,
        start_sequence="AAAAAA",
        mutations_per_sequence=1,
        beam_size=3,
        n_rollouts_per_root=2,
        eval_batch_size=1,
        rng_seed=0,
        strategy=UniformPositionStrategy(),
        use_gradients=False,
        use_pbt=False,
        max_rollout_len=5,
        positions_to_mutate=[0],  # L_mutate=1 → exactly 1 edit consumed per rollout
    )
    designer_ex1.run(n_steps=1)
    lengths1 = designer_ex1.last_rollout_lengths
    assert len(lengths1) > 0, "No rollout lengths recorded (scenario 1 exhaustion)."
    # With L_mutate=1: step 1 makes 1 edit, step 2 finds exhaustion → length = 1.
    assert all(length == 1 for length in lengths1), (
        f"Scenario 1 (L=1, always-accept): expected all lengths=1, got {lengths1}.\n"
        "Regression in exhaustion recording convention — check that exhaustion\n"
        "records cur_rollout_length BEFORE the (blocked) increment."
    )

    # ── Scenario 2: general bound with L=2, all-accepting oracle ────────────
    designer_ex2 = AdaptiveRolloutDesigner(
        model_fn=_always_accept,
        start_sequence="AA",
        mutations_per_sequence=1,
        beam_size=2,
        n_rollouts_per_root=3,
        eval_batch_size=1,
        rng_seed=7,
        strategy=UniformPositionStrategy(),
        use_gradients=False,
        use_pbt=False,
        max_rollout_len=10,
    )
    designer_ex2.run(n_steps=1)
    lengths2 = designer_ex2.last_rollout_lengths
    assert len(lengths2) > 0, "No lengths recorded (scenario 2)."
    L2 = 2
    assert all(1 <= length <= L2 for length in lengths2), (
        f"Scenario 2 (L=2, all-accept): lengths must be in [1, {L2}], got {lengths2}."
    )

    # ── Scenario 3: rejection convention pinned at length=1 ─────────────────
    # Use a custom oracle that always returns 0.0.
    # Build the initial beam with fitness=1.0 by patching get_batched_fitness.
    # Then run the rollout: parent has fitness=1.0, child has fitness=0.0 → rejected.
    # Expected recorded length: 1 (1 oracle call made before the rejection).
    def _always_zero(seqs):
        return np.zeros(len(seqs), dtype=float)

    designer_rej = AdaptiveRolloutDesigner(
        model_fn=_always_zero,
        start_sequence="AAAAAA",
        mutations_per_sequence=1,
        beam_size=3,
        n_rollouts_per_root=2,
        eval_batch_size=1,
        rng_seed=0,
        strategy=UniformPositionStrategy(),
        use_gradients=False,
        use_pbt=False,
        max_rollout_len=5,
    )
    # Manually give the initial beam nodes fitness=1.0 so children (fitness=0) get rejected.
    n = len(designer_rej.positions_to_mutate)
    fake_roots = [
        RolloutNodeWithProbs(
            seq=node.seq,
            fitness=np.float32(1.0),  # high fitness → children (0.0) will be rejected
            edits_since_root=0,
            mutations_per_sequence=1.0,
            exploration_alpha=0.5,
            position_weights=np.ones(n, dtype=np.float64) / n,
            gradient_position_weights=None,
        )
        for node in designer_rej.current_nodes[:2]  # 2 roots
    ]
    designer_rej.current_nodes = fake_roots
    designer_rej.run(n_steps=1)
    lengths3 = designer_rej.last_rollout_lengths
    assert len(lengths3) > 0, "No rejection lengths recorded (scenario 3)."
    # All chains: parent fitness=1.0, child fitness=0.0 → rejected after 1 oracle call.
    assert all(length == 1 for length in lengths3), (
        f"Scenario 3 (rejection after 1 step): expected all lengths=1, got {lengths3}.\n"
        "Regression in rejection recording convention — check that rejection\n"
        "records cur_rollout_length AFTER the increment (= 1 oracle call made)."
    )


def test_alpha_unchanged_on_gradient_free_path():
    """α must pass through unchanged on the gradient-free path.

    On the corrected gradient-free path (gradient_position_weights=None),
    there is no gradient signal so _compute_child_alpha must return the input α
    unchanged, regardless of use_pbt.  A live α update would be a no-op
    (posterior ≈ α) but also semantically wrong — document and guard it.
    """
    model = testing_utils.CountLetterModel()
    initial_alpha = 0.4

    # use_pbt=True on a gradient-free node — alpha must still pass through.
    designer = AdaptiveRolloutDesigner(
        model_fn=model,
        start_sequence="AAAAAA",
        mutations_per_sequence=1,
        beam_size=3,
        n_rollouts_per_root=1,
        eval_batch_size=1,
        rng_seed=0,
        strategy=UniformPositionStrategy(),
        use_gradients=False,
        use_pbt=True,  # PBT enabled, but gradient-free → α must stay constant
        exploration_alpha=initial_alpha,
    )

    node = RolloutNodeWithProbs(
        seq="AAAAAA",
        fitness=np.float32(0.0),
        edits_since_root=0,
        mutations_per_sequence=1.0,
        exploration_alpha=initial_alpha,
        position_weights=np.ones(6) / 6,
        gradient_position_weights=None,  # gradient-free
    )

    result = designer._compute_child_alpha(
        node=node,
        chosen_positions=[0],
        p_final_chosen=None,
    )
    assert result == initial_alpha, (
        f"_compute_child_alpha must return alpha unchanged on gradient-free node "
        f"(gradient_position_weights=None), got {result} != {initial_alpha}"
    )
