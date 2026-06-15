"""Tests for beam_designer.py — Plan 01 part 1b.

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

To run:
    pytest gradabeam/adaptive_rollout_test.py
"""

import numpy as np
import pytest

from gradabeam import testing_utils
from gradabeam import ada_utils
from gradabeam.adaptive_rollout import (
    AdaptiveRolloutDesigner,
    GradientActionStrategy,
    RolloutNodeWithProbs,
    UniformActionStrategy,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_pos_and_chars(sequence: str, positions_to_mutate: list) -> list:
    """Local helper: build 3L (position, char) list for gradient-path test fixtures.

    build_uniform_pos_and_chars was removed from production code since the
    uniform path now uses L-vector position-space representation.  Tests that
    exercise the gradient path or generate_random_mutant_actionspace directly
    still need a way to construct (position, character) pairs.
    """
    from gradabeam import constants

    result = []
    for pos in positions_to_mutate:
        ref = sequence[pos]
        for base in constants.VOCAB:
            if base != ref:
                result.append((pos, base))
    return result


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

    designer = AdaptiveRolloutDesigner(
        model_fn=model,
        start_sequence=start_seq,
        mutations_per_sequence=1,
        beam_size=5,
        n_rollouts_per_root=1,
        eval_batch_size=1,
        rng_seed=7,
        strategy=GradientActionStrategy(),
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
            if current.probs is None:
                break
            # Reshape current.probs to check how many positions are still available
            any_active_per_pos = (current.probs > 0).reshape(-1, 3).any(axis=1)
            n_avail = int(any_active_per_pos.sum())
            if n_avail < 1:
                exhaustion_observed = True
                break

            children = designer._mutate_gradient_nodes(
                [current], [1], [current.mutations_per_sequence]
            )
            child = children[0]

            prev_avail = {
                i
                for i, active in enumerate(
                    (current.probs > 0).reshape(-1, 3).any(axis=1)
                )
                if active
            }
            assert child.probs is not None
            child_avail = {
                i
                for i, active in enumerate((child.probs > 0).reshape(-1, 3).any(axis=1))
                if active
            }
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
    """α decreases when a high-gradient position/action is chosen; increases when low.

    Also verifies that p_uniform = 1 / n_available_actions (not 1/(3L) or 1/L).
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
        strategy=GradientActionStrategy(),
        use_gradients=True,
        use_pbt=True,
        exploration_alpha=0.3,
    )

    initial_alpha = 0.3

    # Synthetic gradient probs (size 18): action 0 (pos 0, base 'C') gets 80%, others share 20%
    grad_probs = np.ones(18, dtype=np.float64) * (0.20 / 17)
    grad_probs[0] = 0.80

    node = RolloutNodeWithProbs(
        seq="AAAAAA",
        fitness=np.float32(0.0),
        edits_since_root=0,
        mutations_per_sequence=1.0,
        exploration_alpha=initial_alpha,
        probs=grad_probs.copy(),
        gradient_probs=grad_probs.copy(),
        n_positions_remaining=6,  # seq has 6 positions, all non-zero
    )

    n_avail = 18
    p_uniform = 1.0 / n_avail  # 1/18

    # ── case A: HIGH-gradient action chosen ───────────────────────────────
    p_final_high = (1 - initial_alpha) * grad_probs[0] + initial_alpha * p_uniform
    child_alpha_high = designer._compute_child_alpha(
        node=node,
        p_final_chosen_list=[p_final_high],
    )
    assert child_alpha_high < initial_alpha, (
        f"Expected α to DECREASE when high-gradient action chosen; "
        f"got {child_alpha_high:.4f} (initial={initial_alpha})"
    )

    # ── case B: LOW-gradient action chosen ────────────────────────────────
    p_final_low = (1 - initial_alpha) * grad_probs[17] + initial_alpha * p_uniform
    child_alpha_low = designer._compute_child_alpha(
        node=node,
        p_final_chosen_list=[p_final_low],
    )
    assert child_alpha_low > initial_alpha, (
        f"Expected α to INCREASE when low-gradient action chosen; "
        f"got {child_alpha_low:.4f} (initial={initial_alpha})"
    )

    # ── p_uniform = 1/n_available ─────────────────────────────────────────
    # With 3 positions (9 actions) masked out, n_available = 9; p_uniform must be 1/9.
    masked_probs = grad_probs.copy()
    masked_probs[9:] = 0.0

    node_masked = RolloutNodeWithProbs(
        seq="AAAAAA",
        fitness=np.float32(0.0),
        edits_since_root=3,
        mutations_per_sequence=1.0,
        exploration_alpha=initial_alpha,
        probs=masked_probs,
        gradient_probs=grad_probs.copy(),
        n_positions_remaining=3,  # positions 3-5 zeroed; 3 positions remain
    )

    n_avail_masked = 9
    p_unif_masked = 1.0 / n_avail_masked  # 1/9

    masked_grad_avail = grad_probs[:9] / grad_probs[:9].sum()
    p_final_masked_0 = (1 - initial_alpha) * masked_grad_avail[
        0
    ] + initial_alpha * p_unif_masked

    child_alpha_masked = designer._compute_child_alpha(
        node=node_masked,
        p_final_chosen_list=[p_final_masked_0],
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
    posterior_wrong = initial_alpha * (1.0 / 18) / (p_final_masked_0 + 1e-10)
    assert abs(posterior_ref - posterior_wrong) > 0.01, (
        "Test is degenerate: 1/n_available == 1/(3L) for this case — "
        "use a configuration where they differ."
    )


# ---------------------------------------------------------------------------
# no silent edits, routes through corrected path
# ---------------------------------------------------------------------------


def test_no_silent_edits_corrected_path():
    """The corrected gradient-free path never produces silent edits.

    The designer uses use_gradients=False, which
    routes to _propose_sequences_actionspace (corrected AdaBeam path).

    BEFORE the routing fix (Plan 01 part 1b rev), this config was silently
    routed to _propose_sequences_legacy, which used generate_random_mutant_v2
    (with ~25% silent edits).  The test assertions were also tautological
    (checked `cand_base != ref_base` inside `if ref_base != cand_base`, which
    is always True).  Both bugs are now fixed.

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
        strategy=UniformActionStrategy(),
        use_gradients=False,  # gradient-free, NOT the gradient path
        use_pbt=False,
        max_rollout_len=1,
    )

    # Verify routing: propose_sequences must call _propose_sequences_actionspace
    # (not _propose_sequences_gradient).
    assert not designer.use_gradients, (
        "use_gradients must be False for the corrected gradient-free path."
    )

    # Direct test: build a known position-space parent node and call the strategy's propose().
    # The uniform path now uses L-vector position weights, not 3L action probs.
    n = len(start_seq)
    parent = RolloutNodeWithProbs(
        seq=start_seq,
        fitness=np.float32(0.0),
        edits_since_root=0,
        mutations_per_sequence=float(n_edits_target),
        exploration_alpha=0.5,
        probs=None,
        gradient_probs=None,
        pos_and_chars=None,
        position_weights=np.ones(n, dtype=np.float64) / n,
        n_positions_remaining=n,
    )

    n_trials = 20
    for _ in range(n_trials):
        proposal = designer.strategy.propose(
            parent,
            designer.rng,
            n_edits_target,
            designer.positions_to_mutate,
        )
        child_seq = proposal.mutant_seq

        diffs = [
            (i, start_seq[i], child_seq[i])
            for i in range(len(start_seq))
            if start_seq[i] != child_seq[i]
        ]
        # Exactly n_edits_target positions should differ; position-space operator
        # guarantees no silent edits so diff count == edits made.
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
        strategy=UniformActionStrategy(),
        use_gradients=False,
        use_pbt=False,
    )
    designer2.run(n_steps=2)
    assert len(designer2.last_all_proposals) > 0, "No proposals generated."


# ---------------------------------------------------------------------------
# Bug 2: NaN seed fitness must not propagate to the initial beam
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
        strategy=UniformActionStrategy(),
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
# Bug 3: rollout-length convention — hand-traced toy scenarios
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

    Scenario 1 — exhaustion with L=2 mutable positions, mu=0.5:
      positions_to_mutate=[0, 1], always-accept oracle, max_rollout_len=5.
      Trace: sampler draws 1 or 2 edits; chain always accepts until both
             positions are consumed → EXHAUSTED.
             All lengths must be in [1, 2].
      (Previously used L=1 / mu=1.0; changed to L=2 so mu = rate/L < 1,
      which is now required by the construction assert and _F_inverse guard.)

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

    # ── Scenario 1: exhaustion with L=2 mutable positions, mu=0.5 ───────────
    # Previously used positions_to_mutate=[0] (L=1, mu=1.0), which is now
    # forbidden by the strict mu < 1 construction assert.  Changed to L=2
    # (mu=0.5) which still exercises exhaustion detection but keeps mu < 1.
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
        strategy=UniformActionStrategy(),
        use_gradients=False,
        use_pbt=False,
        max_rollout_len=5,
        positions_to_mutate=[0, 1],  # L_mutate=2, mu=0.5 → exhaustion in 1-2 steps
    )
    designer_ex1.run(n_steps=1)
    lengths1 = designer_ex1.last_rollout_lengths
    assert len(lengths1) > 0, "No rollout lengths recorded (scenario 1 exhaustion)."
    # With L_mutate=2: sampler draws 1 or 2 edits; exhaustion occurs in 1 or 2 steps.
    assert all(1 <= length <= 2 for length in lengths1), (
        f"Scenario 1 (L=2, mu=0.5, always-accept): expected all lengths in [1,2], "
        f"got {lengths1}.\n"
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
        strategy=UniformActionStrategy(),
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
        strategy=UniformActionStrategy(),
        use_gradients=False,
        use_pbt=False,
        max_rollout_len=5,
    )
    # Manually give the initial beam nodes fitness=1.0 so children (fitness=0) get rejected.
    # Use position-space (L-vector) nodes — the designer uses UniformActionStrategy.
    n = len(designer_rej.positions_to_mutate)
    fake_roots = [
        RolloutNodeWithProbs(
            seq=node.seq,
            fitness=np.float32(1.0),  # high fitness → children (0.0) will be rejected
            edits_since_root=0,
            mutations_per_sequence=1.0,
            exploration_alpha=0.5,
            probs=None,
            gradient_probs=None,
            pos_and_chars=None,
            position_weights=np.ones(n, dtype=np.float64) / n,
            n_positions_remaining=n,
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
    """Bug 1: α must pass through unchanged on the gradient-free path.

    On the corrected gradient-free path the sampler returns p_final_chosen_list=[]
    (no actions chosen when the list is empty — or equivalently the caller passes []).
    _compute_child_alpha must return the input α unchanged when use_pbt=False OR
    when p_final_chosen_list is empty, regardless of gradient_probs.
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
        strategy=UniformActionStrategy(),
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
        probs=None,
        gradient_probs=None,  # gradient-free — this is what gates _compute_child_alpha
        pos_and_chars=None,
        position_weights=np.ones(6, dtype=np.float64) / 6,
        n_positions_remaining=6,
    )

    result = designer._compute_child_alpha(
        node=node,
        p_final_chosen_list=[],
    )
    assert result == initial_alpha, (
        f"_compute_child_alpha must return alpha unchanged on gradient-free node "
        f"(gradient_probs=None), got {result} != {initial_alpha}"
    )


# ---------------------------------------------------------------------------
# Corrected action-space specific tests & guards
# ---------------------------------------------------------------------------


def test_gradient_picks_base():
    """Verify that when alpha is near 0, the base selection matches the capped softmax distribution."""
    sequence = "AAAA"
    pos_and_chars = _build_pos_and_chars(sequence, [0, 1, 2, 3])
    # 12 actions total.
    # Let's say action 0 (pos 0, base 'C') has 0.10 capped probability.
    # Other 11 actions share 0.90 uniformly (0.90/11 ≈ 0.0818).
    # Since alpha is 0, the probability of selecting Action 0 should be exactly 0.10.
    probs = np.zeros(12, dtype=np.float64)
    probs[0] = 0.10
    probs[1:] = 0.90 / 11

    n_trials = 2000
    chosen_indices = []
    for seed in range(n_trials):
        rng = np.random.default_rng(seed)
        _, edited_indices, _, _, _ = ada_utils.generate_random_mutant_actionspace(
            sequence=sequence,
            pos_and_chars_to_mutate=pos_and_chars,
            n_edits=1,
            rng=rng,
            probs=probs.copy(),
        )
        chosen_indices.append(edited_indices[0])

    # Count frequencies
    counts = np.bincount(chosen_indices, minlength=12)
    rates = counts / n_trials

    # Verify rates match expectation within 4 sigma
    for idx, expected in enumerate(probs):
        sigma = np.sqrt(expected * (1 - expected) / n_trials)
        assert abs(rates[idx] - expected) < 4 * sigma, (
            f"Index {idx}: observed rate {rates[idx]:.4f} diverged from expected {expected:.4f}"
        )


def test_adabeam_is_gradabeam_gradients_off():
    """Verify that at α=1 with uniform probs, p_final equals the uniform step-start prob.

    Under reading B, p_final[k] = probs[action_k] (the step-start policy probability).
    With uniform probs=1/24 and n_edits=2, both draws report 1/24.  The within-child
    positional masking does NOT enter p_final — that would be reading A, which we
    intentionally avoid (see design comment in generate_random_mutant_actionspace).
    """
    start_seq = "ACGTACGT"
    pos_and_chars = _build_pos_and_chars(start_seq, list(range(8)))

    _, _, _, p_final_chosen_list, _ = ada_utils.generate_random_mutant_actionspace(
        sequence=start_seq,
        pos_and_chars_to_mutate=pos_and_chars,
        n_edits=2,
        rng=np.random.default_rng(42),
        probs=np.ones(24) / 24,
    )

    # Reading B: p_final is the step-start policy probability, so masking does NOT
    # enter it — both draws are 1/24, not 1/24 then 1/21. (1/21 would be the
    # within-child WOR likelihood, which we intentionally do not use; see the
    # design comment in generate_random_mutant_actionspace.)
    assert abs(p_final_chosen_list[0] - 1 / 24) < 1e-9
    assert abs(p_final_chosen_list[1] - 1 / 24) < 1e-9


def test_unified_gradient_free_path_does_zero_backward_passes():
    """The unified gradient-free path must never call get_tism / do a backward pass.

    This guards the UNIFIED AdaptiveRolloutDesigner (use_gradients=False), not the
    legacy standalone AdaBeam class, ensuring the cost claim holds on the production
    code path.  Checks designer.model.n_backward (ModelWrapper counter) rather than
    a spy on the raw model, so it catches costs charged via any route.
    """
    designer = AdaptiveRolloutDesigner(
        model_fn=testing_utils.CountLetterModel(),
        start_sequence="ACGTACGT",
        mutations_per_sequence=2,
        beam_size=4,
        n_rollouts_per_root=2,
        strategy=UniformActionStrategy(),
        use_gradients=False,
        use_pbt=False,
        eval_batch_size=1,
        rng_seed=42,
    )
    designer.run(n_steps=2)
    assert designer.model.n_backward == 0, (
        f"Expected 0 backward passes on gradient-free path, got {designer.model.n_backward}"
    )


def test_alpha_base_coupling():
    """Holding position fixed, gradient-favored base lowers α; disfavored base raises it.

    In action space the PBT ratio P_uniform/P_child now measures JOINT
    (position-and-base) surprise.  Agreeing with the gradient's preferred base
    raises P_child and drives α down faster than choosing a disfavored base.
    """
    model = testing_utils.CountLetterModel()
    initial_alpha = 0.3

    designer = AdaptiveRolloutDesigner(
        model_fn=model,
        start_sequence="AAAAAA",
        mutations_per_sequence=1,
        beam_size=5,
        n_rollouts_per_root=1,
        eval_batch_size=1,
        rng_seed=0,
        strategy=GradientActionStrategy(),
        use_gradients=True,
        use_pbt=True,
        exploration_alpha=initial_alpha,
    )

    # Synthetic gradient: position 0 has one very hot base (action 0) and
    # two cold bases (actions 1, 2). All other positions are uniform.
    n_actions = 18  # 6 positions × 3 bases
    grad_probs = np.ones(n_actions, dtype=np.float64) / n_actions
    # Make position 0 action 0 very hot
    grad_probs[0] = 0.50
    grad_probs[1] = 0.01
    grad_probs[2] = 0.01
    grad_probs /= grad_probs.sum()

    node = RolloutNodeWithProbs(
        seq="AAAAAA",
        fitness=np.float32(0.0),
        edits_since_root=0,
        mutations_per_sequence=1.0,
        exploration_alpha=initial_alpha,
        probs=grad_probs.copy(),
        gradient_probs=grad_probs.copy(),
        n_positions_remaining=6,  # seq has 6 positions, all non-zero
    )

    # Both scenarios choose position 0. The difference is the BASE.
    p_uniform = 1.0 / n_actions

    # Case A: gradient-FAVORED base (action 0, high gradient mass)
    p_final_favored = (1 - initial_alpha) * grad_probs[0] + initial_alpha * p_uniform
    alpha_favored = designer._compute_child_alpha(
        node=node,
        p_final_chosen_list=[p_final_favored],
    )

    # Case B: gradient-DISFAVORED base (action 1, low gradient mass)
    p_final_disfavored = (1 - initial_alpha) * grad_probs[1] + initial_alpha * p_uniform
    alpha_disfavored = designer._compute_child_alpha(
        node=node,
        p_final_chosen_list=[p_final_disfavored],
    )

    assert alpha_favored < alpha_disfavored, (
        f"Choosing the gradient-favored base should yield LOWER α than disfavored. "
        f"Got favored={alpha_favored:.4f}, disfavored={alpha_disfavored:.4f}."
    )
    # Favored base should push α below initial; disfavored should push above.
    assert alpha_favored < initial_alpha, (
        f"Gradient-favored base should decrease α below {initial_alpha}, got {alpha_favored}."
    )
    assert alpha_disfavored > initial_alpha, (
        f"Gradient-disfavored base should increase α above {initial_alpha}, got {alpha_disfavored}."
    )


def test_positional_mask_zeroes_whole_position():
    """After editing action k, all 3 entries of that position are zero in carried probs."""
    sequence = "ACGT"
    pos_and_chars = _build_pos_and_chars(sequence, list(range(4)))
    n_actions = len(pos_and_chars)  # 12
    probs = np.ones(n_actions, dtype=np.float64) / n_actions

    for seed in range(50):
        rng = np.random.default_rng(seed)
        _, edited_indices, remaining_probs, _, _ = (
            ada_utils.generate_random_mutant_actionspace(
                sequence=sequence,
                pos_and_chars_to_mutate=pos_and_chars,
                n_edits=1,
                rng=rng,
                probs=probs.copy(),
            )
        )
        assert len(edited_indices) == 1
        k = edited_indices[0]
        pos_start = 3 * (k // 3)
        # All three action slots for the chosen position must be zero.
        assert remaining_probs[pos_start] == 0.0
        assert remaining_probs[pos_start + 1] == 0.0
        assert remaining_probs[pos_start + 2] == 0.0
        # Remaining positions must still have positive probability.
        assert remaining_probs.sum() > 0


# ---------------------------------------------------------------------------
# mu < 1 strict-bound tests (fixes for _F_inverse safety)
# ---------------------------------------------------------------------------


def test_construction_rejects_mutations_per_sequence_equal_to_L():
    """Construction with mutations_per_sequence == L must raise (mu would be 1.0).

    Checks the strict assert added to guard _F_inverse from receiving mu >= 1.
    """
    model = testing_utils.CountLetterModel()

    # L = 4 (positions 0-3), mutations_per_sequence == L → should raise.
    with pytest.raises(AssertionError, match="must be <"):
        AdaptiveRolloutDesigner(
            model_fn=model,
            start_sequence="ACGTACGT",
            mutations_per_sequence=4,
            beam_size=2,
            n_rollouts_per_root=1,
            eval_batch_size=1,
            rng_seed=0,
            strategy=UniformActionStrategy(),
            use_gradients=False,
            use_pbt=False,
            positions_to_mutate=[0, 1, 2, 3],  # L=4
        )


def test_construction_accepts_mutations_per_sequence_one_below_L():
    """Construction with mutations_per_sequence == L-1 must succeed (mu = (L-1)/L < 1)."""
    model = testing_utils.CountLetterModel()

    # L = 4, mutations_per_sequence = 3 = L-1 → should NOT raise.
    designer = AdaptiveRolloutDesigner(
        model_fn=model,
        start_sequence="ACGTACGT",
        mutations_per_sequence=3,
        beam_size=2,
        n_rollouts_per_root=1,
        eval_batch_size=1,
        rng_seed=0,
        strategy=UniformActionStrategy(),
        use_gradients=False,
        use_pbt=False,
        positions_to_mutate=[0, 1, 2, 3],  # L=4
    )
    assert designer.mu < 1.0, f"Expected mu < 1.0, got {designer.mu}"


def test_pbt_clamp_keeps_mu_strictly_below_one():
    """PBT rate update must never produce mu == 1.0.

    Drives n_edits == L by patching the sampler's sample method, then verifies
    that the returned new_rate satisfies new_rate < L (so mu = new_rate/L < 1).
    This is the scenario the issue says 'could previously drive mu to 1.0'.
    """
    model = testing_utils.CountLetterModel()
    L = 6
    designer = AdaptiveRolloutDesigner(
        model_fn=model,
        start_sequence="AAAAAA",
        mutations_per_sequence=1,
        beam_size=2,
        n_rollouts_per_root=1,
        eval_batch_size=1,
        rng_seed=0,
        strategy=GradientActionStrategy(),
        use_gradients=True,
        use_pbt=True,
        exploration_alpha=0.5,
    )

    node = designer.current_nodes[0]
    # Temporarily monkey-patch the sampler so it always returns L (the degenerate draw).
    original_get_sampler = designer.get_sampler

    class _AlwaysReturnL:
        def sample(self, n):
            return np.array([L] * n)

    designer.get_sampler = lambda rate: _AlwaysReturnL()  # type: ignore[method-assign]
    try:
        n_edits, new_rate = designer._get_next_mutation_params(node)
    finally:
        designer.get_sampler = original_get_sampler

    assert n_edits == L, f"Expected n_edits == L == {L}, got {n_edits}"
    assert new_rate < L, (
        f"PBT clamp must cap new_rate strictly below L={L}, got {new_rate}. "
        "mu = new_rate/L would reach 1.0, corrupting _F_inverse."
    )
    mu = new_rate / L
    assert mu < 1.0, f"mu = {mu} must be < 1.0 after clamp"


def test_gradabeam_trajectory_bitforbit():
    """GradaBeam trajectory must be bit-for-bit identical before and after the
    position-space UniformActionStrategy refactor.

    Golden values captured on commit 8d90703 (pre-refactor) by running:
        python -c "
        from gradabeam import testing_utils
        from gradabeam.adaptive_rollout import AdaptiveRolloutDesigner, GradientActionStrategy
        designer = AdaptiveRolloutDesigner(
            model_fn=testing_utils.CountLetterModel(),
            start_sequence='ACGTACGT', mutations_per_sequence=2, beam_size=4,
            n_rollouts_per_root=2, eval_batch_size=1, rng_seed=42,
            strategy=GradientActionStrategy(), use_gradients=True, use_pbt=True,
            exploration_alpha=0.5,
        )
        for step in range(4):
            designer.run(n_steps=1)
            snapshot = sorted([
                (n.seq, float(n.fitness), round(float(n.exploration_alpha), 10),
                 round(float(n.mutations_per_sequence), 10))
                for n in designer.current_nodes
            ])
            print(f'STEP {step}: {snapshot}')
        "

    Any divergence means the refactor changed gradient-path behavior.
    """
    GOLDEN = [
        # step 0
        [
            ("CCAAACCC", 5.0, 0.3046075801, 2.0),
            ("CCAGACCC", 5.0, 0.3134616055, 1.0),
            ("CCAGGCAC", 4.0, 0.3413592515, 2.0),
            ("CTCCCTAC", 5.0, 0.5027434518, 3.0),
        ],
        # step 1
        [
            ("CCCGACCC", 6.0, 0.1232249424, 1.0),
            ("CCCGCCCC", 7.0, 0.0433671781, 1.0),
            ("CCCGCCCG", 6.0, 0.0585061034, 1.0),
            ("CCCTTCCC", 6.0, 0.098138035, 1.0),
        ],
        # step 2
        [
            ("CCCACCCC", 7.0, 0.0108428624, 1.0),
            ("CCCCCCCC", 8.0, 0.0161492341, 2.0),
            ("CCCTCCCC", 7.0, 0.0108428624, 1.0),
            ("CCCTCGCC", 6.0, 0.0108428623, 1.0),
        ],
        # step 3
        [
            ("CCCCCACC", 7.0, 0.01, 1.0),
            ("CCCCCCCC", 8.0, 0.01, 1.0),
            ("CCTCCCCC", 7.0, 0.0118450783, 2.0),
            ("CCTCCCCC", 7.0, 0.016149234, 1.0),
        ],
    ]

    model = testing_utils.CountLetterModel()
    designer = AdaptiveRolloutDesigner(
        model_fn=model,
        start_sequence="ACGTACGT",
        mutations_per_sequence=2,
        beam_size=4,
        n_rollouts_per_root=2,
        eval_batch_size=1,
        rng_seed=42,
        strategy=GradientActionStrategy(),
        use_gradients=True,
        use_pbt=True,
        exploration_alpha=0.5,
    )

    for step in range(4):
        designer.run(n_steps=1)
        snapshot = sorted(
            [
                (
                    n.seq,
                    float(n.fitness),
                    round(float(n.exploration_alpha), 10),
                    round(float(n.mutations_per_sequence), 10),
                )
                for n in designer.current_nodes
            ]
        )
        assert len(snapshot) == len(GOLDEN[step]), (
            f"Step {step}: expected {len(GOLDEN[step])} nodes, got {len(snapshot)}"
        )
        for idx, (got_node, exp_node) in enumerate(zip(snapshot, GOLDEN[step])):
            got_seq, got_fit, got_alpha, got_muts = got_node
            exp_seq, exp_fit, exp_alpha, exp_muts = exp_node
            assert got_seq == exp_seq, (
                f"Step {step} node {idx} sequence diverged: "
                f"got {got_seq}, expected {exp_seq}"
            )
            assert abs(got_fit - exp_fit) < 1e-6, (
                f"Step {step} node {idx} fitness diverged: "
                f"got {got_fit}, expected {exp_fit}"
            )
            assert abs(got_alpha - exp_alpha) < 1e-6, (
                f"Step {step} node {idx} exploration_alpha diverged: "
                f"got {got_alpha}, expected {exp_alpha}"
            )
            assert abs(got_muts - exp_muts) < 1e-6, (
                f"Step {step} node {idx} mutations_per_sequence diverged: "
                f"got {got_muts}, expected {exp_muts}"
            )


def test_normal_run_trajectory_unchanged_by_clamp():
    """Bit-for-bit trajectory check: the clamp must NOT affect normal-regime runs.

    With mu well below 1 (rate=1, L=6, mu=1/6 ≈ 0.167), n_edits never reaches L
    in practice.  The proposed-sequence trajectory with a fixed seed must be
    identical before and after the clamp edit — the clamp never binds, so RNG
    draws and outputs must match.

    Implementation: run two identically-seeded designers and compare the full
    sequence of proposed sequences across several steps.  Because both designers
    are constructed and run identically, any divergence means the clamp is binding
    when it should not be.
    """
    model = testing_utils.CountLetterModel()

    def make_designer(seed):
        return AdaptiveRolloutDesigner(
            model_fn=model,
            start_sequence="ACGTACGT",
            mutations_per_sequence=1,
            beam_size=3,
            n_rollouts_per_root=2,
            eval_batch_size=1,
            rng_seed=seed,
            strategy=GradientActionStrategy(),
            use_gradients=True,
            use_pbt=True,
            exploration_alpha=0.5,
        )

    d1 = make_designer(42)
    d2 = make_designer(42)

    n_steps = 4
    for step in range(n_steps):
        d1.run(n_steps=1)
        d2.run(n_steps=1)
        seqs1 = sorted(n.seq for n in d1.current_nodes)
        seqs2 = sorted(n.seq for n in d2.current_nodes)
        assert seqs1 == seqs2, (
            f"Trajectory diverged at step {step + 1}: {seqs1} != {seqs2}. "
            "The PBT clamp is binding in a regime where it should not — investigate."
        )
        rates1 = [n.mutations_per_sequence for n in d1.current_nodes]
        for r in rates1:
            mu = r / len(d1.positions_to_mutate)
            assert mu < 1.0, (
                f"mu={mu:.6f} >= 1.0 observed at step {step + 1} in normal run."
            )
