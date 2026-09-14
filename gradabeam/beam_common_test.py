"""Tests for shared beam helpers."""

from types import SimpleNamespace

import numpy as np

from gradabeam import beam_common


def test_accumulate_first_observed_keeps_insertion_order():
    store: dict[str, str] = {}
    beam_common.accumulate_first_observed(store, "a", "first")
    beam_common.accumulate_first_observed(store, "b", "second")
    beam_common.accumulate_first_observed(store, "a", "later")
    assert list(store.values()) == ["first", "second"]


def test_rank_nodes_by_fitness_uses_only_tie_rng():
    mutation_rng = np.random.default_rng(0)
    before = mutation_rng.bit_generator.state
    nodes = [
        SimpleNamespace(seq="A", fitness=1.0),
        SimpleNamespace(seq="B", fitness=1.0),
        SimpleNamespace(seq="C", fitness=2.0),
    ]
    ranked = beam_common.rank_nodes_by_fitness(
        nodes, limit=3, tie_rng=np.random.default_rng(7)
    )
    assert ranked[0].seq == "C"
    assert mutation_rng.bit_generator.state == before


def test_rank_nodes_by_fitness_is_deterministic_for_ties():
    nodes = [
        SimpleNamespace(seq="A", fitness=1.0),
        SimpleNamespace(seq="B", fitness=1.0),
        SimpleNamespace(seq="C", fitness=1.0),
    ]
    first = [
        n.seq
        for n in beam_common.rank_nodes_by_fitness(nodes, 3, np.random.default_rng(11))
    ]
    second = [
        n.seq
        for n in beam_common.rank_nodes_by_fitness(nodes, 3, np.random.default_rng(11))
    ]
    assert first == second


def test_filter_accepted_children_uses_inclusive_geq():
    parents = [SimpleNamespace(fitness=1.0), SimpleNamespace(fitness=2.0)]
    children = [SimpleNamespace(fitness=1.0), SimpleNamespace(fitness=1.5)]
    kept, terminated = beam_common.filter_accepted_children(children, parents, 3)
    assert kept == [children[0]]
    assert terminated == [3]


def test_sample_sequences_from_nodes_is_a_prefix():
    nodes = [SimpleNamespace(seq="AAA"), SimpleNamespace(seq="CCC")]
    assert beam_common.sample_sequences_from_nodes(nodes, 1) == ["AAA"]


def test_resolve_positions_to_mutate_defaults_to_all():
    assert beam_common.resolve_positions_to_mutate("ACGT", None) == [0, 1, 2, 3]
    assert beam_common.resolve_positions_to_mutate("ACGT", [1, 2]) == [1, 2]


def test_adabeam_candidate_key_includes_fitness():
    a = SimpleNamespace(seq="AAAA", fitness=1.0)
    b = SimpleNamespace(seq="AAAA", fitness=2.0)
    c = SimpleNamespace(seq="CCCC", fitness=1.0)
    assert beam_common.adabeam_candidate_key(a) == ("AAAA", 1.0)
    assert beam_common.adabeam_candidate_key(a) != beam_common.adabeam_candidate_key(b)
    assert beam_common.adabeam_candidate_key(a) != beam_common.adabeam_candidate_key(c)
    store: dict[tuple, object] = {}
    beam_common.accumulate_first_observed(
        store, beam_common.adabeam_candidate_key(a), a
    )
    beam_common.accumulate_first_observed(
        store, beam_common.adabeam_candidate_key(b), b
    )
    assert list(store.values()) == [a, b]


def test_gradabeam_candidate_key_includes_pbt_and_edits():
    a = SimpleNamespace(
        seq="AAAA",
        fitness=1.0,
        edits_since_root=1,
        mutations_per_sequence=1.0,
        exploration_alpha=0.05,
    )
    b = SimpleNamespace(
        seq="AAAA",
        fitness=1.0,
        edits_since_root=1,
        mutations_per_sequence=2.0,
        exploration_alpha=0.05,
    )
    c = SimpleNamespace(
        seq="AAAA",
        fitness=1.0,
        edits_since_root=2,
        mutations_per_sequence=1.0,
        exploration_alpha=0.05,
    )
    assert beam_common.gradabeam_candidate_key(
        a
    ) != beam_common.gradabeam_candidate_key(b)
    assert beam_common.gradabeam_candidate_key(
        a
    ) != beam_common.gradabeam_candidate_key(c)


def test_rank_nodes_by_fitness_is_nonincreasing():
    nodes = [
        SimpleNamespace(seq="A", fitness=1.0),
        SimpleNamespace(seq="B", fitness=3.0),
        SimpleNamespace(seq="C", fitness=2.0),
    ]
    ranked = beam_common.rank_nodes_by_fitness(nodes, 3, np.random.default_rng(0))
    fitnesses = [n.fitness for n in ranked]
    assert fitnesses == sorted(fitnesses, reverse=True)
    assert ranked[0].seq == "B"
