"""Tests for utils.

To test:
```zsh
pytest gradabeam/opt_utils_test.py
```
"""

import numpy as np

from gradabeam import ada_utils
from gradabeam import opt_utils as utils


def test_get_locations_to_edit():
    locs = utils.get_locations_to_edit(
        positions_to_mutate=[0, 1],
        random_n_loc=2,
        rng=np.random.default_rng(42),
        method="random",
    )
    assert len(locs) == 2


def _make_node(seq: str, fitness: float) -> ada_utils.RolloutNode:
    return ada_utils.RolloutNode(seq=seq, fitness=np.float32(fitness))


_SORT_KEY = lambda x: (x.fitness, x.seq)


def test_best_ever_top_k_by_fitness():
    """update with out-of-order fitnesses; best(k) returns k highest, descending."""
    be = utils.BestEver(sort_key=_SORT_KEY, capacity=10)
    nodes = [
        _make_node("AAA", 0.1),
        _make_node("CCC", 0.9),
        _make_node("GGG", 0.5),
        _make_node("TTT", 0.7),
        _make_node("ACG", 0.3),
    ]
    for n in nodes:
        be.update([n])

    result = be.best(3)
    assert len(result) == 3
    assert [r.seq for r in result] == ["CCC", "TTT", "GGG"]
    # Verify descending order.
    fitnesses = [r.fitness for r in result]
    assert fitnesses == sorted(fitnesses, reverse=True)


def test_best_ever_deduplication():
    """Updating with the same seq twice yields a single entry."""
    be = utils.BestEver(sort_key=_SORT_KEY, capacity=10)
    node_a = _make_node("AAA", 0.5)
    be.update([node_a])
    be.update([node_a])
    be.update([_make_node("AAA", 0.5)])  # same seq, same fitness

    assert len(be._by_seq) == 1
    result = be.best(5)
    assert len(result) == 1
    assert result[0].seq == "AAA"


def test_best_ever_bounded_capacity():
    """After many updates with more distinct seqs than capacity, dict stays bounded."""
    capacity = 3
    be = utils.BestEver(sort_key=_SORT_KEY, capacity=capacity)
    for i in range(20):
        be.update([_make_node(f"S{i:03d}", float(i))])

    assert len(be._by_seq) <= capacity
    # The kept entries should be the highest-fitness ones.
    result = be.best(capacity)
    fitnesses = [r.fitness for r in result]
    assert fitnesses == sorted(fitnesses, reverse=True)
    assert min(fitnesses) >= 17.0  # top 3 of 0..19 are 17, 18, 19


def test_best_ever_lazy_capacity_growth():
    """Requesting more than capacity returns whatever was kept, not an error."""
    be = utils.BestEver(sort_key=_SORT_KEY, capacity=2)
    for i in range(10):
        be.update([_make_node(f"S{i:03d}", float(i))])

    # After inserting 10 nodes with capacity=2, only 2 are kept.
    assert len(be._by_seq) == 2

    # best(5) should grow capacity and return whatever is available (2 nodes).
    result = be.best(5)
    assert len(result) == 2
    assert be._capacity == 5
