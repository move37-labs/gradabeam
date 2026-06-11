"""Tests for utils.

To test:
```zsh
pytest gradabeam/opt_utils_test.py
```
"""

import numpy as np

from gradabeam import opt_utils as utils


def test_get_locations_to_edit():
    locs = utils.get_locations_to_edit(
        positions_to_mutate=[0, 1],
        random_n_loc=2,
        rng=np.random.default_rng(42),
        method="random",
    )
    assert len(locs) == 2

