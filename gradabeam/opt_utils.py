"""Common utils for optimization algorithms."""

import numpy as np


def get_locations_to_edit(
    positions_to_mutate: list[int],
    random_n_loc: int,
    rng: np.random.Generator,
    method: str,
) -> list[int]:
    """Selects locations to edit."""
    assert random_n_loc > 0
    assert random_n_loc <= len(positions_to_mutate)

    if method == "all":
        return positions_to_mutate
    elif method == "random":
        return rng.choice(positions_to_mutate, size=random_n_loc, replace=False)
    else:
        raise ValueError("Arg not recognized.")


def generate_single_mutant_multiedits(
    base_str: str,
    locs_to_edit: list[int],
    alphabet: list[str],
    rng: np.random.Generator,
) -> str:
    """Return a mutant."""
    assert isinstance(alphabet, list)
    assert len(alphabet) > 1
    mutant = list(base_str)

    for i in locs_to_edit:
        # TODO(joelshor): This should be `rng.choice(set(alphabet) - mutant[i])`,
        # but we want to keep it for consistency with the publication.
        # Expect this behavior to change in the future.
        mutant[i] = rng.choice(alphabet)
    return "".join(mutant)
