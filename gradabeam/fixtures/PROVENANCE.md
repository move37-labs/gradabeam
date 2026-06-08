# Golden Fixture Provenance

## Do not regenerate against post-refactor code

The JSON files in this directory are **static baselines** committed at a particular point when the code is deterministic. It is meant to be used during heavy refactors that are meant to not change performance.  Regenerating them against post-refactor code would
replace genuine published-AdaBeam behavior with refactored behavior, silently
invalidating the equivalence gate (`test_adabeam_equivalence`).

Only regenerate if you are deliberately updating the baseline (e.g. after
changing the oracle or sampler), and document the new anchor commit below.

## Anchor commit

| Field | Value |
|---|---|
| Commit hash | `982c75a` |
| Committed | Before any Plan-01-1b refactor source existed |
| Operator | `generate_random_mutant_v2` (4-base sampling, ~25 % silent edits) |
| AdaBeam flag | `allow_silent_edits=True` (legacy path) |
| `rng_seed` | `42` |
| `start_sequence` | `"AAAAAA"` |
| `n_steps` | `3` |
| `beam_size` | `10` |
| `n_rollouts_per_root` | `4` |

## Oracles

| File | Oracle |
|---|---|
| `adabeam_golden_count_letter.json` | `testing_utils.CountLetterModel()` |
| `adabeam_golden_substring_count.json` | `CountSubstringModel(substring="AC")` |

## Regeneration guard

`generate_golden.py` contains two runtime guards that prevent accidental
regeneration with a non-silent configuration:

1. `_CapturingAdaBeam.__init__` asserts `allow_silent_edits=True`.
2. `generate_fixture` asserts `opt.strategy.is_legacy()` after construction.

Both guards will raise `AssertionError` if you try to run the generator
without explicitly setting the legacy flag.
