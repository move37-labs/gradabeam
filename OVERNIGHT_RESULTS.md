# Overnight GrAdaBeam/AdaBeam Algorithm Audit

## Scope and baseline

- Audit branch: `overnight/gradabeam-algorithm-audit`
- Detached baseline: `a67a0902b2dc0ac68bddbaeff96425fefe0fde0f`
- No remote, cloud, merge, or reference-implementation operation was performed.
- `gradabeam/adabeam_reference.py` and `gradabeam/gradabeam_reference.py` remain unchanged.

The reusable local runner is `benchmarks/overnight_audit.py`. It records current
beam and best-ever fitness, normalized gain, forward/TISM calls, unique
evaluations, requested/actual Hamming edits, GrAdaBeam rate/alpha trajectories,
and wall time. The tracked [compact summary](benchmarks/audit_summary.json) and
[reproduction commands](benchmarks/AUDIT_REPRODUCTION.md) replace the former
multi-megabyte raw trajectories; generate those under `/private/tmp`.

## Accepted stack

1. `70e642b` — deterministic, seed-derived ranking for ties. Ranking/getter
   paths do not consume mutation RNG; this also removes process hash-order
   dependence.
2. `6ee1f57` — bounded unique best-ever archives, while retaining
   `get_current_samples()` for current-beam access.
3. `222e679` — immutable per-sequence raw TISM cache, no in-place mutation of
   model-owned logits, correct rollout-length accumulation, explicit mutable
   position/batch validation, and equality/hash consistency for transient node
   state.
4. The final typing-only follow-up casts NumPy edit counts to `int` in the
   optimized GrAdaBeam path.

The state/cache changes repair invariants. The archive changes result reporting
only; proposal generation, oracle cache behavior, and current-beam selection
remain distinct from archive retrieval.

## Paired quality gates

All benchmarks used beam size 4, four rollouts/root, max depth 30, fixed local
oracle-cost behavior, and paired seeds. Median normalized-gain and per-step wall
time ratios below are audit/baseline.

| Task | Optimizer | Gain ratio | Time ratio |
| --- | --- | ---: | ---: |
| Count-letter (30 seeds) | AdaBeam | 1.0185 | 1.0351 |
| Count-letter (30 seeds) | GrAdaBeam | 1.0000 | 0.9514 |
| Substring (30 seeds) | AdaBeam | 1.0000 | 1.0065 |
| Substring (30 seeds) | GrAdaBeam | 0.9924 | 1.0371 |
| NK, five fixed landscapes × 20 seeds | AdaBeam | 1.0000 | 1.0117 |
| NK, five fixed landscapes × 20 seeds | GrAdaBeam | 0.9937 | 0.9388 |

Every measured task retained more than 98% of baseline median gain and had less
than 5% median throughput regression. Five-seed smoke results for count-letter,
additive, substring, and NK are also retained in the raw artifacts.

## Timing

The local 3,000-position count-letter dummy suite used three serial repeats of
eight measured steps:

| Configuration | Median seconds/step |
| --- | ---: |
| AdaBeam | 0.157424 |
| GrAdaBeam, PBT disabled | 0.251249 |
| GrAdaBeam, PBT enabled | 0.047470 |

These are configuration timings, not cross-policy fitness comparisons. The
optional local BPNet package was absent (`ModuleNotFoundError: bpnet`), so the
ATAC directional validation was skipped rather than downloading/installing it.

## Rejected candidate

Sibling-action masking in GrAdaBeam was tested in a temporary isolated worktree.
It was rejected before promotion: `TestGradientAlignment` failed and emitted an
invalid probability-normalization warning on a degenerate distribution. The
temporary worktree and branch were removed; no production mutation kernel was
changed.

The PBT snapping behavior and AdaBeam's legacy four-base mutation policy remain
unchanged. In particular, no no-silent-edit AdaBeam setting was promoted without
the requested mutation-rate retuning study.

## Validation status and follow-up

- `pytest -q`: **55 passed, 8 skipped**.
- `ruff check .` and `ruff format --check .`: **passed**.
- `mypy gradabeam`: two remaining errors in the intentionally untouched
  `gradabeam/gradabeam_reference.py` at lines 326 and 332. The optimized paths
  are type-clean; these are pre-existing reference-file annotations and were not
  modified to preserve the byte-for-byte constraint.

Recommended larger follow-up: run the same paired framework with a preinstalled
local BPNet environment; separately test no-silent AdaBeam at corrected rates
{0.75, 1.0, 1.25, 1.5, 2.0}; and only revisit position-level masking after a
kernel that preserves finite probability mass and the gradient-alignment test is
available.

## Stage 2: bounded ranking and fair-gate rerun

Stage 1's full sort was a draft. The accepted replacement is a bounded
`heapq.nsmallest` top-k ordered by fitness and a process-independent versioned
seeded sequence hash, `xxh64(f"{sequence}\\0{seed + 1}")`. It does not consume
mutation RNG. The evaluator now supports repeated fixed-workload runs and a
post-initialization oracle-cost budget.

The following isolated commits were tested against `a67a090` on count-letter,
substring, and NK (five seeds, three repeats, 20 fixed-workload steps and a
1,000 proposal-cost budget):

| Candidate | Commit | Fixed-cost result | Decision |
| --- | --- | --- | --- |
| ascending `xxh64(seed, sequence)` | `12bcb64` | AdaBeam/count-letter 0.9643× gain | Reject |
| reverse priority | `d60018c` | AdaBeam/count-letter 0.9643× gain | Reject |
| sequence-first, unsalted | `4e0da83` | AdaBeam/count-letter 0.9643× gain | Reject |
| sequence-first, version salt 1 | `0f5362a` | min gain 0.9848×; max time 1.0115× | Accept |

The accepted candidate's equal-workload minimum gain ratio was 0.9848× and
maximum time ratio 1.0115×. At equal proposal cost, minimum gain was 1.0000×
and maximum time ratio 0.9988×. Raw repeated JSON is stored outside the source
tree in `/private/tmp/audit_topk_*`; the reproduction document gives the
commands.

## Stage 3: historical unification and PBT audit

Historical unification branches were inspected and executed in detached local
worktrees. The position-space unified refactor `b10736e` replaces the split
optimizers with 2,399 added lines and fails 24/81 tests under the current
xxhash API; the action-space follow-up `0e2720a` changes 1,817 lines and fails
26/81 tests for the same `xxhash.xxh64(str)` incompatibility. Both violate the
conservative-unification size constraint and are rejected without transplant.

The small historical PBT perturbation policy was recreated as isolated commit
`233953d`: with probability 0.2 it multiplies the inherited rate by 0.8 or 1.2,
clamps it, and uses a dedicated PBT RNG so mutation sampling is unaffected.
Focused tests passed (11 passed, 3 skipped). Against current snap PBT over five
seeds × three repeats, its workload timing reached 1.0865×/1.0849× on
count-letter/substring, and its fixed-cost NK gain was 0.9516×. It is rejected;
snap remains the production policy. Raw outputs are `/private/tmp/audit_pbt_*`.

## Stage 4: AdaBeam silent-edit audit

Candidate commits `5124b04` and `6137f1c` add an opt-in no-silent mutation
operator and sweep harness. Its correctness test passed (three requested edits
produce exactly three Hamming changes). The legacy policy's three-repeat
workload baseline had 1.5824 requested versus 1.1825 actual edits on average
(0.7473 actual/requested; 9,360 silent proposals); no-silent mode was exactly
1.0 with zero silent proposals.

No corrected rate cleared the quality screen. Each used count-letter,
substring, and NK with five seeds under fixed workload and 1,000 proposal-cost
budgets:

| Corrected rate | Workload min gain ratio | Fixed-cost min gain ratio | Decision |
| ---: | ---: | ---: | --- |
| 0.75 | 0.879 | 0.902 | Reject |
| 1.00 | 0.865 | 0.903 | Reject |
| 1.25 | 0.905 | 0.929 | Reject |
| 1.50 | 0.861 | 0.878 | Reject |
| 2.00 | 0.846 | 0.893 | Reject |

These are screen results (one repeat per seed); none qualified for the larger
repeated promotion gate. Raw outputs are `/private/tmp/audit_adabeam_*`.

## Stage 5: corrected sibling-action masking rerun

The earlier masking experiment was rerun as isolated commit `c5581ec`, fixing
the final-step zero-normalization bug. It samples one action at a time, masks
all sibling actions for each selected position, and carries a deterministic
gradient-alignment test plus a distinct-position Hamming-edit test. Focused
tests passed (10 passed, 3 skipped).

Against the accepted action-space kernel (five seeds × three repeats), the
candidate failed both gates: equal-workload gain ratios were 1.0000,
0.9846, and 0.9651 for count-letter, substring, and NK, with up to 1.1101×
time; fixed-cost ratios were 1.0000, 0.9773, and 0.9974 with up to 1.1396×
time. It is rejected. Raw outputs are `/private/tmp/audit_action_*`.

## Stage 6: minimal position-space kernel rerun

Candidate `04c3c20` is the conservative position-space counterpart to the
historical unified implementation: it marginalizes action probabilities by
position, samples distinct positions without replacement, uses a uniform
alternative base, masks all selected positions, and computes α posterior from
position rather than action probability. Its first implementation exposed a
stale-action lookup failure; the repaired candidate passes focused tests (10
passed, 3 skipped).

It does not clear empirical gates. Across five seeds × three repeats, equal
workload timing was 2.7077× on count-letter, 1.7386× on substring, and 1.0938×
on NK. At equal proposal cost, substring gain was 0.9773× and NK gain 0.9249×,
with 1.70–2.66× timing. Reject it and retain action-space GrAdaBeam. Raw
outputs are `/private/tmp/audit_position_*`.

## Audit checklist status

All locally executable checklist items were attempted: bounded deterministic
ranking; repeated equal-workload and fixed-cost gates; compact artifact
replacement; both historical unification snapshots; corrected sibling-action
and position-space reruns; AdaBeam no-silent mutation-rate sweep; and PBT
snap/perturb comparison. Only the bounded top-k ranking was promoted. No
external dependency or cloud action was used; BPNet remains the sole skipped
directional check because its optional package is unavailable locally.
