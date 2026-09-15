# Reproducing the local algorithm audit

The committed [audit summary](audit_summary.json) is deliberately compact.
Full trajectory JSON is generated locally and should be written outside the
source tree:

```sh
audit_dir=/private/tmp/gradabeam-audit-results
mkdir -p "$audit_dir"

uv run --extra dev python benchmarks/overnight_audit.py \
  --source-dir /private/tmp/gradabeam-baseline-a67a090 \
  --tasks count_letter substring --optimizers adabeam gradabeam \
  --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 \
  --steps 30 --output "$audit_dir/baseline.json"

uv run --extra dev python benchmarks/overnight_audit.py \
  --tasks count_letter substring --optimizers adabeam gradabeam \
  --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 \
  --steps 30 --output "$audit_dir/candidate.json"
```

For equal-oracle-cost gates, append `--oracle-cost-budget N`; for equal-workload
gates use a fixed `--steps N`. Run paired baseline/candidate commands serially
and summarize medians by task and optimizer.
