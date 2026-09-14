# GrAdaBeam

[![bioRxiv](https://img.shields.io/badge/bioRxiv-10.1101%2F2025.06.20.660785-b31b1b.svg)](https://www.biorxiv.org/content/10.1101/2025.06.20.660785)
[![codecov](https://codecov.io/gh/move37-labs/gradabeam/graph/badge.svg)](https://codecov.io/gh/move37-labs/gradabeam)

Gradient-guided adaptive beam search optimizer with Population Based Training (PBT), for nucleic acid sequence design.

GrAdaBeam is the design algorithm introduced in
[**"GrAdaBeam: Combining model gradients with evolutionary search for generalizable nucleic acid design"**](https://www.biorxiv.org/content/10.1101/2025.06.20.660785).
It unifies the broad exploration of evolutionary search with the precise guidance of model gradients, and statistically
outperformed seven other design algorithms across the [NucleoBench](https://github.com/move37-labs/nucleobench)
benchmark.

## Overview

This package provides sequence optimizers for designing biomolecular sequences:

| Optimizer | Gradient-guided | PBT | File |
|-----------|:--------------:|:---:|------|
| **GradaBeam** | Yes | Optional | `gradabeam/gradabeam_optimizer.py` |
| **AdaBeam** | No (random) | No | `gradabeam/adabeam_optimizer.py` |
| **GradaBeamReference** | Yes | Optional | `gradabeam/gradabeam_reference.py` |
| **AdaBeamReference** | No (random) | No | `gradabeam/adabeam_reference.py` |

`AdaBeam` and `GradaBeam` are the current implementations. `AdaBeamReference` and
`GradaBeamReference` are additive, paper-result ports of the internal NucleoBench
designers and are intended to reproduce those published trajectories. They reuse
this package's mutation helpers, but keep the original ranking, initialization,
sampler-caching, PBT, and fitness-dtype behavior.

Both use adaptive beam search with rollouts. Each round, the beam (a set of candidate sequences) is expanded by rolling out random or gradient-guided mutations, and the top-scoring candidates are kept.

**GradaBeam** additionally uses [TISM](https://en.wikipedia.org/wiki/In_silico_mutagenesis) (a sequence-level
gradient) to bias mutations toward positions and characters that improve the model score, and can adapt its mutation
rate on-the-fly via Population Based Training (PBT).

> **Scoring convention:** the optimizers **minimize** the oracle output — *lower is better*. The bundled demo oracles
> negate their underlying quantity (e.g. a letter count) so that minimizing the score maximizes that quantity.

## Installation

From PyPI:

```bash
pip install gradabeam
```

To run the built-in examples (e.g. BPNet), install the `examples` extra:

```bash
pip install "gradabeam[examples]"
```

From source:

```bash
git clone https://github.com/move37-labs/gradabeam.git
cd gradabeam
pip install -e .
# Or, to include the example oracles (e.g. BPNet):
# pip install -e ".[examples]"
```

## Quick Start

The optimizers take a `model_fn` oracle: a callable mapping `list[str] -> list[float]` whose output is **minimized**
(lower is better). GradaBeam additionally requires the oracle to expose gradient information (see
[Oracle interface](#oracle-interface)).

### GradaBeam (gradient-guided)

```python
from gradabeam import GradaBeam

optimizer = GradaBeam(
    model_fn=your_model,  # callable: list[str] -> list[float], minimized
    start_sequence="ACGTACGTACGT",
    mutations_per_sequence=2.0,
    beam_size=10,
    n_rollouts_per_root=4,
    exploration_alpha=0.5,  # 0.0 = fully gradient-guided, 1.0 = uniform random
    use_pbt=True,  # adapt the mutation rate via Population Based Training
)

optimizer.run(n_steps=20)
top_sequences = optimizer.get_samples(n_samples=5)
print(top_sequences)
```

### AdaBeam (gradient-free)

```python
from gradabeam import AdaBeam

optimizer = AdaBeam(
    model_fn=your_model,
    start_sequence="ACGTACGTACGT",
    mutations_per_sequence=2.0,
    beam_size=10,
    n_rollouts_per_root=4,
    eval_batch_size=1,
    skip_repeat_sequences=True,
)

optimizer.run(n_steps=20)
top_sequences = optimizer.get_samples(n_samples=5)
```

### Paper-result reference designers

```python
from gradabeam import AdaBeamReference, GradaBeamReference

adabeam_ref = AdaBeamReference(
    model_fn=your_model,
    start_sequence="ACGTACGTACGT",
    mutations_per_sequence=2.0,
    beam_size=10,
    n_rollouts_per_root=4,
    eval_batch_size=1,
    skip_repeat_sequences=True,
)

gradabeam_ref = GradaBeamReference(
    model_fn=your_model,
    start_sequence="ACGTACGTACGT",
    mutations_per_sequence=2.0,
    beam_size=10,
    n_rollouts_per_root=4,
    exploration_alpha=0.5,
    use_pbt=True,
)
```

These classes were ported from NucleoBench:

- AdaBeam: `nucleobench/optimizations/ada/adabeam/adabeam.py` blob `767402c810da5b8df45f81b3238897d7bb194af0`
- GradaBeam: `nucleobench/optimizations/ada/gradabeam/gradabeam.py` blob `569ea54af6331d07edcd4bb294cc09186b451824`

## Command-Line Interface

The CLI runs either optimizer against an oracle that you supply via `--oracle_script`. The script must define a
`make_oracle()` function (see [`oracles/template.py`](oracles/template.py) for a starting point). Several ready-to-run
oracles ship in the [`oracles/`](oracles) directory:

- [`oracles/count_letter.py`](oracles/count_letter.py) — maximizes the count of a target letter (tiny, no extra deps).
- [`oracles/substring_count.py`](oracles/substring_count.py) — maximizes occurrences of a target substring.
- [`oracles/bpnet.py`](oracles/bpnet.py) — a real BPNet transcription-factor-binding model (requires the `examples` extra).

You must pass `--beam_size`, `--mutations_per_sequence`, and `--n_rollouts_per_root`, plus exactly one of `--n_steps`
or `--time_budget`. For `--optimizer gradabeam` or `--optimizer gradabeam-reference` you must also pass `--use_pbt`.
`--optimizer adabeam-reference` and `--optimizer gradabeam-reference` select the NucleoBench paper-result designers.
Any extra flags are forwarded to the oracle's `make_oracle()`.

```bash
# GradaBeam demo: maximize C-content with the count_letter oracle
python -m gradabeam \
    --oracle_script oracles/count_letter.py \
    --start_sequence AAAAAAAAAA \
    --n_steps 10 \
    --beam_size 5 \
    --mutations_per_sequence 2.0 \
    --n_rollouts_per_root 4 \
    --exploration_alpha 0.5 \
    --use_pbt True

# AdaBeam demo: maximize occurrences of a target substring (oracle arg passed through)
python -m gradabeam \
    --optimizer adabeam \
    --oracle_script oracles/substring_count.py \
    --start_sequence AAAAAAAAAAAAAAAAAAAA \
    --n_steps 10 \
    --beam_size 2 \
    --mutations_per_sequence 1.0 \
    --n_rollouts_per_root 4 \
    --substring ATGTC

# GradaBeam with the BPNet neural-network oracle on a real biological sequence
# (requires `pip install -e ".[examples]"`); --protein is forwarded to the oracle
python -m gradabeam \
    --oracle_script oracles/bpnet.py \
    --start_sequence local://ATAC_start_seq.txt \
    --time_budget 300 \
    --beam_size 2 \
    --mutations_per_sequence 2.0 \
    --n_rollouts_per_root 4 \
    --use_pbt False \
    --protein ATAC

# Paper-result AdaBeam from NucleoBench
python -m gradabeam \
    --optimizer adabeam-reference \
    --oracle_script oracles/count_letter.py \
    --start_sequence AAAAAAAAAA \
    --n_steps 10 \
    --beam_size 5 \
    --mutations_per_sequence 2.0 \
    --n_rollouts_per_root 4
```

The `--start_sequence` (and `--positions_to_mutate`) flags support two special prefixes:

```bash
# Load the sequence from a local file
python -m gradabeam --oracle_script oracles/count_letter.py \
    --start_sequence local://path/to/seq.txt \
    --n_steps 5 --beam_size 5 --mutations_per_sequence 2.0 --n_rollouts_per_root 4 --use_pbt True
```

See all options:

```bash
python -m gradabeam --help
```

## Oracle interface

The `model_fn` oracle must be callable and return one score per sequence, **lower is better**:

```python
def __call__(self, sequences: list[str]) -> list[float]:
    """Return a score per sequence. The optimizer minimizes this (lower is better)."""
    ...
```

**GradaBeam** additionally requires the oracle to provide gradient-based mutation information via `tism_torch`
(used by `ModelWrapper.get_tism`). The easiest way to satisfy this is to inherit from
`gradabeam.tism.TISMModelClass`, which implements `get_tism` and `tism_torch` given a small set of model hooks
(`vocab`, `vocab_array`, `vocab_to_idx`, and `inference_on_tensor`). See [`oracles/template.py`](oracles/template.py),
[`oracles/bpnet.py`](oracles/bpnet.py), and [`oracles/substring_count.py`](oracles/substring_count.py).

## Key Parameters

| Parameter | Applies to | Description |
|-----------|-----------|-------------|
| `start_sequence` | All | Initial DNA string (alphabet `ACGT`). |
| `mutations_per_sequence` | All | Expected number of edits applied per mutation step. |
| `beam_size` | All | Number of candidate sequences carried between rounds. |
| `n_rollouts_per_root` | All | Rollouts launched from each beam candidate per round. |
| `eval_batch_size` | All | Sequences sent to the model per batch call. AdaBeam and both reference designers support values > 1. GradaBeam requires `1`. |
| `rng_seed` | All | Seed for reproducibility. |
| `positions_to_mutate` | All | Optional list of mutable positions (0-based). Defaults to all. |
| `max_rollout_len` | All | Max rollout depth before stopping. |
| `exploration_alpha` | GradaBeam / GradaBeamReference | Blend of gradient-guided (0.0) vs. uniform-random (1.0) mutations. |
| `use_pbt` | GradaBeam / GradaBeamReference | Enable Population Based Training for an adaptive mutation rate. |
| `gradient_prob_cap` | GradaBeam / GradaBeamReference | Per-action probability cap applied after softmax. |
| `max_logit` | GradaBeam / GradaBeamReference | Dynamic temperature ceiling for TISM logit scaling. |
| `skip_repeat_sequences` | AdaBeam / AdaBeamReference | Skip already-evaluated sequences during rollouts. |

## Development

To contribute or run the tests locally, we recommend using `micromamba` (or `mamba`/`conda`) to set up the
development environment:

```bash
micromamba create -f environment.yml
micromamba activate gradabeam
pytest gradabeam/
```

Or with coverage:

```bash
pytest --cov=gradabeam gradabeam/
```

### Performance regression testing

A self-comparison harness detects step-throughput regressions by benchmarking
both the current `HEAD` and a baseline Git ref **on the same machine**, then
failing if either designer is more than 1.20× slower than the baseline.

**Prerequisites:** the benchmark loads BPNet, which requires the `examples` extra:

```bash
pip install -e ".[dev,examples]"
```

**Run locally** (against the merge-base of your current branch and `main`):

```bash
python benchmarks/perf_regression.py --designer both
```

Key flags:

| Flag | Default | Description |
|---|---|---|
| `--designer` | `both` | `gradabeam`, `adabeam`, or `both` |
| `--baseline-ref` | *(merge-base)* | Explicit Git ref to compare against |
| `--base-branch` | `main` | Branch used to compute the merge-base |
| `--n-repeats` | `5` | Measured repeats per side (after warmup) |
| `--steps-per-repeat` | `200` | Optimizer steps per repeat |
| `--max-slowdown` | `1.20` | Ratio threshold that triggers a failure |
| `--json-out` | *(none)* | Write full results to a JSON file |

**In CI** the workflow `.github/workflows/perf-regression.yml` runs on every
pull request targeting `main`. It runs `gradabeam` and `adabeam` as parallel
jobs (~4 min wall-clock) and uploads the JSON results as artifacts.

**`git bisect` recipe** — useful for finding the exact commit that introduced a
regression. Copy the script outside the tree first so it survives checkout:

```bash
cp benchmarks/perf_regression.py /tmp/perf_regression.py

# Mark the known-good and known-bad commits, then let bisect run the driver.
# The driver exits 0 (good) when performance is within tolerance,
# and exits 1 (bad) when a regression is detected.
git bisect start
git bisect bad HEAD
git bisect good <last-known-good-sha>
git bisect run python /tmp/perf_regression.py \
    --designer both \
    --baseline-ref <last-known-good-sha> \
    --n-repeats 3 \
    --steps-per-repeat 50
```

## Citation

If you use GrAdaBeam, please cite:

```bibtex
@article{shor2025gradabeam,
  author  = {Shor, Joel and Strand, Erik and McLean, Cory Y.},
  title   = {{GrAdaBeam: Combining model gradients with evolutionary search for generalizable nucleic acid design}},
  journal = {bioRxiv},
  year    = {2025},
  doi     = {10.1101/2025.06.20.660785},
  url     = {https://www.biorxiv.org/content/10.1101/2025.06.20.660785}
}
```

If you use the [NucleoBench](https://github.com/move37-labs/nucleobench) benchmark or the AdaBeam algorithm, please
also cite:

```bibtex
@article{shor2025nucleobench,
  author  = {Shor, Joel and Strand, Erik and McLean, Cory Y.},
  title   = {{NucleoBench: A Large-Scale Benchmark of Neural Nucleic Acid Design Algorithms}},
  journal = {bioRxiv},
  year    = {2025},
  doi     = {10.1101/2025.06.20.660785},
  url     = {https://www.biorxiv.org/content/10.1101/2025.06.20.660785}
}
```

## License

Apache License 2.0, consistent with the upstream
[nucleobench](https://github.com/move37-labs/nucleobench/blob/main/LICENSE) project.
