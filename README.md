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

This package provides two sequence optimizers for designing biomolecular sequences:

| Optimizer | Gradient-guided | PBT | File |
|-----------|:--------------:|:---:|------|
| **GradaBeam** | Yes | Optional | `gradabeam/gradabeam_optimizer.py` |
| **AdaBeam** | No (random) | No | `gradabeam/adabeam_optimizer.py` |

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
    model_fn=your_model,          # callable: list[str] -> list[float], minimized
    start_sequence="ACGTACGTACGT",
    mutations_per_sequence=2.0,
    beam_size=10,
    n_rollouts_per_root=4,
    exploration_alpha=0.5,       # 0.0 = fully gradient-guided, 1.0 = uniform random
    use_pbt=True,                 # adapt the mutation rate via Population Based Training
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

## Command-Line Interface

The CLI runs either optimizer against an oracle that you supply via `--oracle_script`. The script must define a
`make_oracle()` function (see [`oracles/template.py`](oracles/template.py) for a starting point). Several ready-to-run
oracles ship in the [`oracles/`](oracles) directory:

- [`oracles/count_letter.py`](oracles/count_letter.py) — maximizes the count of a target letter (tiny, no extra deps).
- [`oracles/substring_count.py`](oracles/substring_count.py) — maximizes occurrences of a target substring.
- [`oracles/bpnet.py`](oracles/bpnet.py) — a real BPNet transcription-factor-binding model (requires the `examples` extra).

You must pass `--beam_size`, `--mutations_per_sequence`, and `--n_rollouts_per_root`, plus exactly one of `--n_steps`
or `--time_budget`. For `--optimizer gradabeam` you must also pass `--use_pbt`. Any extra flags are forwarded to the
oracle's `make_oracle()`.

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

**GradaBeam** additionally requires the oracle to provide gradient-based mutation information via `get_tism`
(internally it also relies on a `tism_torch` method):

```python
def get_tism(
    self, sequence: str, idxs: list[int] | None = None
) -> tuple[list[tuple[int, str]], np.ndarray]:
    """Return (pos_and_chars_to_mutate, logits) for the mutable positions."""
    ...
```

The easiest way to satisfy this is to inherit from `gradabeam.tism.TISMModelClass`, which implements `get_tism` and
`tism_torch` for you given a small set of model hooks (`vocab`, `vocab_array`, `vocab_to_idx`, and
`inference_on_tensor`). See [`oracles/bpnet.py`](oracles/bpnet.py) and
[`oracles/substring_count.py`](oracles/substring_count.py) for reference implementations.

## Key Parameters

| Parameter | Applies to | Description |
|-----------|-----------|-------------|
| `start_sequence` | Both | Initial DNA string (alphabet `ACGT`). |
| `mutations_per_sequence` | Both | Expected number of edits applied per mutation step. |
| `beam_size` | Both | Number of candidate sequences carried between rounds. |
| `n_rollouts_per_root` | Both | Rollouts launched from each beam candidate per round. |
| `eval_batch_size` | Both | Sequences sent to the model per batch call. |
| `rng_seed` | Both | Seed for reproducibility. |
| `positions_to_mutate` | Both | Optional list of mutable positions (0-based). Defaults to all. |
| `max_rollout_len` | Both | Max rollout depth before stopping. |
| `exploration_alpha` | GradaBeam | Blend of gradient-guided (0.0) vs. uniform-random (1.0) mutations. |
| `use_pbt` | GradaBeam | Enable Population Based Training for an adaptive mutation rate. |
| `gradient_prob_cap` | GradaBeam | Per-action probability cap applied after softmax. |
| `max_logit` | GradaBeam | Dynamic temperature ceiling for TISM logit scaling. |
| `skip_repeat_sequences` | AdaBeam | Skip already-evaluated sequences during rollouts. |

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
