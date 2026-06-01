# GradaBeam

Gradient-guided adaptive beam search optimizer with Population Based Training (PBT), for biological sequence design.

Extracted from [nucleobench](https://github.com/move37-labs/nucleobench) as a standalone package.

## Overview

GradaBeam provides two sequence optimizers for directed evolution of nucleic acid (DNA/RNA) sequences:

| Optimizer | Gradient-guided | PBT | File |
|-----------|:--------------:|:---:|------|
| **GradaBeam** | Yes (TISM) | Yes | `gradabeam/optimizer.py` |
| **AdaBeam** | No (random) | No | `gradabeam/adabeam.py` |

Both use adaptive beam search with rollouts. Each round, the beam (a set of candidate sequences) is expanded by rolling out random or gradient-guided mutations, and the top-scoring candidates are kept.

**GradaBeam** additionally uses [TISM](https://en.wikipedia.org/wiki/In_silico_mutagenesis) (sequence-level gradient) to bias mutations toward positions and characters that increase model score, and adapts mutation rate on-the-fly via Population Based Training (PBT).

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
git clone https://github.com/move37-labs/gradabeam
cd gradabeam
pip install -e .
# Or, to include examples:
# pip install -e ".[examples]"
```

## Quick Start

### GradaBeam (gradient-guided)

```python
from gradabeam import GradaBeam

optimizer = GradaBeam(
    model_fn=your_model,          # callable: list[str] -> list[float]
    start_sequence="ACGTACGTACGT",
    mutations_per_sequence=2.0,
    beam_size=10,
    n_rollouts_per_root=4,
    exploration_alpha=0.05,       # 0 = fully gradient-guided, 1 = uniform random
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

A built-in demo CLI runs either optimizer using a `CountLetterModel` oracle
(maximizes C-content — useful for testing without a real model):

```bash
# GradaBeam demo
python -m gradabeam \
    --start_sequence AAAAAAAAAA \
    --n_steps 10 \
    --exploration_alpha 0.05 \
    --beam_size 5

# AdaBeam demo
python -m gradabeam \
    --optimizer adabeam \
    --start_sequence AAAAAAAAAA \
    --n_steps 10

# Load sequence from a local file
python -m gradabeam --start_sequence local://path/to/seq.txt --n_steps 5

# Load from the Zenodo Enformer dataset (requires network)
python -m gradabeam --start_sequence enformer://12 --n_steps 5
```

See all options:

```bash
python -m gradabeam --help
```

## Model Interface

The `model_fn` must be callable:

```python
def model_fn(sequences: list[str]) -> list[float]:
    """Return a fitness score per sequence. Higher is better."""
    ...
```

**GradaBeam** additionally requires the model to implement `tism_torch` for gradient-based mutation probabilities:

```python
def tism_torch(sequence: str, idxs: list[int] | None = None) -> torch.Tensor:
    """Return a (vocab_size, seq_len) TISM tensor."""
    ...

def get_tism(
    self, sequence: str, idxs: list[int] | None = None
) -> tuple[list[tuple[int, str]], np.ndarray]:
    """Return (pos_and_chars, logits) for the mutable positions."""
    ...
```

See `gradabeam.tism.TISMModelClass` for a reference implementation you can inherit from.

## Key Parameters

| Parameter | Applies to | Description |
|-----------|-----------|-------------|
| `start_sequence` | Both | Initial DNA/RNA string. |
| `mutations_per_sequence` | Both | Expected number of edits applied per mutation step. |
| `beam_size` | Both | Number of candidate sequences carried between rounds. |
| `n_rollouts_per_root` | Both | Random rollouts launched from each beam candidate per round. |
| `eval_batch_size` | Both | Sequences sent to the model per batch call. |
| `rng_seed` | Both | Seed for reproducibility. |
| `positions_to_mutate` | Both | Optional list of mutable positions (0-based). Defaults to all. |
| `max_rollout_len` | Both | Max rollout depth before stopping. |
| `exploration_alpha` | GradaBeam | Blend of uniform (1.0) vs. gradient-guided (0.0) mutations. |
| `use_pbt` | GradaBeam | Enable Population Based Training for adaptive mutation rate. |
| `skip_repeat_sequences` | AdaBeam | Skip already-evaluated sequences during rollouts. |

## Development

If you want to contribute to `gradabeam` or run the tests locally, we recommend using `mamba` (or `conda`) to set up the development environment:

```bash
micromamba create -f environment.yml
micromamba activate gradabeam
pytest gradabeam/
```

## Running Tests

```bash
pytest gradabeam/
```

Or with coverage:

```bash
pytest --cov=gradabeam gradabeam/
```

## License

See [nucleobench](https://github.com/move37-labs/nucleobench) for license information.
