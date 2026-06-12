"""Performance benchmark for GradaBeam and AdaBeam designers.

Measures step throughput for regression tracking. Run directly with:

    python benchmarks/perf_benchmark.py

Results are printed to stdout; no pass/fail threshold is enforced so that
numbers can be compared across runs and machines without false CI failures.
"""

import sys
import time

from gradabeam import GradaBeam, AdaBeam

sys.path.insert(0, "oracles")
from bpnet import BPNet  # noqa: E402


N_WARMUP_STEPS = 20
N_MEASURED_STEPS = 200

_START_SEQUENCE = "A" * 3000
_MODEL = BPNet(protein="GATA2")

_ADABEAM_KWARGS = dict(
    model_fn=_MODEL,
    start_sequence=_START_SEQUENCE,
    beam_size=2,
    mutations_per_sequence=1.0,
    n_rollouts_per_root=4,
    eval_batch_size=1,
    rng_seed=5,
)

_GRADABEAM_KWARGS = dict(
    model_fn=_MODEL,
    start_sequence=_START_SEQUENCE,
    beam_size=2,
    mutations_per_sequence=2.0,
    n_rollouts_per_root=4,
    exploration_alpha=0.5,
    use_pbt=True,
    max_rollout_len=200,
    eval_batch_size=1,
    rng_seed=5,
)

_KWARGS = {AdaBeam: _ADABEAM_KWARGS, GradaBeam: _GRADABEAM_KWARGS}


def benchmark_designer(designer_cls) -> None:
    designer = designer_cls(**_KWARGS[designer_cls])

    # Warm-up: prime Python interpreter, NumPy/PyTorch kernel caches
    designer.run(n_steps=N_WARMUP_STEPS)

    start_time = time.perf_counter()
    designer.run(n_steps=N_MEASURED_STEPS)
    elapsed_time = time.perf_counter() - start_time

    seconds_per_step = elapsed_time / N_MEASURED_STEPS
    print(
        f"{designer_cls.__name__}: {seconds_per_step:.4f} s/step "
        f"({N_MEASURED_STEPS} steps in {elapsed_time:.3f} s)"
    )


if __name__ == "__main__":
    print(f"Warm-up steps: {N_WARMUP_STEPS}  |  Measured steps: {N_MEASURED_STEPS}\n")
    for cls in [GradaBeam, AdaBeam]:
        benchmark_designer(cls)
