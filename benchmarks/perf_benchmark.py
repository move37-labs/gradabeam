"""Performance benchmark for GradaBeam and AdaBeam designers.

Measures step throughput for regression tracking. Run directly with:

    python benchmarks/perf_benchmark.py

Results are printed to stdout; no pass/fail threshold is enforced so that
numbers can be compared across runs and machines without false CI failures.
"""

import perf_core

N_WARMUP_STEPS = 20
N_MEASURED_STEPS = 1000


def benchmark_designer(designer_name: str) -> None:
    # Run 1 repeat of 200 steps to perfectly match the original benchmark's size
    results = perf_core.measure(
        designer_name=designer_name,
        n_repeats=1,
        warmup_steps=N_WARMUP_STEPS,
        steps_per_repeat=N_MEASURED_STEPS,
    )

    elapsed_time = results["total_seconds"]
    seconds_per_step = results["median_s_per_step"]
    print(
        f"{results['designer']}: {seconds_per_step:.4f} s/step "
        f"({N_MEASURED_STEPS} steps in {elapsed_time:.3f} s)"
    )


if __name__ == "__main__":
    print(f"Warm-up steps: {N_WARMUP_STEPS}  |  Measured steps: {N_MEASURED_STEPS}\n")
    for designer in ["GradaBeam", "AdaBeam"]:
        benchmark_designer(designer)
