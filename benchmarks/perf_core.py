"""Core timing logic for performance profiling of GradaBeam and AdaBeam.

This file provides a reusable measurement function and a command-line interface
that outputs timing results as a single line of JSON. It can be dynamically
pointed to a specific source directory to benchmark different git refs.
"""

import sys
import os

import argparse
import json
import time

from gradabeam import GradaBeam, AdaBeam
from bpnet import BPNet

# Inspect sys.argv for --source-dir before any other imports
source_dir = None
for i, arg in enumerate(sys.argv):
    if arg == "--source-dir" and i + 1 < len(sys.argv):
        source_dir = sys.argv[i + 1]
        break

if source_dir:
    src_dir = os.path.abspath(source_dir)
    sys.path.insert(0, src_dir)
    sys.path.insert(1, os.path.join(src_dir, "oracles"))
else:
    # Use workspace root as default
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.insert(0, repo_root)
    sys.path.insert(1, os.path.join(repo_root, "oracles"))


def get_median(lst: list[float]) -> float:
    """Calculate the median of a list of floats in pure Python."""
    if not lst:
        return 0.0
    s = sorted(lst)
    n = len(s)
    if n % 2 == 1:
        return s[n // 2]
    else:
        return (s[n // 2 - 1] + s[n // 2]) / 2.0


def measure(
    designer_name: str,
    protein: str = "GATA2",
    n_repeats: int = 5,
    warmup_steps: int = 20,
    steps_per_repeat: int = 200,
) -> dict:
    """Measures step throughput for a designer over N repeats.

    Runs warmup_steps first, then executes n_repeats runs of steps_per_repeat steps.
    """
    start_seq = "A" * 3000
    model = BPNet(protein=protein)

    adabeam_kwargs = dict(
        model_fn=model,
        start_sequence=start_seq,
        beam_size=2,
        mutations_per_sequence=1.0,
        n_rollouts_per_root=4,
        skip_repeat_sequences=False,
        eval_batch_size=1,
        rng_seed=5,
    )

    gradabeam_kwargs = dict(
        model_fn=model,
        start_sequence=start_seq,
        beam_size=2,
        mutations_per_sequence=2.0,
        n_rollouts_per_root=4,
        exploration_alpha=0.5,
        use_pbt=True,
        max_rollout_len=200,
        eval_batch_size=1,
        rng_seed=5,
    )

    if designer_name.lower() == "gradabeam":
        designer_cls = GradaBeam
        kwargs = gradabeam_kwargs
    elif designer_name.lower() == "adabeam":
        designer_cls = AdaBeam
        kwargs = adabeam_kwargs
    else:
        raise ValueError(f"Unknown designer: {designer_name}")

    designer = designer_cls(**kwargs)

    # Warm-up: prime interpreter, caches, etc.
    if warmup_steps > 0:
        designer.run(n_steps=warmup_steps)

    per_repeat_s_per_step = []
    total_start_time = time.perf_counter()

    for i in range(n_repeats):
        rep_start = time.perf_counter()
        designer.run(n_steps=steps_per_repeat)
        rep_elapsed = time.perf_counter() - rep_start
        s_per_step = rep_elapsed / steps_per_repeat
        per_repeat_s_per_step.append(s_per_step)

    total_elapsed = time.perf_counter() - total_start_time

    return {
        "designer": designer_cls.__name__,
        "per_repeat_s_per_step": per_repeat_s_per_step,
        "median_s_per_step": get_median(per_repeat_s_per_step),
        "min_s_per_step": min(per_repeat_s_per_step) if per_repeat_s_per_step else 0.0,
        "n_repeats": n_repeats,
        "steps_per_repeat": steps_per_repeat,
        "total_seconds": total_elapsed,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Profile GradaBeam / AdaBeam step throughput."
    )
    parser.add_argument(
        "--designer",
        choices=["gradabeam", "adabeam"],
        required=True,
        help="Optimizer to measure.",
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=5,
        help="Number of measured repeats.",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=20,
        help="Warmup steps to run before measurement.",
    )
    parser.add_argument(
        "--steps-per-repeat",
        type=int,
        default=200,
        help="Number of steps in each repeat.",
    )
    parser.add_argument(
        "--source-dir",
        type=str,
        default=None,
        help="Optional source directory to run against.",
    )
    parser.add_argument(
        "--protein",
        type=str,
        default="GATA2",
        help="Protein to use for BPNet.",
    )

    args = parser.parse_args()

    results = measure(
        designer_name=args.designer,
        protein=args.protein,
        n_repeats=args.n_repeats,
        warmup_steps=args.warmup_steps,
        steps_per_repeat=args.steps_per_repeat,
    )

    print(json.dumps(results))


if __name__ == "__main__":
    main()
