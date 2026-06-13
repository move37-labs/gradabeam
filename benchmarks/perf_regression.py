"""Performance regression testing driver for GradaBeam and AdaBeam.

Compares step throughput on the current HEAD vs a baseline Git reference
on the same machine/runner. Automatically exits with 1 if any performance
regression exceeding the specified threshold is detected, making it ideal
for CI and git bisect.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile


def run_cmd(
    cmd: str, cwd: str | None = None, check: bool = True
) -> subprocess.CompletedProcess:
    """Helper to run a shell command and capture its output."""
    res = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=cwd)
    if check and res.returncode != 0:
        raise RuntimeError(
            f"Command failed: {cmd}\n"
            f"Exit code: {res.returncode}\n"
            f"Stdout: {res.stdout.strip()}\n"
            f"Stderr: {res.stderr.strip()}"
        )
    return res


def get_git_revision(ref: str) -> str:
    """Resolve a git ref to its full SHA."""
    res = run_cmd(f"git rev-parse {ref}")
    return res.stdout.strip()


def get_merge_base(base_branch: str = "main") -> str:
    """Find the merge-base between HEAD and base_branch."""
    # Try origin/base_branch first, then local base_branch
    for branch in [f"origin/{base_branch}", base_branch]:
        try:
            res = run_cmd(f"git merge-base HEAD {branch}", check=False)
            if res.returncode == 0:
                ref = res.stdout.strip()
                if ref:
                    return ref
        except Exception:
            continue

    # Fallback to HEAD~1
    print(
        f"Warning: Could not find merge-base with {base_branch}. "
        "Falling back to HEAD~1 as baseline ref.",
        file=sys.stderr,
    )
    res = run_cmd("git rev-parse HEAD~1")
    return res.stdout.strip()


def run_measurement(
    perf_core_path: str,
    designer: str,
    cwd: str,
    n_repeats: int,
    warmup_steps: int,
    steps_per_repeat: int,
) -> dict:
    """Runs perf_core.py as a subprocess with a specific cwd and returns the parsed JSON."""
    cmd = (
        f"{sys.executable} {perf_core_path} "
        f"--designer {designer} "
        f"--n-repeats {n_repeats} "
        f"--warmup-steps {warmup_steps} "
        f"--steps-per-repeat {steps_per_repeat} "
        f"--source-dir {cwd}"
    )
    res = run_cmd(cmd)

    # Extract the JSON line from stdout, ignoring other print statements
    json_data = None
    for line in res.stdout.splitlines():
        line_str = line.strip()
        if line_str.startswith('{"') and line_str.endswith("}"):
            try:
                json_data = json.loads(line_str)
                break
            except json.JSONDecodeError:
                continue

    if json_data is not None:
        return json_data

    raise RuntimeError(
        f"Failed to find or decode JSON from measurement run.\n"
        f"Command: {cmd}\n"
        f"Output: {res.stdout}\n"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run performance regression tests against a baseline Git ref."
    )
    parser.add_argument(
        "--designer",
        choices=["gradabeam", "adabeam", "both"],
        default="both",
        help="Designer to test. Use 'both' to run both GradaBeam and AdaBeam.",
    )
    parser.add_argument(
        "--baseline-ref",
        default=None,
        help="Git ref to use as baseline. If not provided, resolves via merge-base.",
    )
    parser.add_argument(
        "--base-branch",
        default="main",
        help="Base branch to calculate merge-base against.",
    )
    parser.add_argument(
        "--max-slowdown",
        type=float,
        default=1.20,
        help="Maximum allowed performance ratio (Head s/step / Baseline s/step). Default: 1.20.",
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=5,
        help="Number of measured repeats to run per designer.",
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
        "--json-out",
        default=None,
        help="Path to write the detailed JSON comparison results.",
    )

    args = parser.parse_args()

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    perf_core_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "perf_core.py")
    )

    # Determine baseline ref
    if args.baseline_ref:
        baseline_ref = get_git_revision(args.baseline_ref)
    else:
        baseline_ref = get_merge_base(args.base_branch)

    head_ref = get_git_revision("HEAD")

    print("============================================================")
    print("Performance Regression Test Setup")
    print("============================================================")
    print(f"Repo Root:     {repo_root}")
    print(f"Baseline Ref:  {baseline_ref}")
    print(f"Head Ref:      {head_ref}")
    print(f"Tolerance:     {args.max_slowdown}x max slowdown limit")
    print(
        f"Parameters:    {args.n_repeats} repeats x {args.steps_per_repeat} steps "
        f"(warmup: {args.warmup_steps})"
    )
    print("============================================================\n")

    if baseline_ref == head_ref:
        print(
            "Warning: Baseline ref is identical to HEAD ref. Self-comparing HEAD.",
            file=sys.stderr,
        )

    # List of designers to run
    designers = ["gradabeam", "adabeam"] if args.designer == "both" else [args.designer]

    # Create temporary directory for git worktree
    temp_dir = tempfile.mkdtemp(prefix="gradabeam_perf_baseline_")
    worktree_added = False

    try:
        print(f"Creating baseline worktree at {temp_dir}...")
        run_cmd(f"git worktree add --detach {temp_dir} {baseline_ref}")
        worktree_added = True

        results = {}
        any_regression = False

        for ds in designers:
            print(f"\n--- Measuring {ds.upper()} on Baseline ({baseline_ref[:7]}) ---")
            base_res = run_measurement(
                perf_core_path=perf_core_path,
                designer=ds,
                cwd=temp_dir,
                n_repeats=args.n_repeats,
                warmup_steps=args.warmup_steps,
                steps_per_repeat=args.steps_per_repeat,
            )

            print(f"--- Measuring {ds.upper()} on Head ({head_ref[:7]}) ---")
            head_res = run_measurement(
                perf_core_path=perf_core_path,
                designer=ds,
                cwd=repo_root,
                n_repeats=args.n_repeats,
                warmup_steps=args.warmup_steps,
                steps_per_repeat=args.steps_per_repeat,
            )

            base_med = base_res["median_s_per_step"]
            head_med = head_res["median_s_per_step"]
            ratio = head_med / base_med if base_med > 0 else 1.0
            verdict = "FAIL" if ratio > args.max_slowdown else "PASS"

            if verdict == "FAIL":
                any_regression = True

            results[ds] = {
                "baseline": base_res,
                "head": head_res,
                "ratio": ratio,
                "verdict": verdict,
            }

        # Print detailed report
        header = f"{'Designer':<12} | {'Baseline (s/step)':<18} | {'Head (s/step)':<14} | {'Ratio':<8} | {'Verdict':<7}"
        sep = "-" * len(header)
        print("\n" + "=" * len(header))
        print("PERFORMANCE REGRESSION REPORT")
        print("=" * len(header))
        print(header)
        print(sep)
        for ds, data in results.items():
            base_med = data["baseline"]["median_s_per_step"]
            head_med = data["head"]["median_s_per_step"]
            ratio = data["ratio"]
            verdict = data["verdict"]
            ratio_str = f"{ratio:.2f}x"
            print(
                f"{ds.upper():<12} | {base_med:<18.4f} | {head_med:<14.4f} | {ratio_str:<8} | {verdict:<7}"
            )
        print("=" * len(header) + "\n")

        # Write JSON output if requested
        if args.json_out:
            out_data = {
                "baseline_ref": baseline_ref,
                "head_ref": head_ref,
                "max_slowdown": args.max_slowdown,
                "results": results,
            }
            with open(args.json_out, "w") as f:
                json.dump(out_data, f, indent=2)
            print(f"Wrote detailed JSON results to {args.json_out}")

        if any_regression:
            print("ERROR: Performance regression detected!")
            sys.exit(1)
        else:
            print("SUCCESS: Performance is within acceptable limits.")
            sys.exit(0)

    finally:
        # Cleanup worktree
        if worktree_added:
            print("\nCleaning up baseline worktree...")
            # We run without check=False because we want to guarantee we remove the dir
            run_cmd(f"git worktree remove --force {temp_dir}", check=False)
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
