"""Benchmark harness for Steps 1 and 2 of the overhead localisation task.

Run this SAME file on both branches (block-timed for Step 1, per-step for Step 2):

    git checkout main            && conda run -n gradabeam python -u benchmarks/bench_harness.py
    git checkout action_space_mutation && conda run -n gradabeam python -u benchmarks/bench_harness.py

Step 1: real BPNet/GATA2, 3 timed blocks of 200 steps each (20-step warm-up),
        AdaBeam and GrAdaBeam.  All 3 blocks run on the SAME designer (steady-state).
Step 2: dummy zero-cost oracle, AdaBeam, 3 runs × 200 steps, per-step timing.
        Exposes Python overhead isolated from model cost.
        Reports model.n_forward per step to check oracle-call count.
"""

import sys
import statistics
import time

sys.path.insert(0, ".")
sys.path.insert(0, "oracles")

import numpy as np

from gradabeam import AdaBeam, GradaBeam

# ---------------------------------------------------------------------------
# Hparams — GATA2 benchmark (same as perf_benchmark.py)
# ---------------------------------------------------------------------------
L = 3000
_START_SEQUENCE = "A" * L

_ADABEAM_KWARGS = dict(
    model_fn=None,
    start_sequence=_START_SEQUENCE,
    beam_size=2,
    mutations_per_sequence=1.0,
    n_rollouts_per_root=4,
    skip_repeat_sequences=False,
    eval_batch_size=1,
    rng_seed=5,
)

_GRADABEAM_KWARGS = dict(
    model_fn=None,
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

N_WARMUP  = 20
N_BLOCK   = 200   # steps per timed block
N_BLOCKS  = 3     # repeated timed blocks on same designer (= run-to-run spread)

# ---------------------------------------------------------------------------
# Dummy oracle
# ---------------------------------------------------------------------------

class DummyModel:
    """Near-zero cost oracle: constant fitness, no I/O, no torch."""
    def __call__(self, seqs):
        return [0.0] * len(seqs)
    def eval(self):
        pass
    def parameters(self):
        return iter([])
    # GrAdaBeam needs tism_torch attr check + get_tism
    def get_tism(self, sequence, idxs=None):
        positions = list(range(len(sequence))) if idxs is None else list(idxs)
        pac = [(p, b) for p in positions for b in ["A","C","G","T"] if b != sequence[p]]
        return pac, np.zeros(len(pac), dtype=np.float32)
    def tism_torch(self, *a, **kw):
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Step 1 — real BPNet/GATA2
# ---------------------------------------------------------------------------

def step1_real_bpnet():
    print("\n" + "=" * 70, flush=True)
    print("STEP 1 — Real BPNet/GATA2, both algorithms", flush=True)
    print(f"  {N_WARMUP} warm-up steps, then {N_BLOCKS}x{N_BLOCK}-step timed blocks", flush=True)
    print("=" * 70, flush=True)
    try:
        from bpnet import BPNet
        model = BPNet(protein="GATA2")
        print("BPNet loaded.", flush=True)
    except Exception as e:
        print(f"[SKIP] BPNet not available: {e}", flush=True)
        return

    results = {}
    for cls, kw_tmpl, label in [
        (AdaBeam,   _ADABEAM_KWARGS,   "AdaBeam  "),
        (GradaBeam, _GRADABEAM_KWARGS, "GrAdaBeam"),
    ]:
        kw = dict(kw_tmpl); kw["model_fn"] = model
        print(f"\n{label}: constructing + {N_WARMUP}-step warm-up ...", flush=True)
        d = cls(**kw)
        d.run(n_steps=N_WARMUP)
        print(f"{label}: warm-up done.  Running {N_BLOCKS} blocks of {N_BLOCK} steps ...", flush=True)

        block_means = []
        for b in range(N_BLOCKS):
            t0 = time.perf_counter()
            d.run(n_steps=N_BLOCK)
            elapsed = time.perf_counter() - t0
            mean_s = elapsed / N_BLOCK
            block_means.append(mean_s)
            print(f"  block {b+1}: {mean_s:.4f} s/step  ({elapsed:.2f} s total)", flush=True)

        grand = statistics.mean(block_means)
        spread = max(block_means) - min(block_means)
        stdev  = statistics.stdev(block_means) if N_BLOCKS > 1 else 0.0
        print(f"  --> grand mean: {grand:.4f} s/step  block-spread: {spread:.4f}  stdev: {stdev:.4f}", flush=True)
        results[label.strip()] = {"mean": grand, "spread": spread, "stdev": stdev, "blocks": block_means}

    print("\nSUMMARY TABLE (Step 1):", flush=True)
    print(f"  {'Algorithm':<12} {'mean s/step':>12} {'stdev':>8} {'block-spread':>13}", flush=True)
    print("  " + "-" * 48, flush=True)
    for alg, r in results.items():
        print(f"  {alg:<12} {r['mean']:>12.4f} {r['stdev']:>8.4f} {r['spread']:>13.4f}", flush=True)
    return results


# ---------------------------------------------------------------------------
# Step 2 — dummy oracle, AdaBeam, Python overhead
# ---------------------------------------------------------------------------

def step2_dummy_oracle():
    print("\n" + "=" * 70, flush=True)
    print("STEP 2 — Dummy oracle, Python overhead isolation", flush=True)
    print("=" * 70, flush=True)

    model = DummyModel()

    # AdaBeam
    print("\nAdaBeam (dummy oracle):", flush=True)
    ada_results = _run_dummy(AdaBeam, _ADABEAM_KWARGS, model, "AdaBeam")

    # GrAdaBeam (control — confirm not slower on action_space_mutation)
    print("\nGrAdaBeam (dummy oracle, CONTROL):", flush=True)
    _run_dummy(GradaBeam, _GRADABEAM_KWARGS, model, "GrAdaBeam")

    return ada_results


def _run_dummy(cls, kw_tmpl, model, label):
    kw = dict(kw_tmpl); kw["model_fn"] = model
    d = cls(**kw)
    d.run(n_steps=N_WARMUP)

    n_fwd_before = getattr(d.model, "n_forward", 0) or 0
    n_bwd_before = getattr(d.model, "n_backward", 0) or 0

    block_means = []
    for b in range(N_BLOCKS):
        t0 = time.perf_counter()
        d.run(n_steps=N_BLOCK)
        elapsed = time.perf_counter() - t0
        mean_s = elapsed / N_BLOCK
        block_means.append(mean_s)
        print(f"  block {b+1}: {mean_s*1000:.4f} ms/step  ({elapsed:.3f} s total)", flush=True)

    n_fwd_after  = getattr(d.model, "n_forward",  None)
    n_bwd_after  = getattr(d.model, "n_backward", None)
    grand = statistics.mean(block_means)
    spread = max(block_means) - min(block_means)

    print(f"  --> grand mean: {grand*1000:.4f} ms/step  spread: {spread*1000:.4f} ms", flush=True)

    if n_fwd_after is not None:
        total_steps = N_WARMUP + N_BLOCKS * N_BLOCK
        fwd_delta = n_fwd_after  - n_fwd_before
        bwd_delta = (n_bwd_after or 0) - (n_bwd_before or 0)
        print(f"  n_forward/step : {fwd_delta/total_steps:.3f}  (total {fwd_delta})", flush=True)
        print(f"  n_backward/step: {bwd_delta/total_steps:.3f}  (total {bwd_delta})", flush=True)
    else:
        print(f"  (n_forward not available on this branch)", flush=True)

    return {"mean_ms": grand * 1000, "spread_ms": spread * 1000, "blocks": block_means}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import subprocess
    try:
        branch = subprocess.check_output(["git","rev-parse","--abbrev-ref","HEAD"],text=True).strip()
        commit = subprocess.check_output(["git","rev-parse","--short","HEAD"],text=True).strip()
        print(f"Branch: {branch}  commit: {commit}", flush=True)
    except Exception:
        print("(could not read git branch)", flush=True)

    step1_real_bpnet()
    step2_dummy_oracle()
    print("\nDone.", flush=True)
