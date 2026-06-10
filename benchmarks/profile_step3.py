"""Step 3 profiler — localise the overhead within action_space_mutation.

Run ONLY on action_space_mutation with the dummy oracle so Python dominates:

    conda run -n gradabeam python -u benchmarks/profile_step3.py

Produces:
  1. cProfile tottime summary (top 30 functions).
  2. Per-call timing spy on:
       - build_uniform_pos_and_chars
       - generate_random_mutant_actionspace
       - ModelWrapper.get_fitness  (cache overhead)
  3. Call-count analysis: confirms calls/step to build_uniform_pos_and_chars.

Uses plain function-swapping (no wrapt dependency).
"""

import cProfile
import io
import pstats
import statistics
import sys
import time

sys.path.insert(0, ".")
sys.path.insert(0, "oracles")

import numpy as np

# ---------------------------------------------------------------------------
# Hparams — identical to bench_harness.py / perf_benchmark.py
# ---------------------------------------------------------------------------
L = 3000
_START_SEQUENCE = "A" * L
BEAM_SIZE = 2
N_ROLLOUTS = 4
# _attach_uniform_probs is called BEAM_SIZE * N_ROLLOUTS = 8 times per step

_ADABEAM_KWARGS = dict(
    start_sequence=_START_SEQUENCE,
    beam_size=BEAM_SIZE,
    mutations_per_sequence=1.0,
    n_rollouts_per_root=N_ROLLOUTS,
    skip_repeat_sequences=False,
    eval_batch_size=1,
    rng_seed=5,
)

N_WARMUP         = 20
N_PROFILE_STEPS  = 200


# ---------------------------------------------------------------------------
# Dummy oracle
# ---------------------------------------------------------------------------

class DummyModel:
    """Near-zero cost oracle: constant 0.0 fitness."""
    def __call__(self, seqs):
        return [0.0] * len(seqs)
    def eval(self):
        pass
    def parameters(self):
        return iter([])
    def get_tism(self, sequence, idxs=None):
        positions = list(range(len(sequence))) if idxs is None else list(idxs)
        pac = [(p, b) for p in positions for b in ["A","C","G","T"] if b != sequence[p]]
        return pac, np.zeros(len(pac), dtype=np.float32)


# ---------------------------------------------------------------------------
# Simple call/time spy (no external dependencies)
# ---------------------------------------------------------------------------

class Spy:
    def __init__(self, name):
        self.name = name
        self.calls = 0
        self.total_s = 0.0
        self._step_calls = []
        self._step_s = []
        self._cur_calls = 0
        self._cur_s = 0.0

    def flush_step(self):
        self._step_calls.append(self._cur_calls)
        self._step_s.append(self._cur_s)
        self._cur_calls = 0
        self._cur_s = 0.0

    def mean_calls_per_step(self):
        return statistics.mean(self._step_calls) if self._step_calls else 0

    def mean_ms_per_step(self):
        return statistics.mean(self._step_s) * 1000 if self._step_s else 0

    def stdev_ms_per_step(self):
        return statistics.stdev(self._step_s) * 1000 if len(self._step_s) > 1 else 0


def _make_timed(spy, fn):
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        dt = time.perf_counter() - t0
        spy.calls += 1
        spy.total_s += dt
        spy._cur_calls += 1
        spy._cur_s += dt
        return result
    wrapper.__name__ = fn.__name__
    wrapper.__qualname__ = fn.__qualname__
    return wrapper


def install_spies():
    """Patch three prime-suspect call sites and return the spy objects."""
    import gradabeam.ada_utils as ada_utils
    import gradabeam.adaptive_rollout as ar

    spy_bup  = Spy("build_uniform_pos_and_chars")
    spy_gma  = Spy("generate_random_mutant_actionspace")
    spy_gf   = Spy("ModelWrapper.get_fitness")

    # 1. build_uniform_pos_and_chars
    orig_bup = ada_utils.build_uniform_pos_and_chars
    ada_utils.build_uniform_pos_and_chars   = _make_timed(spy_bup, orig_bup)
    ar.ada_utils.build_uniform_pos_and_chars = ada_utils.build_uniform_pos_and_chars

    # 2. generate_random_mutant_actionspace
    orig_gma = ada_utils.generate_random_mutant_actionspace
    ada_utils.generate_random_mutant_actionspace   = _make_timed(spy_gma, orig_gma)
    ar.ada_utils.generate_random_mutant_actionspace = ada_utils.generate_random_mutant_actionspace

    # 3. ModelWrapper.get_fitness (captures cache-lookup overhead)
    from gradabeam.ada_utils import ModelWrapper
    orig_gf = ModelWrapper.get_fitness.__func__ if hasattr(ModelWrapper.get_fitness, '__func__') else ModelWrapper.get_fitness
    def patched_gf(self, m_input):
        t0 = time.perf_counter()
        result = orig_gf(self, m_input)
        dt = time.perf_counter() - t0
        spy_gf.calls += 1
        spy_gf.total_s += dt
        spy_gf._cur_calls += 1
        spy_gf._cur_s += dt
        return result
    ModelWrapper.get_fitness = patched_gf

    return spy_bup, spy_gma, spy_gf


# ---------------------------------------------------------------------------
# Step 3a — cProfile
# ---------------------------------------------------------------------------

def run_cprofile():
    from gradabeam import AdaBeam
    print("\n" + "=" * 72, flush=True)
    print("STEP 3a — cProfile (tottime + cumtime), 200 steps, dummy oracle", flush=True)
    print("=" * 72, flush=True)

    model = DummyModel()
    d = AdaBeam(model_fn=model, **_ADABEAM_KWARGS)
    d.run(n_steps=N_WARMUP)

    pr = cProfile.Profile()
    pr.enable()
    d.run(n_steps=N_PROFILE_STEPS)
    pr.disable()

    buf = io.StringIO()
    ps = pstats.Stats(pr, stream=buf).sort_stats("tottime")
    ps.print_stats(30)
    print("--- tottime top 30 ---", flush=True)
    print(buf.getvalue(), flush=True)

    buf2 = io.StringIO()
    ps2 = pstats.Stats(pr, stream=buf2).sort_stats("cumtime")
    ps2.print_stats(20)
    print("--- cumtime top 20 ---", flush=True)
    print(buf2.getvalue(), flush=True)


# ---------------------------------------------------------------------------
# Step 3b — spy timings
# ---------------------------------------------------------------------------

def run_spy_timings():
    from gradabeam import AdaBeam
    print("\n" + "=" * 72, flush=True)
    print("STEP 3b — Per-function spy timings (200 steps, dummy oracle)", flush=True)
    print("=" * 72, flush=True)

    spy_bup, spy_gma, spy_gf = install_spies()
    spies = [spy_bup, spy_gma, spy_gf]

    model = DummyModel()
    d = AdaBeam(model_fn=model, **_ADABEAM_KWARGS)
    d.run(n_steps=N_WARMUP)

    step_times = []
    for _ in range(N_PROFILE_STEPS):
        for s in spies:
            s._cur_calls = 0
            s._cur_s = 0.0
        t0 = time.perf_counter()
        d.run(n_steps=1)
        step_times.append(time.perf_counter() - t0)
        for s in spies:
            s.flush_step()

    total_mean_ms = statistics.mean(step_times) * 1000
    total_stdev_ms = statistics.stdev(step_times) * 1000

    print(f"\nTotal step time: {total_mean_ms:.3f} ± {total_stdev_ms:.3f} ms/step", flush=True)
    print(f"Expected calls/step to _attach_uniform_probs / build_uniform_pos_and_chars:", flush=True)
    print(f"  beam_size={BEAM_SIZE} × n_rollouts={N_ROLLOUTS} = {BEAM_SIZE * N_ROLLOUTS} calls/step\n", flush=True)

    header = f"  {'Function':<42} {'calls/step':>10} {'ms/step':>10} {'stdev':>8} {'% total':>8}"
    print(header, flush=True)
    print("  " + "-" * 82, flush=True)
    for spy in spies:
        cps  = spy.mean_calls_per_step()
        mps  = spy.mean_ms_per_step()
        std  = spy.stdev_ms_per_step()
        pct  = mps / total_mean_ms * 100 if total_mean_ms > 0 else 0
        print(f"  {spy.name:<42} {cps:>10.1f} {mps:>10.3f} {std:>8.3f} {pct:>7.1f}%", flush=True)

    print(flush=True)
    print("VERDICT:", flush=True)
    bup_calls = spy_bup.mean_calls_per_step()
    bup_ms    = spy_bup.mean_ms_per_step()
    gma_calls = spy_gma.mean_calls_per_step()
    gma_ms    = spy_gma.mean_ms_per_step()
    gf_ms     = spy_gf.mean_ms_per_step()
    print(f"  build_uniform_pos_and_chars  : {bup_ms:.3f} ms/step "
          f"({bup_calls:.0f} calls × {bup_ms/max(bup_calls,1)*1000:.1f} µs/call)", flush=True)
    print(f"  generate_random_mutant_actsp : {gma_ms:.3f} ms/step "
          f"({gma_calls:.0f} calls × {gma_ms/max(gma_calls,1)*1000:.1f} µs/call)", flush=True)
    print(f"  get_fitness (cache+oracle)   : {gf_ms:.3f} ms/step "
          f"({spy_gf.mean_calls_per_step():.0f} calls/step)", flush=True)
    print(f"  Unaccounted remainder        : "
          f"{total_mean_ms - bup_ms - gma_ms - gf_ms:.3f} ms/step", flush=True)

    return spy_bup, spy_gma, spy_gf, total_mean_ms


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
        pass

    run_cprofile()
    run_spy_timings()
    print("\nDone.", flush=True)
