"""Generate golden trajectory fixtures for test_adabeam_equivalence.

Run from the repo root:
    python gradabeam/fixtures/generate_golden.py

MUST be run BEFORE any changes to adabeam_optimizer.py or gradabeam_optimizer.py.
The produced JSON files are static fixtures committed to the repo.  The equivalence
test compares against them, never re-runs AdaBeam live.
"""

import json
import os
import sys

import numpy as np

# Make sure the repo root is on the path so we can import gradabeam and oracles.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO_ROOT)

from gradabeam.adabeam_optimizer import AdaBeam  # noqa: E402
from gradabeam import testing_utils  # noqa: E402

# Oracles directory
sys.path.insert(0, os.path.join(_REPO_ROOT, "oracles"))
from substring_count import CountSubstringModel  # noqa: E402


# ---------------------------------------------------------------------------
# Capturing subclass — identical inner loop, captures ALL sorted proposals
# ---------------------------------------------------------------------------

class _CapturingAdaBeam(AdaBeam):
    """AdaBeam subclass that records every proposed sequence per step.

    The inner rollout loop is a verbatim copy of AdaBeam.propose_sequences so
    that the RNG consumption is IDENTICAL to the original.  The only addition
    is the ``self.trajectory.append(...)`` call after building sorted_sequences.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.trajectory: list[list[dict]] = []

    def propose_sequences(self, root_nodes):
        sequences, rollout_lengths = set(), []
        root_nodes_effective = root_nodes * self.n_rollouts_per_root
        for i in range(0, len(root_nodes_effective), self.eval_batch_size):
            cur_root_nodes = root_nodes_effective[i : i + self.eval_batch_size]
            parent_nodes = cur_root_nodes

            cur_rollout_length = 0
            while (
                len(parent_nodes) > 0
                and cur_rollout_length < self.max_rollout_len
            ):
                num_edit_locs = self.num_mutations_sampler.sample(len(parent_nodes))
                children = self.mutate_nodes(parent_nodes, num_edit_locs)
                sequences.update(children)

                cur_rollout_length += 1
                new_nodes = []
                for child, comparison_node in zip(children, parent_nodes):
                    if child.fitness >= comparison_node.fitness:
                        new_nodes.append(child)
                    else:
                        rollout_lengths.append(cur_rollout_length)
                parent_nodes = new_nodes

        if len(sequences) == 0:
            raise ValueError("No sequences generated.")

        sorted_sequences = sorted(
            sequences, key=lambda x: (x.fitness, x.seq), reverse=True
        )

        # ── capture point ──────────────────────────────────────────────────
        self.trajectory.append(
            [{"seq": n.seq, "fitness": float(n.fitness)} for n in sorted_sequences]
        )
        # ───────────────────────────────────────────────────────────────────

        return sorted_sequences[: self.beam_size]


# ---------------------------------------------------------------------------
# Verification helper — compare capturing subclass against plain AdaBeam
# ---------------------------------------------------------------------------

def _verify_identical_top_beam(oracle_name, model_fn, start_seq, n_steps, rng_seed):
    """Assert that _CapturingAdaBeam top-beam equals plain AdaBeam top-beam."""
    kwargs = AdaBeam.debug_init_args()
    kwargs.update(model_fn=model_fn, start_sequence=start_seq, rng_seed=rng_seed)

    plain = AdaBeam(**kwargs)
    capturing = _CapturingAdaBeam(**{**kwargs})

    for step in range(n_steps):
        plain.run(n_steps=1)
        capturing.run(n_steps=1)

        plain_beam = sorted((n.seq, float(n.fitness)) for n in plain.current_nodes)
        cap_beam = sorted(
            (n.seq, float(n.fitness)) for n in capturing.current_nodes
        )
        assert plain_beam == cap_beam, (
            f"{oracle_name} step {step}: top-beam mismatch!\n"
            f"  plain:     {plain_beam[:3]}\n"
            f"  capturing: {cap_beam[:3]}"
        )
    print(f"  {oracle_name}: capturing subclass verified identical to plain AdaBeam ✓")


# ---------------------------------------------------------------------------
# Fixture generation
# ---------------------------------------------------------------------------

def generate_fixture(oracle_name, model_fn, start_seq, n_steps=3, rng_seed=42):
    kwargs = AdaBeam.debug_init_args()
    kwargs.update(model_fn=model_fn, start_sequence=start_seq, rng_seed=rng_seed)

    opt = _CapturingAdaBeam(**kwargs)

    initial_beam = [
        {"seq": n.seq, "fitness": float(n.fitness)} for n in opt.current_nodes
    ]

    opt.run(n_steps=n_steps)

    return {
        "oracle": oracle_name,
        "rng_seed": rng_seed,
        "n_steps": n_steps,
        "beam_size": kwargs["beam_size"],
        "mutations_per_sequence": kwargs["mutations_per_sequence"],
        "n_rollouts_per_root": kwargs["n_rollouts_per_root"],
        "start_sequence": start_seq,
        "initial_beam": initial_beam,
        "steps": opt.trajectory,
    }


if __name__ == "__main__":
    out_dir = os.path.dirname(os.path.abspath(__file__))
    N_STEPS = 3
    RNG_SEED = 42
    START_SEQ = "AAAAAA"

    print("Verifying capturing subclass reproduces plain AdaBeam exactly …")

    models = {
        "CountLetterModel": testing_utils.CountLetterModel(),
        "CountSubstringModel": CountSubstringModel(substring="AC"),
    }

    for name, model in models.items():
        _verify_identical_top_beam(name, model, START_SEQ, N_STEPS, RNG_SEED)

    print("\nGenerating fixtures …")

    for oracle_name, model_fn in models.items():
        slug = "count_letter" if "Letter" in oracle_name else "substring_count"
        data = generate_fixture(oracle_name, model_fn, START_SEQ, N_STEPS, RNG_SEED)
        path = os.path.join(out_dir, f"adabeam_golden_{slug}.json")
        with open(path, "w") as fh:
            json.dump(data, fh, indent=2)
        n_total = sum(len(s) for s in data["steps"])
        print(
            f"  {oracle_name}: {N_STEPS} steps, "
            f"{n_total} total proposals → {path}"
        )

    print("\nDone.  Commit the JSON files before editing any source.")
