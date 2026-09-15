"""Local, reproducible audit harness for AdaBeam and GradaBeam.

The script deliberately has no cloud, network, or repository side effects. It
emits one JSON document suitable for comparing a checkout with a detached
baseline worktree:

    uv run python benchmarks/overnight_audit.py --output results.json
    uv run python benchmarks/overnight_audit.py --source-dir /tmp/baseline \
        --output baseline.json
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _source_dir_from_argv() -> str | None:
    for index, argument in enumerate(sys.argv):
        if argument == "--source-dir" and index + 1 < len(sys.argv):
            return sys.argv[index + 1]
    return None


SOURCE_DIR = _source_dir_from_argv()
if SOURCE_DIR:
    sys.path.insert(0, os.path.abspath(SOURCE_DIR))

from gradabeam import AdaBeam, GradaBeam, constants, seq_utils, tism
from gradabeam.testing_utils import CountLetterModel


class InstrumentedCountLetter(CountLetterModel):
    """Count-letter oracle that exposes actual string and TISM calls."""

    def __init__(self) -> None:
        super().__init__()
        self.string_evaluations = 0
        self.tism_calls = 0

    def __call__(self, sequences: list[str]) -> list[float]:
        self.string_evaluations += len(sequences)
        return super().__call__(sequences)

    def get_tism(
        self, sequence: str, idxs: list[int] | None = None
    ) -> tuple[list[tuple[int, str]], np.ndarray]:
        self.tism_calls += 1
        return super().get_tism(sequence, idxs)


class InstrumentedSubstring(tism.TISMModelClass, torch.nn.Module):
    """Small differentiable substring oracle independent of CLI loading."""

    def __init__(self, substring: str = "ATGTC") -> None:
        super().__init__()
        self.substring = substring
        self.vocab = constants.VOCAB
        self.vocab_to_idx = {nt: index for index, nt in enumerate(self.vocab)}
        self.vocab_array = np.array(self.vocab)
        self.target = seq_utils.dna2tensor(substring, vocab_list=self.vocab).unsqueeze(
            0
        )
        self.string_evaluations = 0
        self.tism_calls = 0

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        matches = torch.nn.functional.conv1d(values, self.target)
        return -torch.square(matches.squeeze(1)).sum(dim=1)

    def inference_on_tensor(self, values: torch.Tensor) -> torch.Tensor:
        return self.forward(values)

    def __call__(self, sequences: list[str]) -> list[float]:
        self.string_evaluations += len(sequences)
        values = seq_utils.dna2tensor_batch(sequences, vocab_list=self.vocab)
        return [float(value) for value in self.inference_on_tensor(values)]

    def get_tism(
        self, sequence: str, idxs: list[int] | None = None
    ) -> tuple[list[tuple[int, str]], np.ndarray]:
        self.tism_calls += 1
        return super().get_tism(sequence, idxs)


class InstrumentedAdditive(tism.TISMModelClass, torch.nn.Module):
    """A seeded positional additive landscape with an exact finite upper bound."""

    def __init__(self, length: int, seed: int) -> None:
        super().__init__()
        self.vocab = constants.VOCAB
        self.vocab_to_idx = {nt: index for index, nt in enumerate(self.vocab)}
        self.vocab_array = np.array(self.vocab)
        generator = np.random.default_rng(seed)
        self.weights = torch.tensor(generator.random((4, length)), dtype=torch.float32)
        self.string_evaluations = 0
        self.tism_calls = 0

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return -(values * self.weights).sum(dim=(1, 2))

    def inference_on_tensor(self, values: torch.Tensor) -> torch.Tensor:
        return self.forward(values)

    def __call__(self, sequences: list[str]) -> list[float]:
        self.string_evaluations += len(sequences)
        values = seq_utils.dna2tensor_batch(sequences, vocab_list=self.vocab)
        return [float(value) for value in self.inference_on_tensor(values)]

    def get_tism(
        self, sequence: str, idxs: list[int] | None = None
    ) -> tuple[list[tuple[int, str]], np.ndarray]:
        self.tism_calls += 1
        return super().get_tism(sequence, idxs)


class InstrumentedNK(tism.TISMModelClass, torch.nn.Module):
    """A seeded differentiable NK landscape with bounded fitness in [0, length]."""

    def __init__(self, length: int, k: int, seed: int) -> None:
        super().__init__()
        self.vocab = constants.VOCAB
        self.vocab_to_idx = {nt: index for index, nt in enumerate(self.vocab)}
        self.vocab_array = np.array(self.vocab)
        generator = np.random.default_rng(seed)
        self.neighborhoods = [
            [(position + offset) % length for offset in range(k + 1)]
            for position in range(length)
        ]
        self.tables = [
            torch.tensor(generator.random(4 ** (k + 1)), dtype=torch.float32)
            for _ in range(length)
        ]
        self.string_evaluations = 0
        self.tism_calls = 0

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        score = torch.zeros(values.shape[0], dtype=values.dtype)
        for neighborhood, table in zip(self.neighborhoods, self.tables):
            weighted = table.reshape((1,) + (4,) * len(neighborhood)).expand(
                values.shape[0], *([-1] * len(neighborhood))
            )
            for position in reversed(neighborhood):
                factor = values[:, :, position].reshape(
                    (values.shape[0],) + (1,) * (weighted.ndim - 2) + (4,)
                )
                weighted = (weighted * factor).sum(dim=-1)
            score = score + weighted
        return -score

    def inference_on_tensor(self, values: torch.Tensor) -> torch.Tensor:
        return self.forward(values)

    def __call__(self, sequences: list[str]) -> list[float]:
        self.string_evaluations += len(sequences)
        values = seq_utils.dna2tensor_batch(sequences, vocab_list=self.vocab)
        return [float(value) for value in self.inference_on_tensor(values)]

    def get_tism(
        self, sequence: str, idxs: list[int] | None = None
    ) -> tuple[list[tuple[int, str]], np.ndarray]:
        self.tism_calls += 1
        return super().get_tism(sequence, idxs)


def make_task(name: str, seed: int, sequence_length: int) -> tuple[Any, str, float]:
    if name == "count_letter":
        return InstrumentedCountLetter(), "A" * sequence_length, float(sequence_length)
    if name == "additive":
        model = InstrumentedAdditive(length=sequence_length, seed=seed)
        return (
            model,
            "A" * sequence_length,
            float(model.weights.max(dim=0).values.sum()),
        )
    if name == "substring":
        return (
            InstrumentedSubstring(),
            "A" * sequence_length,
            float(sequence_length - 4) * 25.0,
        )
    if name == "nk":
        return (
            InstrumentedNK(length=sequence_length, k=3, seed=seed),
            "A" * sequence_length,
            float(sequence_length),
        )
    raise ValueError(f"Unknown task: {name}")


def _install_edit_meter(designer: AdaBeam | GradaBeam) -> dict[str, Any]:
    meter: dict[str, Any] = {"requested": [], "actual": [], "evaluated": {}}

    def record(children: list[Any]) -> None:
        for child in children:
            previous = meter["evaluated"].get(child.seq)
            if previous is None or child.fitness > previous.fitness:
                meter["evaluated"][child.seq] = child

    if isinstance(designer, GradaBeam):
        original = designer.mutate_nodes_gradabeam

        def wrapped(nodes, counts, rates):
            children = original(nodes, counts, rates)
            meter["requested"].extend(int(value) for value in counts)
            meter["actual"].extend(
                sum(left != right for left, right in zip(node.seq, child.seq))
                for node, child in zip(nodes, children)
            )
            record(children)
            return children

        designer.mutate_nodes_gradabeam = wrapped
    else:
        original = designer.mutate_nodes

        def wrapped(nodes, counts, max_num_tries=300):
            children = original(nodes, counts, max_num_tries=max_num_tries)
            meter["requested"].extend(int(value) for value in counts)
            meter["actual"].extend(
                sum(left != right for left, right in zip(node.seq, child.seq))
                for node, child in zip(nodes, children)
            )
            record(children)
            return children

        designer.mutate_nodes = wrapped
    return meter


def run_trial(
    task: str,
    optimizer: str,
    seed: int,
    steps: int,
    landscape_seed: int | None = None,
    sequence_length: int = 30,
    use_pbt: bool = True,
    oracle_cost_budget: float | None = None,
    repeat: int = 0,
) -> dict[str, Any]:
    model, start_sequence, upper_bound = make_task(
        task, seed if landscape_seed is None else landscape_seed, sequence_length
    )
    start_fitness = -model([start_sequence])[0]
    kwargs: dict[str, Any] = {
        "model_fn": model,
        "start_sequence": start_sequence,
        "mutations_per_sequence": 1.0,
        "beam_size": 4,
        "n_rollouts_per_root": 4,
        "eval_batch_size": 1,
        "rng_seed": seed,
    }
    if optimizer == "gradabeam":
        designer: AdaBeam | GradaBeam = GradaBeam(
            **kwargs, exploration_alpha=0.05, use_pbt=use_pbt, max_rollout_len=30
        )
    elif optimizer == "adabeam":
        designer = AdaBeam(**kwargs, skip_repeat_sequences=False, max_rollout_len=30)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer}")

    meter = _install_edit_meter(designer)
    for node in designer.current_nodes:
        meter["evaluated"][node.seq] = node
    initial_oracle_cost = designer.model.cost
    trajectories: list[dict[str, Any]] = []
    started = time.perf_counter()
    step = 0
    while step < steps and (
        oracle_cost_budget is None
        or designer.model.cost - initial_oracle_cost < oracle_cost_budget
    ):
        designer.run(1)
        current_fitness = max(node.fitness for node in designer.current_nodes)
        archive = getattr(designer, "best_nodes", meter["evaluated"])
        best_fitness = max(node.fitness for node in archive.values())
        row: dict[str, Any] = {
            "step": step + 1,
            "current_beam_fitness": current_fitness,
            "best_ever_fitness": best_fitness,
        }
        if isinstance(designer, GradaBeam):
            row["mutation_rates"] = [
                node.mutations_per_sequence for node in designer.current_nodes
            ]
            row["exploration_alphas"] = [
                node.exploration_alpha for node in designer.current_nodes
            ]
        trajectories.append(row)
        step += 1
    wall_seconds = time.perf_counter() - started
    archive = getattr(designer, "best_nodes", meter["evaluated"])
    best_fitness = max(node.fitness for node in archive.values())
    return {
        "task": task,
        "optimizer": optimizer,
        "seed": seed,
        "repeat": repeat,
        "landscape_seed": landscape_seed,
        "steps": step,
        "requested_max_steps": steps,
        "sequence_length": sequence_length,
        "use_pbt": use_pbt if optimizer == "gradabeam" else None,
        "start_fitness": start_fitness,
        "current_beam_fitness": max(node.fitness for node in designer.current_nodes),
        "best_ever_fitness": best_fitness,
        "normalized_gain": (best_fitness - start_fitness)
        / max(upper_bound - start_fitness, 1e-12),
        "forward_sequence_evaluations": model.string_evaluations,
        "tism_calls": model.tism_calls,
        "initial_oracle_cost": initial_oracle_cost,
        "oracle_cost": designer.model.cost,
        "proposal_oracle_cost": designer.model.cost - initial_oracle_cost,
        "oracle_cost_budget": oracle_cost_budget,
        "unique_evaluations": len(designer.model.cache),
        "requested_hamming_edits": meter["requested"],
        "actual_hamming_edits": meter["actual"],
        "wall_seconds": wall_seconds,
        "trajectory": trajectories,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["count_letter", "additive", "substring", "nk"],
    )
    parser.add_argument("--optimizers", nargs="+", default=["adabeam", "gradabeam"])
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(5)))
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument(
        "--oracle-cost-budget",
        type=float,
        help="Stop each trial at this model-wrapper cost, subject to --steps.",
    )
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=30)
    parser.add_argument(
        "--no-pbt", action="store_true", help="Disable GradaBeam PBT for a timing run."
    )
    parser.add_argument(
        "--landscape-seed",
        type=int,
        help="Fix a task landscape while varying optimizer seeds (used for NK gates).",
    )
    args = parser.parse_args()
    results = [
        run_trial(
            task,
            optimizer,
            seed,
            args.steps,
            args.landscape_seed,
            args.sequence_length,
            not args.no_pbt,
            args.oracle_cost_budget,
            repeat,
        )
        for task in args.tasks
        for optimizer in args.optimizers
        for seed in args.seeds
        for repeat in range(args.repeats)
    ]
    payload = {
        "source_dir": os.path.abspath(args.source_dir) if args.source_dir else None,
        "tasks": args.tasks,
        "optimizers": args.optimizers,
        "seeds": args.seeds,
        "steps": args.steps,
        "repeats": args.repeats,
        "oracle_cost_budget": args.oracle_cost_budget,
        "sequence_length": args.sequence_length,
        "use_pbt": not args.no_pbt,
        "landscape_seed": args.landscape_seed,
        "results": results,
    }
    Path(args.output).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
