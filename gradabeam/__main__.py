"""Standalone CLI entrypoint for GradaBeam / AdaBeam sequence optimizers.

Runs sequence optimization using a user-specified oracle script.

Usage
-----
    python -m gradabeam \
        --optimizer gradabeam \
        --oracle_script oracles/count_letter.py \
        --start_sequence AAAAAAAAAA \
        --time_budget 15 \
        --beam_size 2 \
        --mutations_per_sequence 2.0 \
        --exploration_alpha 0.0 \
        --n_rollouts_per_root 4 \
        --use_pbt True

Examples
--------

    # AdaBeam on substring_count — adaptive search; also shows oracle-arg passthrough
    python -m gradabeam \
        --optimizer adabeam \
        --oracle_script oracles/substring_count.py \
        --substring ATGTC \
        --start_sequence AAAAAAAAAAAAAAAAAAAA \
        --time_budget 15 \
        --beam_size 2 \
        --mutations_per_sequence 1.0 \
        --n_rollouts_per_root 4

    # GradaBeam with the BPNet neural-network oracle on a real biological sequence
    python -m gradabeam \
        --optimizer gradabeam \
        --oracle_script oracles/bpnet.py \
        --protein ATAC \
        --start_sequence local://ATAC_start_seq.txt \
        --time_budget 300 \
        --beam_size 2 \
        --mutations_per_sequence 2.0 \
        --n_rollouts_per_root 4  \
        --use_pbt False \
        --debug True
"""

import argparse
import importlib.util
import inspect
import time

from gradabeam import argparse_lib
from gradabeam.gradabeam_designer import GradaBeam
from gradabeam.adabeam_designer import AdaBeam


_OPTIMIZERS = {
    "gradabeam": GradaBeam,
    "adabeam": AdaBeam,
}


def _load_oracle(path: str, unknown_args: list[str] | None = None):
    """Load make_oracle() from a user-supplied script file."""
    spec = importlib.util.spec_from_file_location("_user_oracle", path)
    assert spec is not None, f"Could not load module spec from {path!r}"
    assert spec.loader is not None, f"Module spec has no loader for {path!r}"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    if not hasattr(module, "make_oracle"):
        raise AttributeError(f"{path!r} must define a make_oracle() function.")

    # Check if make_oracle accepts arguments
    sig = inspect.signature(module.make_oracle)
    if len(sig.parameters) > 0:
        return module.make_oracle(unknown_args)
    else:
        if unknown_args:
            raise ValueError(f"Unrecognized arguments: {unknown_args}")
        return module.make_oracle()


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m gradabeam",
        description=(
            "Run GradaBeam or AdaBeam sequence optimization.\n"
            "Requires a custom oracle script via --oracle_script."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ------------------------------------------------------------------ #
    # Main / shared args                                                   #
    # ------------------------------------------------------------------ #
    p.add_argument(
        "--optimizer",
        choices=list(_OPTIMIZERS),
        default="gradabeam",
        help="Which optimizer to run.",
    )
    p.add_argument(
        "--oracle_script",
        required=True,
        metavar="PATH",
        help=(
            "Path to a Python file that defines make_oracle() -> oracle. "
            "See oracles/template.py for a working starting point."
        ),
    )
    p.add_argument(
        "--start_sequence",
        required=True,
        help=(
            "Starting sequence. Supports special prefixes:\n"
            "  local://<path>   — read the sequence from a local file"
        ),
    )
    p.add_argument(
        "--positions_to_mutate",
        default=None,
        help=(
            'Comma-separated 0-based positions to mutate, e.g. "0,1,2,50". '
            "Also supports local:// prefix. "
            "Defaults to all positions."
        ),
    )
    termination = p.add_mutually_exclusive_group(required=True)
    termination.add_argument(
        "--n_steps",
        type=int,
        help="Number of optimization steps to run.",
    )
    termination.add_argument(
        "--time_budget",
        type=float,
        metavar="SECONDS",
        help="Wall-clock time budget in seconds (alternative to --n_steps).",
    )
    p.add_argument(
        "--n_output_seqs",
        type=int,
        default=5,
        help="Number of top sequences to print at the end.",
    )

    # ------------------------------------------------------------------ #
    # Shared optimizer args (present in both GradaBeam and AdaBeam)       #
    # ------------------------------------------------------------------ #
    shared = p.add_argument_group("Shared optimizer options")
    shared.add_argument(
        "--beam_size",
        type=int,
        required=True,
        help="Number of candidates to keep between rounds.",
    )
    shared.add_argument(
        "--mutations_per_sequence",
        type=float,
        required=True,
        help="Expected number of mutations per rollout step.",
    )
    shared.add_argument(
        "--n_rollouts_per_root",
        type=int,
        required=True,
        help="Rollouts launched from each beam candidate per round.",
    )
    shared.add_argument(
        "--eval_batch_size",
        type=int,
        default=1,
        help="Sequences sent to the oracle per batch call.",
    )
    shared.add_argument(
        "--rng_seed",
        type=int,
        default=42,
        help="Seed for the pseudo-random number generator.",
    )
    shared.add_argument(
        "--max_rollout_len",
        type=int,
        default=200,
        help="Maximum rollout depth before stopping.",
    )
    shared.add_argument(
        "--debug",
        type=argparse_lib.str_to_bool,
        default=False,
        metavar="BOOL",
        help="Print debug information during optimization.",
    )

    # ------------------------------------------------------------------ #
    # GradaBeam-only args                                                  #
    # ------------------------------------------------------------------ #
    gb = p.add_argument_group("GradaBeam-only options (ignored for adabeam)")
    gb.add_argument(
        "--exploration_alpha",
        type=float,
        default=None,
        help=(
            "Mix between gradient-guided (0.0) and uniform-random (1.0) mutations. "
            "Adaptively updated by PBT when --use_pbt is true. "
            "[default: 0.5]"
        ),
    )
    gb.add_argument(
        "--gradient_prob_cap",
        type=float,
        default=None,
        help="Per-action probability cap applied after softmax. [default: 0.10]",
    )
    gb.add_argument(
        "--max_logit",
        type=float,
        default=None,
        help="Dynamic temperature ceiling for TISM logit scaling. [default: 3.0]",
    )
    gb.add_argument(
        "--use_pbt",
        type=argparse_lib.str_to_bool,
        default=None,
        metavar="BOOL",
        help="Enable Population Based Training for adaptive mutation rate. Required for gradabeam.",
    )

    return p


def main(argv=None):
    parser = _build_parser()
    args, unknown_args = parser.parse_known_args(argv)

    # ------------------------------------------------------------------ #
    # Resolve start sequence and positions                                 #
    # ------------------------------------------------------------------ #
    start_sequence = argparse_lib.possibly_parse_start_sequence(args.start_sequence)
    positions_to_mutate = argparse_lib.possibly_parse_positions_to_mutate(
        args.positions_to_mutate
    )

    n_mutable = len(positions_to_mutate) if positions_to_mutate else len(start_sequence)
    mutations_per_sequence = args.mutations_per_sequence

    # ------------------------------------------------------------------ #
    # Load custom oracle                                                   #
    # ------------------------------------------------------------------ #
    model_fn = _load_oracle(args.oracle_script, unknown_args)

    # ------------------------------------------------------------------ #
    # Print run summary                                                    #
    # ------------------------------------------------------------------ #
    seq_display = (
        start_sequence if len(start_sequence) <= 60 else start_sequence[:57] + "..."
    )
    print(
        f"Optimizer            : {args.optimizer}"
    )
    print(f"Sequence ({len(start_sequence):,} bp)  : {seq_display}")
    print(f"Mutable positions    : {n_mutable}")
    print(f"Mutations/step       : {mutations_per_sequence:.2f}")
    if args.n_steps is not None:
        print(f"Steps                : {args.n_steps}")
    else:
        print(f"Time budget          : {args.time_budget}s")
    print(f"Beam size            : {args.beam_size}")
    print(f"Oracle               : {args.oracle_script}")
    print()

    # ------------------------------------------------------------------ #
    # Build shared kwargs                                                  #
    # ------------------------------------------------------------------ #
    shared_kwargs = dict(
        model_fn=model_fn,
        start_sequence=start_sequence,
        positions_to_mutate=positions_to_mutate,
        mutations_per_sequence=mutations_per_sequence,
        beam_size=args.beam_size,
        n_rollouts_per_root=args.n_rollouts_per_root,
        eval_batch_size=args.eval_batch_size,
        rng_seed=args.rng_seed,
        max_rollout_len=args.max_rollout_len,
        debug=args.debug,
    )

    # ------------------------------------------------------------------ #
    # Instantiate optimizer                                                #
    # ------------------------------------------------------------------ #
    if args.optimizer == "gradabeam":
        if args.use_pbt is None:
            parser.error("--use_pbt is required when --optimizer is gradabeam")
        optimizer = GradaBeam(
            **shared_kwargs,
            exploration_alpha=args.exploration_alpha
            if args.exploration_alpha is not None
            else 0.5,
            gradient_prob_cap=args.gradient_prob_cap
            if args.gradient_prob_cap is not None
            else 0.10,
            max_logit=args.max_logit if args.max_logit is not None else 3.0,
            use_pbt=args.use_pbt,
        )
    elif args.optimizer == "adabeam":
        optimizer = AdaBeam(**shared_kwargs)
    else:
        parser.error(f"Unknown optimizer: {args.optimizer}")

    # ------------------------------------------------------------------ #
    # Run                                                                  #
    # ------------------------------------------------------------------ #
    start_time = time.perf_counter()
    steps_run = 0

    if args.n_steps is not None:
        optimizer.run(n_steps=args.n_steps)
        steps_run = args.n_steps
    else:
        deadline = start_time + args.time_budget
        while time.perf_counter() < deadline:
            optimizer.run(n_steps=1)
            steps_run += 1

    total_step_time = time.perf_counter() - start_time
    time_per_step = total_step_time / steps_run if steps_run else 0.0

    # ------------------------------------------------------------------ #
    # Output                                                               #
    # ------------------------------------------------------------------ #
    top_seqs = optimizer.get_samples(args.n_output_seqs)
    print(
        f"\nTop {len(top_seqs)} sequence(s) after {steps_run} step(s) "
        f"({total_step_time:.2f}s in optimizer steps), ({time_per_step:.2f}s per step):"
    )
    scores = model_fn(top_seqs)
    for rank, (seq, score) in enumerate(zip(top_seqs, scores), 1):
        print(
            f"  [{rank}] {seq[:80]}{'...' if len(seq) > 80 else ''}  "
            f"(oracle score: {score:.4f})"
        )


if __name__ == "__main__":
    main()
