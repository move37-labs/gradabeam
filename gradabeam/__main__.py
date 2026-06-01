"""Standalone CLI entrypoint for GradaBeam / AdaBeam sequence optimizers.

Runs in demo mode using a built-in CountLetterModel oracle (maximizes C-content
in the sequence). To use a real oracle, import GradaBeam or AdaBeam directly.

Usage
-----
    python -m gradabeam [options]

Examples
--------
    # GradaBeam on a short sequence (10 steps, demo oracle)
    python -m gradabeam --start_sequence AAAAAAAAAA --n_steps 10

    # AdaBeam variant
    python -m gradabeam --optimizer adabeam --start_sequence AAAAAAAAAA --n_steps 10

    # Load sequence from a local text file
    python -m gradabeam --start_sequence local://seq.txt --n_steps 5

    # Load from the Zenodo Enformer dataset (requires network access)
    python -m gradabeam --start_sequence enformer://12 --n_steps 3
"""

import argparse
import sys

from gradabeam import argparse_lib, constants, testing_utils
from gradabeam.gradabeam_optimizer import GradaBeam
from gradabeam.adabeam_optimizer import AdaBeam


_OPTIMIZERS = {
    'gradabeam': GradaBeam,
    'adabeam': AdaBeam,
}


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='python -m gradabeam',
        description=(
            'Run GradaBeam or AdaBeam in demo mode.\n'
            'The demo oracle (CountLetterModel) maximizes C-content in the sequence.\n'
            'To use a real oracle, import GradaBeam / AdaBeam directly.'
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ------------------------------------------------------------------ #
    # Main / shared args                                                   #
    # ------------------------------------------------------------------ #
    p.add_argument(
        '--optimizer',
        choices=list(_OPTIMIZERS),
        default='gradabeam',
        help='Which optimizer to run.',
    )
    p.add_argument(
        '--start_sequence',
        required=True,
        help=(
            'Starting DNA/RNA sequence. Supports special prefixes:\n'
            '  local://<path>   — read the sequence from a local file\n'
            '  enformer://<idx> — fetch from the Zenodo Enformer dataset'
        ),
    )
    p.add_argument(
        '--positions_to_mutate',
        default=None,
        help=(
            'Comma-separated 0-based positions to mutate, e.g. "0,1,2,50". '
            'Also supports local:// and enformer:// prefixes. '
            'Defaults to all positions.'
        ),
    )
    p.add_argument(
        '--n_steps',
        type=int,
        default=10,
        help='Number of optimization steps to run.',
    )
    p.add_argument(
        '--n_output_seqs',
        type=int,
        default=5,
        help='Number of top sequences to print at the end.',
    )

    # ------------------------------------------------------------------ #
    # Shared optimizer args (present in both GradaBeam and AdaBeam)       #
    # ------------------------------------------------------------------ #
    shared = p.add_argument_group('Shared optimizer options')
    shared.add_argument('--beam_size', type=int, default=10,
                        help='Number of candidates to keep between rounds.')
    shared.add_argument(
        '--mutations_per_sequence',
        type=float,
        default=None,
        help=(
            'Expected number of mutations per rollout step. '
            'Defaults to max(1, 1%% of mutable positions).'
        ),
    )
    shared.add_argument('--n_rollouts_per_root', type=int, default=4,
                        help='Rollouts launched from each beam candidate per round.')
    shared.add_argument('--eval_batch_size', type=int, default=1,
                        help='Sequences sent to the oracle per batch call.')
    shared.add_argument('--rng_seed', type=int, default=42)
    shared.add_argument('--max_rollout_len', type=int, default=200,
                        help='Maximum rollout depth before stopping.')
    shared.add_argument('--debug', action='store_true', default=False,
                        help='Print debug information during optimization.')

    # ------------------------------------------------------------------ #
    # GradaBeam-only args                                                  #
    # ------------------------------------------------------------------ #
    gb = p.add_argument_group('GradaBeam-only options')
    gb.add_argument(
        '--exploration_alpha',
        type=float,
        default=0.05,
        help=(
            'Mix between gradient-guided (0.0) and uniform-random (1.0) mutations. '
            'Adaptively updated by PBT when --use_pbt is true.'
        ),
    )
    gb.add_argument('--gradient_prob_cap', type=float, default=0.10,
                    help='Per-action probability cap applied after softmax.')
    gb.add_argument('--max_logit', type=float, default=3.0,
                    help='Dynamic temperature ceiling for TISM logit scaling.')
    gb.add_argument(
        '--use_pbt',
        type=argparse_lib.str_to_bool,
        default=True,
        metavar='BOOL',
        help='Enable Population Based Training for adaptive mutation rate.',
    )

    # ------------------------------------------------------------------ #
    # AdaBeam-only args                                                    #
    # ------------------------------------------------------------------ #
    ab = p.add_argument_group('AdaBeam-only options')
    ab.add_argument(
        '--skip_repeat_sequences',
        type=argparse_lib.str_to_bool,
        default=True,
        metavar='BOOL',
        help='Skip sequences already evaluated during rollouts.',
    )

    return p


def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)

    # ------------------------------------------------------------------ #
    # Resolve start sequence and positions                                 #
    # ------------------------------------------------------------------ #
    start_sequence = argparse_lib.possibly_parse_start_sequence(args.start_sequence)
    positions_to_mutate = argparse_lib.possibly_parse_positions_to_mutate(
        args.positions_to_mutate
    )

    n_mutable = len(positions_to_mutate) if positions_to_mutate else len(start_sequence)

    # ------------------------------------------------------------------ #
    # Default mutations_per_sequence                                       #
    # ------------------------------------------------------------------ #
    mutations_per_sequence = args.mutations_per_sequence
    if mutations_per_sequence is None:
        mutations_per_sequence = max(1.0, n_mutable * 0.01)
        print(
            f'[INFO] --mutations_per_sequence not set; using {mutations_per_sequence:.2f}',
            file=sys.stderr,
        )

    # ------------------------------------------------------------------ #
    # Demo oracle                                                          #
    # ------------------------------------------------------------------ #
    # CountLetterModel(vocab_i=1) counts 'C' occurrences.
    # flip_sign=True makes model output negative (–C_count).
    # ModelWrapper.get_fitness negates the model output, so the optimizer
    # internally maximizes +C_count (higher C-content = better fitness).
    model_fn = testing_utils.CountLetterModel(
        vocab_i=constants.VOCAB.index('C'),
        flip_sign=True,
    )

    # ------------------------------------------------------------------ #
    # Print run summary                                                    #
    # ------------------------------------------------------------------ #
    seq_display = start_sequence if len(start_sequence) <= 60 else start_sequence[:57] + '...'
    print(f'Optimizer            : {args.optimizer}')
    print(f'Sequence ({len(start_sequence):,} bp)  : {seq_display}')
    print(f'Mutable positions    : {n_mutable}')
    print(f'Mutations/step       : {mutations_per_sequence:.2f}')
    print(f'Steps                : {args.n_steps}')
    print(f'Beam size            : {args.beam_size}')
    print(f'Oracle               : CountLetterModel (maximizes C-content — demo only)')
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
    if args.optimizer == 'gradabeam':
        optimizer = GradaBeam(
            **shared_kwargs,
            exploration_alpha=args.exploration_alpha,
            gradient_prob_cap=args.gradient_prob_cap,
            max_logit=args.max_logit,
            use_pbt=args.use_pbt,
        )
    elif args.optimizer == 'adabeam':
        optimizer = AdaBeam(
            **shared_kwargs,
            skip_repeat_sequences=args.skip_repeat_sequences,
        )
    else:
        parser.error(f'Unknown optimizer: {args.optimizer}')

    # ------------------------------------------------------------------ #
    # Run                                                                  #
    # ------------------------------------------------------------------ #
    optimizer.run(n_steps=args.n_steps)

    # ------------------------------------------------------------------ #
    # Output                                                               #
    # ------------------------------------------------------------------ #
    top_seqs = optimizer.get_samples(args.n_output_seqs)
    print(f'\nTop {len(top_seqs)} sequence(s) after {args.n_steps} step(s):')
    for rank, seq in enumerate(top_seqs, 1):
        n_c = seq.count('C')
        frac_c = n_c / len(seq)
        print(f'  [{rank}] {seq[:80]}{"..." if len(seq) > 80 else ""}  '
              f'(C-count: {n_c}, C-frac: {frac_c:.2%})')


if __name__ == '__main__':
    main()
