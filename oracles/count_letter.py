"""Working CountLetterModel oracle for gradabeam.

Implements the CountLetterModel demo oracle (maximizes target letter content).
Run:

    python -m gradabeam --oracle_script oracles/count_letter.py --start_sequence AAAAAAAAAA
    python -m gradabeam --oracle_script oracles/count_letter.py --start_sequence AAAAAAAAAA -- --target_char G

Interface
---------
make_oracle() returns an object with:

- __call__(seqs: list[str]) -> list[float]
      Fitness score for each sequence. Lower = better (negated count).
      Required by both GradaBeam and AdaBeam.

- get_tism(sequence: str, idxs: list[int] | None)
      -> tuple[list[tuple[int, str]], np.ndarray]
      Required only for GradaBeam (gradient-guided mutations).
"""

import argparse

from gradabeam import testing_utils


def make_oracle(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_char", type=str, default="C")
    args = parser.parse_args(argv)
    # CountLetterModel counts target_char occurrences and negates the count (lower = better).
    # The optimizer's internal negation of the score maximizes target letter content.
    return testing_utils.CountLetterModel(
        target_char=args.target_char,
    )
