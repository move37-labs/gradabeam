"""Template for a custom gradabeam oracle.

Copy this file, implement the two methods, then run:

    python -m gradabeam --oracle_script oracles/template.py --start_sequence ATGC...

Interface
---------
make_oracle() must return an object with:

- __call__(seqs: list[str]) -> list[float]
      Fitness score for each sequence. Lower = better.
      Required by both GradaBeam and AdaBeam.

- get_tism(sequence: str, idxs: list[int] | None)
      -> tuple[list[tuple[int, str]], np.ndarray]
      Required only for GradaBeam (gradient-guided mutations).
      Returns (pos_and_chars_to_mutate, logits).
      See gradabeam/tism.TISMModelClass for a reference implementation.
"""

import numpy as np


def make_oracle():
    return MyOracle()


class MyOracle:

    def __call__(self, seqs: list[str]) -> list[float]:
        raise NotImplementedError

    def get_tism(self, sequence: str, idxs: list[int] | None = None) -> tuple[list[tuple[int, str]], np.ndarray]:
        raise NotImplementedError
