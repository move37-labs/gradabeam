"""SubstringCount oracle for gradabeam.

Taken from nucleobench/models/substring_count_net in git remote public-base.

Maximizes the number of occurrences of a target substring in the sequence,
using a convolutional scoring function.

Run:

    python -m gradabeam --oracle_script oracles/substring_count.py \
        --start_sequence AAAAAAAAAAAAAAAAAAA -- --substring ATGTC

    python -m gradabeam --optimizer adabeam \
        --oracle_script oracles/substring_count.py \
        --start_sequence AAAAAAAAAAAAAAAAAAA -- --substring ATGTC

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

import numpy as np
import torch
import torch.nn.functional as F

from gradabeam import constants
from gradabeam import seq_utils
from gradabeam import tism


class CountSubstringModel(torch.nn.Module, tism.TISMModelClass):
    """Count occurrences of a substring via convolutions.

    Uses a convolutional filter matched to the target substring.
    The output is the sum of squared conv responses — nonlinear so that
    a single window fully matching the substring scores higher than two
    windows with partial matches.

    Scores are negated so that lower = more substrings (gradabeam convention).
    """

    def __init__(
        self,
        substring: str,
        vocab: list[str] = constants.VOCAB,
    ):
        super().__init__()
        self.substring = substring
        self.vocab = vocab
        self.vocab_to_idx = {nt: i for i, nt in enumerate(vocab)}
        self.vocab_array = np.array(vocab)

        substr_tensor = seq_utils.dna2tensor(substring, vocab_list=vocab)
        # Shape: [1, vocab_size, substr_len] — one filter for conv1d.
        self.substr_tensor = torch.unsqueeze(substr_tensor, dim=0)
        self.substr_tensor.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Score a batch of one-hot sequences. Lower output = better (more substrings)."""
        assert x.ndim == 3
        assert x.shape[1] == len(self.vocab), x.shape

        out = F.conv1d(x, self.substr_tensor)  # [batch, 1, seq_len - substr_len + 1]
        out = torch.squeeze(out, dim=1)  # [batch, seq_len - substr_len + 1]
        out = torch.square(out)  # nonlinear: full match >> partial match
        out = torch.sum(out, dim=1)  # [batch]

        # Negate so that lower = better, matching gradabeam convention.
        out = out * -1
        return out

    def inference_on_tensor(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

    def __call__(self, seqs: list[str], return_debug_info: bool = False):
        if isinstance(seqs, str):
            raise ValueError(
                f"CountSubstringModel input must be a list of strings, not a single string: {seqs!r}"
            )
        torch_seq = seq_utils.dna2tensor_batch(seqs, vocab_list=self.vocab)
        result = self.inference_on_tensor(torch_seq)
        scores = [float(v) for v in result]
        if return_debug_info:
            return scores, {}
        return scores


def make_oracle(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--substring",
        type=str,
        required=True,
        help="Target substring to maximize in the sequence.",
    )
    args = parser.parse_args(argv)
    return CountSubstringModel(substring=args.substring)
