"""Template for a custom gradabeam oracle.

Copy this file, implement the two methods, then run:

    python -m gradabeam --oracle_script oracles/template.py --start_sequence ATGC...

Interface
---------
make_oracle() must return an object with:

- __call__(seqs: list[str]) -> list[float]
      Fitness score for each sequence. Lower = better.
      Required by both GradaBeam and AdaBeam.

- tism_torch(sequence, idxs=None) -> torch.Tensor
      Required by GradaBeam. The easiest way to supply this is to inherit
      ``gradabeam.tism.TISMModelClass`` and implement ``inference_on_tensor``
      plus the vocab attributes, as this template does.

This template is a runnable skeleton: it scores the count of ``C`` bases
and exposes TISM gradients through ``TISMModelClass``.
"""

import numpy as np
import torch

from gradabeam import constants
from gradabeam.seq_utils import dna2tensor_batch
from gradabeam.tism import TISMModelClass


def make_oracle():
    return TemplateOracle()


class TemplateOracle(torch.nn.Module, TISMModelClass):
    """Minimal GradaBeam-compatible oracle.

    Replace ``inference_on_tensor`` / ``__call__`` with your model. Keep the
    vocab attributes and inherit ``TISMModelClass`` so GradaBeam can call
    ``tism_torch``.
    """

    def __init__(self, target_char: str = "C"):
        super().__init__()
        self.vocab = constants.VOCAB
        self.vocab_array = np.array(self.vocab)
        self.vocab_to_idx = {nt: i for i, nt in enumerate(self.vocab)}
        if target_char not in self.vocab:
            raise ValueError(f"target_char {target_char!r} must be in {self.vocab}")
        self.target_char = target_char
        self._target_i = self.vocab.index(target_char)

    def inference_on_tensor(self, x: torch.Tensor) -> torch.Tensor:
        # [batch, vocab, length] -> negated target-letter count.
        return -torch.sum(x[:, self._target_i, :], dim=1)

    def __call__(self, seqs: list[str]) -> list[float]:
        tensor = dna2tensor_batch(seqs, vocab_list=self.vocab)
        return [float(v) for v in self.inference_on_tensor(tensor)]
