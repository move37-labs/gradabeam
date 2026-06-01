"""Utils for testing."""

import numpy as np
import torch
from gradabeam import constants
from gradabeam.tism import TISMModelClass
from gradabeam.seq_utils import dna2tensor, dna2tensor_batch, dna2tensor_integer


class CountLetterModel(torch.nn.Module, TISMModelClass):
    """Count number of occurrences of first vocab letter."""

    def __init__(self, 
                 target_char: str = 'C', 
                 flip_sign: bool = False, 
                 vocab: list[str] = constants.VOCAB,
                 ):
        super().__init__()
        if target_char not in vocab:
            raise ValueError(f"target_char '{target_char}' must be in vocab {vocab}")
        self.target_char = target_char
        self.vocab_i = vocab.index(target_char)
        self.flip_sign = flip_sign
        self.vocab = vocab
        self.vocab_array = np.array(vocab)
        self.vocab_to_idx = {nt: i for i, nt in enumerate(vocab)}

    def forward(self, x):
        assert x.ndim == 3
        assert x.shape[1] == len(self.vocab), x.shape
        out_tensor = x[:, self.vocab_i, :]
        out_tensor = torch.sum(out_tensor, dim=[1])
        if self.flip_sign:
            out_tensor *= -1
        return out_tensor

    def inference_on_tensor(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)
    
    def inference_on_strings(self, seqs: list[str]) -> list[float]:
        torch_seq = dna2tensor_batch(seqs, vocab_list=self.vocab)
        result = self.inference_on_tensor(torch_seq)
        return [float(x) for x in result]

    def __call__(self, x):
        return self.inference_on_strings(x)
    
    @property
    def data_params(self):
        return {
            'tasks': {'name':[f'task{i}' for i in range(3)] + ['Neuron']},
            'train': {'seq_len': 200},
        }
    
    def get_task_idxs(self, *args, **kwargs):
        return [0, 1, 2]
