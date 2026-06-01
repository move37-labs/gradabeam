"""Utils for testing."""

import numpy as np
import torch
from gradabeam import constants
from gradabeam.tism import TISMModelClass
from gradabeam.seq_utils import dna2tensor, dna2tensor_batch, dna2tensor_integer


class CountLetterModel(torch.nn.Module, TISMModelClass):
    """Count number of occurrences of first vocab letter."""

    def __init__(self, 
                 vocab_i: int = 1, 
                 flip_sign: bool = False, 
                 extra_channels: int = 0,
                 call_is_on_strings: bool = True,
                 add_unsqueeze_to_output: bool = False,
                 train_seq_len: int = 200,
                 vocab_len: int = 4,
                 aggregate: bool = True,
                 vocab: list[str] = constants.VOCAB,
                 ):
        super().__init__()
        self.vocab_i = vocab_i
        self.flip_sign = flip_sign
        self.extra_channels = extra_channels
        self.call_is_on_strings = call_is_on_strings
        self.add_unsqueeze_to_output = add_unsqueeze_to_output
        self.train_seq_len = train_seq_len
        self.vocab_len = vocab_len
        self.aggregate = aggregate
        self.vocab = vocab
        self.vocab_array = np.array(vocab)
        self.vocab_to_idx = {nt: i for i, nt in enumerate(vocab)}

    def forward(self, x):
        assert x.ndim == 3
        assert x.shape[1] == self.vocab_len, x.shape
        out_tensor = x[:, self.vocab_i, :]
        if self.aggregate:
            out_tensor = torch.sum(out_tensor, dim=[1])
        if self.flip_sign:
            out_tensor *= -1
        if self.extra_channels:
            out_tensor = torch.stack([out_tensor] + [torch.ones_like(out_tensor)] * self.extra_channels, dim=1)
        if self.add_unsqueeze_to_output:
            out_tensor = torch.unsqueeze(out_tensor, dim=-1)
        return out_tensor

    def inference_on_tensor(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)
    
    def inference_on_strings(self, seqs: list[str]) -> list[float]:
        torch_seq = dna2tensor_batch(seqs, vocab_list=self.vocab)
        result = self.inference_on_tensor(torch_seq)
        return [float(x) for x in result]

    def __call__(self, x):
        if self.call_is_on_strings:
            return self.inference_on_strings(x)
        else:
            return self.inference_on_tensor(x)
    
    @property
    def data_params(self):
        return {
            'tasks': {'name':[f'task{i}' for i in range(3)] + ['Neuron']},
            'train': {'seq_len': self.train_seq_len},
        }
    
    def get_task_idxs(self, *args, **kwargs):
        return [0, 1, 2]
