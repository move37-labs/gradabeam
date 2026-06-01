"""Sequence encoding utilities."""

import torch
import torch.nn.functional as F
from gradabeam import constants


def dna2tensor_integer(sequence_str: str, vocab_list: list[str] = constants.VOCAB) -> torch.Tensor:
    vocab_map = {nt: i for i, nt in enumerate(vocab_list)}
    return torch.tensor([vocab_map[c] for c in sequence_str], dtype=torch.long)


def dna2tensor(sequence_str: str, vocab_list: list[str] = constants.VOCAB) -> torch.Tensor:
    int_tensor = dna2tensor_integer(sequence_str, vocab_list)
    one_hot_tensor = F.one_hot(int_tensor, num_classes=len(vocab_list))
    return one_hot_tensor.T.float()


def dna2tensor_batch(sequence_strs: list[str], vocab_list: list[str] = constants.VOCAB) -> torch.Tensor:
    vocab_map = {nt: i for i, nt in enumerate(vocab_list)}
    int_tensor = torch.tensor([[vocab_map[c] for c in seq] for seq in sequence_strs], dtype=torch.long)
    one_hot_tensor = F.one_hot(int_tensor, num_classes=len(vocab_list))
    return one_hot_tensor.permute(0, 2, 1).float()
