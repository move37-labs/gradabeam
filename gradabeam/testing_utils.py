"""Utils for testing."""

import numpy as np
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

def apply_gradient_mask(x: torch.Tensor, idxs: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    assert min(idxs) >= 0
    assert max(idxs) < x.shape[2]
    assert x.ndim == 3, x.shape
    x_grad = x[..., idxs].detach().clone()
    x_grad.requires_grad_(True)
    model_input = x.detach().clone()
    model_input[..., idxs] = x_grad
    return model_input, x_grad

def grad_torch(
    input_tensor: torch.Tensor, 
    model, 
    idxs: list[int] | None = None,
) -> torch.Tensor:
    input_tensor = input_tensor.detach()
    if idxs is None:
        input_tensor.requires_grad_(True)
        x_grad = input_tensor
    else:
        input_tensor, x_grad = apply_gradient_mask(input_tensor, idxs)
        
    y = model(input_tensor)
    y.sum().backward(retain_graph=False)
    grads = x_grad.grad.detach().cpu()
    return grads

def grad_torch_to_tism_torch(sg_tensor: torch.Tensor, base_seq: torch.Tensor) -> torch.Tensor:
    assert sg_tensor.ndim == 2
    assert base_seq.ndim == 1
    assert sg_tensor.shape[1] == base_seq.shape[0]
    vocab_size, seq_len = sg_tensor.shape
    ref_vals = sg_tensor[base_seq, torch.arange(seq_len)]
    ref_vals_expanded = ref_vals.unsqueeze(0).expand(vocab_size, seq_len)
    tism_tensor = sg_tensor - ref_vals_expanded
    tism_tensor[base_seq, torch.arange(seq_len)] = 0.0
    return tism_tensor

class TISMModelClass:
    """Model that supports TISM."""
    def str2tensor(self, x: str) -> torch.Tensor:
        assert hasattr(self, 'vocab'), 'Vocab not set.'
        return dna2tensor(x, vocab_list=self.vocab)
        
    def tensor2int(self, x: torch.Tensor) -> torch.Tensor:
        return dna2tensor_integer(self.vocab_array[x.argmax(dim=0)].tolist(), vocab_list=self.vocab)

    def tism_torch(self, x: str, idxs: list[int] | None = None) -> torch.Tensor:
        input_tensor = self.str2tensor(x)
        sg_tensor = grad_torch(
            input_tensor=torch.unsqueeze(input_tensor, dim=0),
            model=self.inference_on_tensor,
            idxs=idxs,
        )
        if idxs is None:
            x_effective = x
        else:
            x_effective = ''.join([x[idx] for idx in idxs])
        
        vocab_map = {nt: i for i, nt in enumerate(self.vocab)}
        base_seq_idx = torch.tensor([vocab_map[c] for c in x_effective], dtype=torch.long)
        
        tism_tensor = grad_torch_to_tism_torch(torch.squeeze(sg_tensor, dim=0), base_seq_idx)
        return tism_tensor

    def get_tism(self, sequence: str, idxs: list[int] | None = None) -> tuple[list[tuple[int, str]], np.ndarray]:
        assert hasattr(self, 'vocab_to_idx'), 'missing "vocab_to_idx".'
        assert hasattr(self, 'vocab_array'), 'missing "vocab_array".'
        tism_tensor = self.tism_torch(sequence, idxs)
        vocab_size, tism_seq_len = tism_tensor.shape
        
        if idxs is None:
            positions_to_mutate = np.arange(len(sequence), dtype=np.int32)
        else:
            positions_to_mutate = np.array(idxs, dtype=np.int32)
        
        assert len(positions_to_mutate) == tism_seq_len
        
        if tism_tensor.device.type != 'cpu':
            tism_np = tism_tensor.cpu().numpy()
        else:
            tism_np = tism_tensor.numpy()
            
        base_seq_chars = np.array([sequence[pos] for pos in positions_to_mutate])
        base_seq_indices = np.array([self.vocab_to_idx[char] for char in base_seq_chars])
        
        positions_array = np.repeat(positions_to_mutate, vocab_size)
        vocab_repeated = np.tile(self.vocab_array, tism_seq_len)
        vocab_indices = np.tile(np.arange(vocab_size), tism_seq_len)
        pos_indices = np.repeat(np.arange(tism_seq_len), vocab_size)
        
        tism_values = tism_np[vocab_indices, pos_indices]
        base_seq_indices_expanded = np.repeat(base_seq_indices, vocab_size)
        valid_mask = vocab_indices != base_seq_indices_expanded
        
        valid_positions = positions_array[valid_mask]
        valid_vocab = vocab_repeated[valid_mask]
        valid_logits = tism_values[valid_mask]
        
        pos_and_chars_to_mutate = list(zip(valid_positions.tolist(), valid_vocab.tolist()))
        logits = valid_logits.astype(np.float32)
        
        return (pos_and_chars_to_mutate, logits)

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
