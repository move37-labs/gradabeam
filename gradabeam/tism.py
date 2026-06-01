"""TISM (in silico mutagenesis) utilities and base mixin class."""

import numpy as np
import torch
from gradabeam.seq_utils import dna2tensor, dna2tensor_integer


def apply_gradient_mask(
    x: torch.Tensor, idxs: list[int]
) -> tuple[torch.Tensor, torch.Tensor]:
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
    assert x_grad.grad is not None
    grads = x_grad.grad.detach().cpu()
    return grads


def grad_torch_to_tism_torch(
    sg_tensor: torch.Tensor, base_seq: torch.Tensor
) -> torch.Tensor:
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
    """Mixin that adds TISM (gradient-guided mutagenesis) to a model.

    Subclasses must provide:
      - self.vocab: list[str]
      - self.vocab_array: np.ndarray
      - self.vocab_to_idx: dict[str, int]
      - self.inference_on_tensor(x: torch.Tensor) -> torch.Tensor
    """

    vocab: list[str]
    vocab_array: np.ndarray
    vocab_to_idx: dict[str, int]

    def inference_on_tensor(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def str2tensor(self, x: str) -> torch.Tensor:
        assert hasattr(self, "vocab"), "Vocab not set."
        return dna2tensor(x, vocab_list=self.vocab)

    def tensor2int(self, x: torch.Tensor) -> torch.Tensor:
        return dna2tensor_integer(
            self.vocab_array[x.argmax(dim=0)].tolist(), vocab_list=self.vocab
        )

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
            x_effective = "".join([x[idx] for idx in idxs])

        vocab_map = {nt: i for i, nt in enumerate(self.vocab)}
        base_seq_idx = torch.tensor(
            [vocab_map[c] for c in x_effective], dtype=torch.long
        )

        tism_tensor = grad_torch_to_tism_torch(
            torch.squeeze(sg_tensor, dim=0), base_seq_idx
        )
        return tism_tensor

    def get_tism(
        self, sequence: str, idxs: list[int] | None = None
    ) -> tuple[list[tuple[int, str]], np.ndarray]:
        assert hasattr(self, "vocab_to_idx"), 'missing "vocab_to_idx".'
        assert hasattr(self, "vocab_array"), 'missing "vocab_array".'
        tism_tensor = self.tism_torch(sequence, idxs)
        vocab_size, tism_seq_len = tism_tensor.shape

        if idxs is None:
            positions_to_mutate = np.arange(len(sequence), dtype=np.int32)
        else:
            positions_to_mutate = np.array(idxs, dtype=np.int32)

        assert len(positions_to_mutate) == tism_seq_len

        if tism_tensor.device.type != "cpu":
            tism_np = tism_tensor.cpu().numpy()
        else:
            tism_np = tism_tensor.numpy()

        base_seq_chars = np.array([sequence[pos] for pos in positions_to_mutate])
        base_seq_indices = np.array(
            [self.vocab_to_idx[char] for char in base_seq_chars]
        )

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

        pos_and_chars_to_mutate = list(
            zip(valid_positions.tolist(), valid_vocab.tolist())
        )
        logits = valid_logits.astype(np.float32)

        return (pos_and_chars_to_mutate, logits)
