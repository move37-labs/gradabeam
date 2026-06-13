"""BPNet oracle for gradabeam.

Taken from nucleobench/models/bpnet in git remote public-base.

Predicts binding counts for various transcription factors or proteins.

Run:

    python -m gradabeam --oracle_script oracles/bpnet.py \
        --start_sequence local://ATAC_start_seq.txt \
        --time_budget 300 --beam_size 2 --mutations_per_sequence 2.0 \
        --n_rollouts_per_root 4 --debug True -- --protein ATAC

Interface
---------
make_oracle() returns an object with:

- __call__(seqs: list[str]) -> list[float]
      Fitness score for each sequence. Lower = better (negated count prediction).
      Required by both GradaBeam and AdaBeam.

- get_tism(sequence: str, idxs: list[int] | None)
      -> tuple[list[tuple[int, str]], np.ndarray]
      Required only for GradaBeam (gradient-guided mutations).
"""

import argparse
import os
import subprocess
import numpy as np
import torch

try:
    import bpnetlite  # noqa: F401
except ImportError:
    raise ImportError(
        "The BPNet oracle requires 'bpnet-lite'. "
        "Please install it using: pip install bpnet-lite "
        "(or pip install gradabeam[examples])"
    )

from gradabeam import tism
from gradabeam import seq_utils

# Constants
VOCAB = ["A", "C", "G", "T"]
AVAILABLE_MODELS = [
    "ATAC",
    "CTCF",
    "E2F3",
    "ELF4",
    "GATA2",
    "JUNB",
    "MAX",
    "MECOM",
    "MYC",
    "OTX1",
    "RAD21",
    "SOX6",
]
RECORDS = "https://zenodo.org/records/14604495"
CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "gradabeam", "bpnet")


def get_url(model_name: str) -> str:
    assert model_name in AVAILABLE_MODELS
    return f"{RECORDS}/files/{model_name}.torch"


def get_cache_path(model_name: str) -> str:
    return os.path.join(CACHE_DIR, f"{model_name}.torch")


def download(model_name: str):
    assert model_name in AVAILABLE_MODELS

    cache_path = get_cache_path(model_name)
    if os.path.exists(cache_path):
        print(f"Loading cached BPNet model from {cache_path}")
        try:
            model = torch.load(
                cache_path, weights_only=False, map_location=torch.device("cpu")
            )
        except Exception:
            print("Cached file is corrupt, deleting and re-downloading...")
            os.remove(cache_path)

    if not os.path.exists(cache_path):
        url = get_url(model_name)
        os.makedirs(CACHE_DIR, exist_ok=True)
        print(f"Downloading BPNet model for {model_name} to {cache_path}")
        subprocess.run(["curl", url, "--output", cache_path], check=True)
        model = torch.load(
            cache_path, weights_only=False, map_location=torch.device("cpu")
        )

    model = CountWrapper(ControlWrapper(model))
    return model


class CountWrapper(torch.nn.Module):
    """A wrapper class that only returns the predicted counts."""

    def __init__(self, model):
        super(CountWrapper, self).__init__()
        self.model = model

    def forward(self, X, X_ctl=None, **kwargs):
        return self.model(X, X_ctl, **kwargs)[1]


class ControlWrapper(torch.nn.Module):
    """This wrapper automatically creates a control track of all zeroes."""

    def __init__(self, model):
        super(ControlWrapper, self).__init__()
        self.model = model

    def forward(self, X, X_ctl=None):
        if X_ctl is not None:
            return self.model(X, X_ctl)

        if self.model.n_control_tracks == 0:
            return self.model(X)

        X_ctl = torch.zeros(
            X.shape[0],
            self.model.n_control_tracks,
            X.shape[-1],
            dtype=X.dtype,
            device=X.device,
        )
        return self.model(X, X_ctl)


class BPNet(tism.TISMModelClass):
    """BPNet model trained on twelve proteins in K562 whose ChIP-seq
    data is on the ENCODE portal."""

    def __init__(
        self,
        protein: str,
        vocab: list[str] = VOCAB,
        override_model: torch.nn.Module | None = None,
    ):
        self.protein = protein
        if override_model:
            self.model = override_model
        else:
            self.model = download(protein)

        self.vocab = vocab
        self.vocab_to_idx = {nt: i for i, nt in enumerate(vocab)}
        self.vocab_array = np.array(vocab)

    def inference_on_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """Run inference on a one-hot tensor."""
        assert x.ndim == 3  # Batched.
        assert x.shape[1] == 4

        m_out = self.model(x)
        assert m_out.ndim == 2
        assert m_out.shape[1] == 1
        ret = torch.squeeze(m_out, dim=1)

        # Always return something that should be minimized, so flip the sign.
        ret *= -1
        return ret

    def inference_on_strings(self, x: list[str]) -> np.ndarray:
        tensor = seq_utils.dna2tensor_batch(x, vocab_list=self.vocab)
        ret = self.inference_on_tensor(tensor)
        return ret.detach().clone().numpy()

    def __call__(
        self, x: list[str], return_debug_info: bool = False
    ) -> "np.ndarray | tuple[np.ndarray, dict]":
        if isinstance(x, str):
            raise ValueError(
                f"BPNet input needs to be list of strings, not just string: {x}"
            )
        ret = self.inference_on_strings(x)
        if return_debug_info:
            return ret, {}
        else:
            return ret


def make_oracle(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protein", type=str, default="GATA2", choices=AVAILABLE_MODELS
    )
    args = parser.parse_args(argv)
    return BPNet(protein=args.protein)
