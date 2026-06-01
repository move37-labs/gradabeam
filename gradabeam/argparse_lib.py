"""Utilities for parsing arguments.

To verify that Zenodo resources are reachable:
    python -m gradabeam.argparse_lib
"""

from typing import Any, Iterable, Optional, Union

import argparse
import dataclasses
import pandas as pd

from gradabeam import constants


@dataclasses.dataclass
class ParsedArgs:
    main_args: argparse.Namespace
    model_init_args: argparse.Namespace
    opt_init_args: argparse.Namespace


def possibly_parse_start_sequence(start_seq: str) -> str:
    """Possibly parse start sequence from a local or remote file.

    Prefix strings that trigger special handling:
    - ``local://``: Load from a local file.
    """
    if start_seq.startswith("local://"):
        local_fileloc = start_seq[len("local://") :]
        with open(local_fileloc, "r") as f:
            start_seq = f.read()
    return start_seq


def possibly_parse_positions_to_mutate(
    positions_to_mutate: Optional[Union[str, list[int]]],
) -> Optional[list[int]]:
    """Possibly parse ``positions_to_mutate`` from a file, or pass it through unchanged.

    Prefix strings that trigger special handling:
    - ``local://``: Load a newline-separated list of integers from a local file.
    """
    if isinstance(positions_to_mutate, str) and positions_to_mutate.startswith(
        "local://"
    ):
        local_fileloc = positions_to_mutate[len("local://") :]
        with open(local_fileloc, "r") as f:
            loc_str = f.read()
        positions_to_mutate = [int(x) for x in loc_str.split("\n") if x.strip()]
    elif (
        positions_to_mutate is None
        or positions_to_mutate == ""
        or positions_to_mutate == []
    ):
        positions_to_mutate = None
    elif isinstance(positions_to_mutate, list):
        positions_to_mutate = [int(x) for x in positions_to_mutate]
    else:
        assert isinstance(positions_to_mutate, str), type(positions_to_mutate)
        positions_to_mutate = [int(x) for x in positions_to_mutate.split(",")]
    return positions_to_mutate


def handle_leftover_args(known_args: argparse.Namespace, leftover_args: Iterable):
    """Handle leftover arguments, either by failing or by ignoring them."""
    if known_args.ignore_empty_cmd_args:
        for i in leftover_args:
            if i.startswith("--"):
                if "=" in i:
                    arg_val = i.split("=")[1]
                    if arg_val not in [None, ""]:
                        raise ValueError(f"Unused arg, not empty: {leftover_args}")
                continue
            else:
                if i not in [None, ""]:
                    raise ValueError(f"Unused arg, not empty: {leftover_args}")
    else:
        raise ValueError(f"Unused args: {leftover_args}")


def str_to_bool(s):
    if s.lower() in ("yes", "true", "t", "1"):
        return True
    elif s.lower() in ("no", "false", "f", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")
