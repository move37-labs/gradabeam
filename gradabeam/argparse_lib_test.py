"""Tests for argparse_lib.py.

To test:
```zsh
pytest gradabeam/argparse_lib_test.py
```
"""

import os
import tempfile
import unittest

from gradabeam import argparse_lib


class ArgparseLibTest(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.test_dir.cleanup()

    # Tests for possibly_parse_start_sequence
    # ============================================

    def test_parse_start_sequence_no_prefix(self):
        """Test that a standard sequence string is returned unchanged."""
        seq = "ATGC"
        self.assertEqual(argparse_lib.possibly_parse_start_sequence(seq), "ATGC")

    def test_parse_start_sequence_from_local_file(self):
        """Test reading a sequence from a local file."""
        seq = "A" * 100
        tmp_file = os.path.join(self.test_dir.name, "seq.txt")
        with open(tmp_file, "w") as f:
            f.write(seq)

        path = f"local://{tmp_file}"
        self.assertEqual(argparse_lib.possibly_parse_start_sequence(path), seq)

    # Tests for possibly_parse_positions_to_mutate
    # ================================================

    def test_parse_positions_to_mutate_no_prefix(self):
        """Test parsing a comma-separated string of positions."""
        positions_str = "1,5,10"
        expected = [1, 5, 10]
        self.assertEqual(
            argparse_lib.possibly_parse_positions_to_mutate(positions_str), expected
        )

    def test_parse_positions_to_mutate_from_local_file(self):
        """Test reading positions from a local file."""
        positions = [1, 10, 100]
        tmp_file = os.path.join(self.test_dir.name, "pos.txt")
        with open(tmp_file, "w") as f:
            f.write("\n".join(map(str, positions)))

        path = f"local://{tmp_file}"
        self.assertEqual(
            argparse_lib.possibly_parse_positions_to_mutate(path), positions
        )

    def test_parse_positions_to_mutate_empty_and_none(self):
        """Test that empty or None inputs return None."""
        self.assertIsNone(argparse_lib.possibly_parse_positions_to_mutate(None))
        self.assertIsNone(argparse_lib.possibly_parse_positions_to_mutate(""))
        self.assertIsNone(argparse_lib.possibly_parse_positions_to_mutate([]))
