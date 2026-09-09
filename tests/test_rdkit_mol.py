"""Formal-charge inference for Amber MOL2 -> RDKit (no AmberTools)."""

from __future__ import annotations

import io
import sys
import unittest
from contextlib import redirect_stderr
from pathlib import Path

_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))
import _paths  # noqa: E402


def setUpModule():
    _paths.ensure_scission()


def _atom(index, name, element, atom_type, charge, xyz=(0.0, 0.0, 0.0)):
    from scission.Models import Atom

    return Atom(index, name, element, atom_type, charge, xyz)


def _bond(index, a1, a2, bond_type):
    from scission.Models import Bond

    return Bond(index, a1, a2, bond_type)


class TestFormalChargeHints(unittest.TestCase):
    def test_sulfonyl_oxygen_stays_neutral(self):
        from scission.RdkitMol import _formal_charge_hint, infer_formal_charge

        self.assertIsNone(
            _formal_charge_hint("o", "O", 1, -0.50, bond_order_sum=2.0)
        )
        self.assertEqual(
            infer_formal_charge(
                "O", 1, atom_type="o", partial_charge=-0.50, bond_order_sum=2.0
            ),
            0,
        )

    def test_carboxylate_oxygen_is_anion(self):
        from scission.RdkitMol import _formal_charge_hint, infer_formal_charge

        self.assertEqual(
            _formal_charge_hint("o", "O", 1, -0.80, bond_order_sum=1.0),
            -1,
        )
        self.assertEqual(
            infer_formal_charge(
                "O", 1, atom_type="o", partial_charge=-0.80, bond_order_sum=1.0
            ),
            -1,
        )

    def test_amide_nitrogen_stays_neutral(self):
        from scission.RdkitMol import _formal_charge_hint, infer_formal_charge

        self.assertIsNone(
            _formal_charge_hint("n", "N", 3, -0.70, bond_order_sum=3.0)
        )
        self.assertEqual(
            infer_formal_charge(
                "N", 3, atom_type="n", partial_charge=-0.70, bond_order_sum=3.0
            ),
            0,
        )

    def test_quaternary_nitrogen(self):
        from scission.RdkitMol import infer_formal_charge

        self.assertEqual(
            infer_formal_charge(
                "N", 4, atom_type="n4", partial_charge=0.65, bond_order_sum=4.0
            ),
            1,
        )


@unittest.skipUnless(
    __import__("importlib").util.find_spec("rdkit") is not None,
    "rdkit not installed",
)
class TestBuildRdkitMol(unittest.TestCase):
    def test_chaps_like_sulfonate_and_amide_sanitize_quietly(self):
        """Degree-1 S=O oxygens and an amide N must not trip RDKit valence."""
        from scission.RdkitMol import build_rdkit_mol

        # S(=O)(=O)(O-)-C-N(H)-C=O  -- sulfonate + amide, CHAPS-like sites.
        atoms = [
            _atom(1, "O1", "O", "o", -0.50, (1.4, 1.2, 0.0)),
            _atom(2, "O2", "O", "o", -0.52, (1.4, -1.2, 0.0)),
            _atom(3, "O3", "O", "o", -0.80, (-1.4, 0.0, 0.0)),
            _atom(4, "S1", "S", "s6", 1.30, (0.0, 0.0, 0.0)),
            _atom(5, "C1", "C", "c3", -0.10, (0.0, 0.0, 1.8)),
            _atom(6, "H1", "H", "hc", 0.10, (0.9, 0.0, 2.2)),
            _atom(7, "H2", "H", "hc", 0.10, (-0.5, 0.9, 2.2)),
            _atom(8, "N1", "N", "n", -0.70, (0.0, 0.0, 3.2)),
            _atom(9, "H3", "H", "hn", 0.30, (0.9, 0.0, 3.6)),
            _atom(10, "C2", "C", "c", 0.60, (0.0, 0.0, 4.6)),
            _atom(11, "O4", "O", "o", -0.55, (1.2, 0.0, 5.2)),
            _atom(12, "C3", "C", "c3", -0.10, (-1.4, 0.0, 5.2)),
            _atom(13, "H4", "H", "hc", 0.10, (-1.8, 0.9, 5.2)),
            _atom(14, "H5", "H", "hc", 0.10, (-1.8, -0.9, 5.2)),
            _atom(15, "H6", "H", "hc", 0.10, (-2.0, 0.0, 6.0)),
        ]
        bonds = [
            _bond(1, 4, 1, "2"),
            _bond(2, 4, 2, "2"),
            _bond(3, 4, 3, "1"),
            _bond(4, 4, 5, "1"),
            _bond(5, 5, 6, "1"),
            _bond(6, 5, 7, "1"),
            _bond(7, 5, 8, "1"),
            _bond(8, 8, 9, "1"),
            _bond(9, 8, 10, "1"),
            _bond(10, 10, 11, "2"),
            _bond(11, 10, 12, "1"),
            _bond(12, 12, 13, "1"),
            _bond(13, 12, 14, "1"),
            _bond(14, 12, 15, "1"),
        ]
        buf = io.StringIO()
        with redirect_stderr(buf):
            mol = build_rdkit_mol(atoms, bonds, sanitize=True)
        err = buf.getvalue()
        self.assertNotIn("Explicit valence", err)
        charges = [atom.GetFormalCharge() for atom in mol.GetAtoms()]
        self.assertEqual(charges[0], 0)  # S=O
        self.assertEqual(charges[1], 0)  # S=O
        self.assertEqual(charges[2], -1)  # S-O-
        self.assertEqual(charges[7], 0)  # amide N
        self.assertEqual(mol.GetNumAtoms(), 15)

    def test_n4_headgroup_sanitizes(self):
        from scission.RdkitMol import build_rdkit_mol

        atoms = [
            _atom(1, "N1", "N", "n4", 0.65, (0.0, 0.0, 0.0)),
            _atom(2, "C1", "C", "c3", -0.10, (1.5, 0.0, 0.0)),
            _atom(3, "C2", "C", "c3", -0.10, (-0.5, 1.4, 0.0)),
            _atom(4, "C3", "C", "c3", -0.10, (-0.5, -0.7, 1.2)),
            _atom(5, "C4", "C", "c3", -0.10, (-0.5, -0.7, -1.2)),
        ]
        bonds = [
            _bond(1, 1, 2, "1"),
            _bond(2, 1, 3, "1"),
            _bond(3, 1, 4, "1"),
            _bond(4, 1, 5, "1"),
        ]
        extra_atoms = []
        extra_bonds = []
        idx = 6
        bidx = 5
        for carbon in (2, 3, 4, 5):
            for _ in range(3):
                extra_atoms.append(
                    _atom(idx, f"H{idx}", "H", "hc", 0.1, (float(idx), 0.0, 0.0))
                )
                extra_bonds.append(_bond(bidx, carbon, idx, "1"))
                idx += 1
                bidx += 1
        mol = build_rdkit_mol(atoms + extra_atoms, bonds + extra_bonds, sanitize=True)
        self.assertEqual(mol.GetAtomWithIdx(0).GetFormalCharge(), 1)


if __name__ == "__main__":
    unittest.main()
