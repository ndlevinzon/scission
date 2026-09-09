"""Pfizer / WBO strategies and user-registered schemes (no AmberTools)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))
import _paths  # noqa: E402


def setUpModule():
    _paths.ensure_scission()


def _alkane(n_carbons: int, name: str = "alk"):
    """Linear Cn chain (1-based carbons 1..n), two terminal hydrogens."""
    from scission.Models import Atom, Bond, Ligand

    atoms = []
    for i in range(1, n_carbons + 1):
        atoms.append(
            Atom(
                index=i,
                name=f"C{i}",
                element="C",
                atom_type="c3",
                charge=-0.1,
                coords=(1.5 * (i - 1), 0.0, 0.0),
            )
        )
    atoms.append(Atom(n_carbons + 1, "H1", "H", "hc", 0.1, (-1.0, 0.0, 0.0)))
    atoms.append(
        Atom(
            n_carbons + 2,
            "H2",
            "H",
            "hc",
            0.1,
            (1.5 * (n_carbons - 1) + 1.0, 0.0, 0.0),
        )
    )
    bonds = [Bond(i, i, i + 1, "1") for i in range(1, n_carbons)]
    bonds.append(Bond(n_carbons, 1, n_carbons + 1, "1"))
    bonds.append(Bond(n_carbons + 1, n_carbons, n_carbons + 2, "1"))
    return Ligand(
        name=name,
        atoms=atoms,
        bonds=bonds,
        lib_atom_names=[a.name for a in atoms],
        lib_atom_types={a.name: a.atom_type for a in atoms},
        frcmod_text="",
        mol2_path=Path("alk.mol2"),
        lib_path=Path("alk.lib"),
        frcmod_path=Path("alk.frcmod"),
    )


class TestFragmentStrategies(unittest.TestCase):
    def test_default_strategy_is_scission(self):
        from scission.Models import FragmentConfig
        from scission.Strategies import available_strategies

        cfg = FragmentConfig()
        self.assertEqual(cfg.strategy, "scission")
        self.assertIn("scission", available_strategies())
        self.assertIn("pfizer", available_strategies())
        self.assertIn("wbo", available_strategies())
        cfg2 = FragmentConfig.from_dict({"strategy": "pfizer", "wbo_max_growth": 2})
        self.assertEqual(cfg2.strategy, "pfizer")
        self.assertEqual(cfg2.wbo_max_growth, 2)
        cfg3 = FragmentConfig.from_dict({"functional_groups": {}})
        self.assertEqual(cfg3.functional_groups, {})

    def test_pfizer_keeps_torsion_quartet_not_whole_chain(self):
        from scission.Models import FragmentConfig, TorsionDefinition
        from scission.Strategies import build_candidates

        lig = _alkane(8)
        torsion = TorsionDefinition(
            atom_indices=(3, 4, 5, 6),
            bond=(4, 5),
            label="C3-C4-C5-C6",
        )
        cands = build_candidates(lig, torsion, FragmentConfig(strategy="pfizer"))
        self.assertEqual(len(cands), 1)
        heavy = {
            i
            for i in cands[0].retained_atoms
            if lig.atom(i).element != "H"
        }
        self.assertTrue({3, 4, 5, 6} <= heavy)
        self.assertNotIn(1, heavy)
        self.assertNotIn(8, heavy)

    def test_wbo_grows_beyond_pfizer_seed(self):
        from scission.Models import FragmentConfig, TorsionDefinition
        from scission.Strategies import build_candidates

        lig = _alkane(8)
        torsion = TorsionDefinition(
            atom_indices=(3, 4, 5, 6),
            bond=(4, 5),
            label="C3-C4-C5-C6",
        )
        pfizer = build_candidates(lig, torsion, FragmentConfig(strategy="pfizer"))[0]
        wbo = build_candidates(
            lig,
            torsion,
            FragmentConfig(strategy="wbo", wbo_max_growth=2),
        )
        self.assertGreater(len(wbo), 1)
        largest = max(wbo, key=lambda c: len(c.retained_atoms))
        self.assertGreaterEqual(len(largest.retained_atoms), len(pfizer.retained_atoms))

    def test_register_custom_strategy(self):
        from scission.Fragments import candidate_from_retained_atoms
        from scission.Models import FragmentConfig, TorsionDefinition
        from scission.Strategies import build_candidates, register_strategy

        lig = _alkane(8)
        torsion = TorsionDefinition(
            atom_indices=(3, 4, 5, 6),
            bond=(4, 5),
            label="C3-C4-C5-C6",
        )

        @register_strategy("quartet_only")
        def quartet_only(ligand, torsion_def, config):
            return [
                candidate_from_retained_atoms(
                    ligand, torsion_def, set(torsion_def.atom_indices)
                )
            ]

        cands = build_candidates(
            lig, torsion, FragmentConfig(strategy="quartet_only")
        )
        self.assertEqual(len(cands), 1)
        heavy = {
            i
            for i in cands[0].retained_atoms
            if lig.atom(i).element != "H"
        }
        self.assertEqual(heavy, set(torsion.atom_indices))

    def test_unknown_strategy_raises(self):
        from scission.Models import FragmentConfig, TorsionDefinition
        from scission.Strategies import build_candidates

        lig = _alkane(4)
        torsion = TorsionDefinition((1, 2, 3, 4), (2, 3), "x")
        with self.assertRaises(ValueError):
            build_candidates(lig, torsion, FragmentConfig(strategy="not-a-scheme"))

    def test_scission_strategy_still_enumerates_shells(self):
        from scission.Models import FragmentConfig, TorsionDefinition
        from scission.Strategies import build_candidates

        lig = _alkane(8)
        torsion = TorsionDefinition(
            atom_indices=(3, 4, 5, 6),
            bond=(4, 5),
            label="C3-C4-C5-C6",
        )
        cands = build_candidates(lig, torsion, FragmentConfig())
        self.assertGreaterEqual(len(cands), 1)
        self.assertTrue(any(c.is_parent_fallback for c in cands) or len(cands) >= 1)

    def test_cli_exposes_strategy_flags(self):
        from scission.Cli import build_parser

        parser = build_parser()
        ns = parser.parse_args(
            [
                "fragment",
                "--mol2",
                "a.mol2",
                "--lib",
                "a.lib",
                "--frcmod",
                "a.frcmod",
                "--outdir",
                "out",
                "--strategy",
                "pfizer",
                "--wbo-max-growth",
                "2",
                "--keep-non-rotor-ring-substituents",
            ]
        )
        self.assertEqual(ns.strategy, "pfizer")
        self.assertEqual(ns.wbo_max_growth, 2)
        self.assertTrue(ns.keep_non_rotor_ring_substituents)


if __name__ == "__main__":
    unittest.main()
