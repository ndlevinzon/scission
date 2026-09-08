"""Torsion helpers, fragment config, and frcmod merge (no AmberTools)."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
import warnings
from pathlib import Path

_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))
import _paths  # noqa: E402


def setUpModule():
    _paths.ensure_scission()


def _frcmod(lines) -> str:
    return (
        "Remark line goes here\nMASS\n\nBOND\n\nANGLE\n\nDIHE\n"
        + "".join(f"{ln}\n" for ln in lines)
        + "\nIMPROPER\n\nNONB\n\n"
    )


class TestScissionHelpers(unittest.TestCase):
    def test_safe_name_and_param_key(self):
        from scission.Frcmod import _normalize_param_name_to_key
        from scission.Writers import safe_name

        self.assertEqual(safe_name("foo/bar"), "foo_bar")
        self.assertIsNone(_normalize_param_name_to_key("not_a_dihe"))
        key = _normalize_param_name_to_key("LIG_ca-ca-c-o")
        self.assertEqual(len(key), 4)


class TestScissionFunctions(unittest.TestCase):
    def _butane_like_ligand(self):
        """Linear C4 chain with hydrogens - rotatable C-C bonds."""
        from scission.Models import Atom, Bond, Ligand

        atoms = []
        coords = [
            (0.0, 0.0, 0.0),
            (1.5, 0.0, 0.0),
            (3.0, 0.0, 0.0),
            (4.5, 0.0, 0.0),
        ]
        for i, xyz in enumerate(coords, start=1):
            atoms.append(
                Atom(
                    index=i,
                    name=f"C{i}",
                    element="C",
                    atom_type="c3",
                    charge=-0.1,
                    coords=xyz,
                )
            )
        atoms.append(Atom(5, "H1", "H", "hc", 0.1, (-1.0, 0.0, 0.0)))
        atoms.append(Atom(6, "H4", "H", "hc", 0.1, (5.5, 0.0, 0.0)))
        bonds = [
            Bond(1, 1, 2, "1"),
            Bond(2, 2, 3, "1"),
            Bond(3, 3, 4, "1"),
            Bond(4, 1, 5, "1"),
            Bond(5, 4, 6, "1"),
        ]
        return Ligand(
            name="but",
            atoms=atoms,
            bonds=bonds,
            lib_atom_names=[a.name for a in atoms],
            lib_atom_types={a.name: a.atom_type for a in atoms},
            frcmod_text="",
            mol2_path=Path("but.mol2"),
            lib_path=Path("but.lib"),
            frcmod_path=Path("but.frcmod"),
        )

    def test_find_rotatable_bonds_and_enumerate_torsions(self):
        from scission.Torsions import enumerate_torsions, find_rotatable_bonds

        lig = self._butane_like_ligand()
        rots = find_rotatable_bonds(lig)
        self.assertGreaterEqual(len(rots), 1)
        tors = enumerate_torsions(lig)
        self.assertGreaterEqual(len(tors), 1)
        for t in tors:
            self.assertEqual(len(t.atom_indices), 4)
            self.assertEqual(len(t.bond), 2)

    def test_fragment_config_defaults_and_from_dict(self):
        from scission.Models import FragmentConfig

        cfg = FragmentConfig()
        self.assertTrue(hasattr(cfg, "angle_step") or hasattr(cfg, "cap_strategy"))
        cfg2 = FragmentConfig.from_dict({"angle_step": 15})
        self.assertEqual(cfg2.angle_step, 15)

    def test_write_fragment_index_and_merge_accumulate(self):
        from scission.Merge import _load_fragment_update
        from scission.Models import SelectedFragment
        from scission.Writers import write_fragment_index

        frag = SelectedFragment(
            fragment_id="frag_0001",
            source_candidate_id="cand_a",
            retained_atoms=[0, 1, 2],
            cut_bonds=[(2, 3)],
            cap_atoms=[],
            torsions=["t1"],
            fit_torsions=[],
            parent_atom_map={0: 0, 1: 1, 2: 2},
            manifest_path=Path("frags/frag_0001/manifest.json"),
        )
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            path = write_fragment_index([frag], out)
            self.assertTrue(path.is_file())

            frag_dir = out / "frag_0001"
            frag_dir.mkdir()
            (frag_dir / "it01.frcmod").write_text(
                _frcmod(["c3-c3-c3-c3 1 1.00 0.0 1.", "c3-c3-c3-n  1 2.00 0.0 1."])
            )
            (frag_dir / "it02.frcmod").write_text(
                _frcmod(["c3-c3-c3-n  1 3.50 0.0 1."])
            )
            update = _load_fragment_update(frag_dir)
            self.assertIn(("c3", "c3", "c3", "c3"), update["dihe_groups"])
            self.assertIn(("c3", "c3", "c3", "n"), update["dihe_groups"])
            n_lines = update["dihe_groups"][("c3", "c3", "c3", "n")]
            self.assertTrue(any("3.50" in ln for ln in n_lines))
            self.assertFalse(any("2.00" in ln for ln in n_lines))
            c3_lines = update["dihe_groups"][("c3", "c3", "c3", "c3")]
            self.assertTrue(any("1.00" in ln for ln in c3_lines))

    def test_merge_dihe_empty_later_iteration_keeps_earlier(self):
        from scission.Merge import _load_fragment_update

        with tempfile.TemporaryDirectory() as td:
            frag_dir = Path(td) / "frag_0001"
            frag_dir.mkdir()
            (frag_dir / "it01.frcmod").write_text(_frcmod(["c3-c3-c3-c3 1 1.00 0.0 1."]))
            (frag_dir / "it02.frcmod").write_text(_frcmod([]))
            update = _load_fragment_update(frag_dir)
            self.assertIn(("c3", "c3", "c3", "c3"), update["dihe_groups"])

    def test_merge_two_fragments_same_scanned_bytype_key(self):
        from scission.Merge import MergeWarning, merge_fragment_frcmods

        def _fit_json(param: str):
            return json.dumps(
                {
                    "params": {param: {"nprim": 1}},
                    "systems": [
                        {
                            "params": {param: {"nprim": 1}},
                            "profiles": [{"plots": [param]}],
                        }
                    ],
                }
            )

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            parent = root / "parent.frcmod"
            parent.write_text(_frcmod(["c3-c3-n4-c3 1 0.50 0.0 1."]))
            out = root / "merged.frcmod"
            frag6 = root / "fragment_6"
            frag8 = root / "fragment_8"
            for frag, pk in ((frag6, "1.10"), (frag8, "2.20")):
                frag.mkdir()
                (frag / "it01.frcmod").write_text(_frcmod([f"c3-c3-n4-c3 1 {pk} 0.0 1."]))
                (frag / "it01.fit.json").write_text(_fit_json("LIG_c3-c3-n4-c3"))
                (frag / "fit_torsions.json").write_text("[]")

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                report = merge_fragment_frcmods(
                    parent_frcmod_path=parent,
                    output_frcmod_path=out,
                    fragment_dirs=[frag6, frag8],
                )
            self.assertTrue(out.is_file())
            self.assertTrue(any(issubclass(w.category, MergeWarning) for w in caught))
            self.assertEqual(len(report["conflicts"]), 1)
            self.assertEqual(report["conflicts"][0]["resolution"], "first_scanned_wins")
            self.assertIn("1.10", out.read_text())
            self.assertNotIn("2.20", out.read_text())

    def test_frcmod_merge_accumulates_iterations(self):
        from scission.Merge import _load_fragment_update

        with tempfile.TemporaryDirectory() as td:
            frag = Path(td)
            (frag / "it01.frcmod").write_text(
                _frcmod(["c3-c3-c3-c3 1 1.00 0.0 1.", "c3-c3-c3-n  1 2.00 0.0 1."])
            )
            (frag / "it02.frcmod").write_text(_frcmod(["c3-c3-c3-n  1 3.50 0.0 1."]))
            update = _load_fragment_update(frag)
            keys = set(update["dihe_groups"].keys())
            self.assertIn(("c3", "c3", "c3", "c3"), keys)
            self.assertIn(("c3", "c3", "c3", "n"), keys)


if __name__ == "__main__":
    unittest.main()
