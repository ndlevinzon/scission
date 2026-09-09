"""Install validation for standalone scission.

Run after ``pip install -e .`` (or from this checkout)::

    python -m unittest tests.test_install_validation -v
"""

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import sys
import unittest
from pathlib import Path

_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))
import _paths  # noqa: E402


def setUpModule():
    _paths.ensure_scission()


class TestCorePackageInstall(unittest.TestCase):
    def test_version_and_public_api(self):
        import scission

        self.assertTrue(scission.__version__)
        self.assertRegex(scission.__version__, r"^\d+\.\d+")
        self.assertFalse(getattr(scission, "__ligandparam_bundle__", False))
        for name in (
            "fragment_ligand",
            "FragmentConfig",
            "SelectedFragment",
            "match_central_bond_smarts",
            "register_strategy",
        ):
            self.assertTrue(hasattr(scission, name), name)

    def test_merge_and_writers(self):
        from scission.Frcmod import _normalize_param_name_to_key
        from scission.Writers import safe_name

        self.assertEqual(safe_name("a/b c"), "a_b_c")
        key = _normalize_param_name_to_key("LIG_ca-c3-c-o")
        self.assertIsNotNone(key)
        self.assertEqual(len(key), 4)

        merge = importlib.import_module("scission.Merge")
        self.assertTrue(callable(merge.merge_fragment_frcmods))
        self.assertTrue(callable(merge.list_iteration_frcmods))

    def test_cli_entrypoint(self):
        from scission.Cli import main

        self.assertTrue(callable(main))

    def test_installed_console_script_when_available(self):
        try:
            dist = importlib.metadata.distribution("scission")
        except importlib.metadata.PackageNotFoundError:
            self.skipTest("scission distribution metadata unavailable")
        ep_names = {ep.name for ep in dist.entry_points if ep.group == "console_scripts"}
        self.assertIn("scission", ep_names)
        self.assertNotIn("lig-scission", ep_names)


class TestIsolation(unittest.TestCase):
    def test_source_does_not_import_alps_ffpopt_or_ligandparam(self):
        import ast

        root = _paths.package_root()
        hits: list[str] = []
        banned = frozenset({"alps", "ffpopt", "ligandparam"})
        for fp in root.rglob("*.py"):
            if "__pycache__" in fp.parts or "tests" in fp.parts:
                continue
            text = fp.read_text(encoding="utf-8", errors="replace")
            try:
                tree = ast.parse(text, filename=str(fp))
            except SyntaxError:
                continue
            rel = fp.relative_to(root)
            for node in ast.walk(tree):
                names: list[str] = []
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names = [node.module]
                for name in names:
                    if name.split(".")[0] in banned:
                        hits.append(f"{rel}:{node.lineno} {name}")
        self.assertEqual(hits, [], "scission imports companions:\n" + "\n".join(hits))


if __name__ == "__main__":
    unittest.main()
