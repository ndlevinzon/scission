from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import yaml

from .Merge import discover_fragment_dirs, merge_fragment_frcmods
from .Models import FragmentConfig, InputBundle
from .Pipeline import fragment_ligand


def _load_config(path: Path | None) -> FragmentConfig:
    """Load a YAML config file or return the default configuration.

    Args:
        path: Optional path to a YAML config file.

    Returns:
        The parsed :class:`FragmentConfig`, or the default config when no file
        is supplied.
    """

    if path is None:
        return FragmentConfig()
    payload = yaml.safe_load(path.read_text()) or {}
    return FragmentConfig.from_dict(payload)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the ``scission`` CLI.

    Returns:
        A configured top-level argument parser.
    """

    parser = argparse.ArgumentParser(prog="scission")
    subparsers = parser.add_subparsers(dest="command", required=True)
    fragment = subparsers.add_parser("fragment", help="Generate torsion fragments")
    fragment.add_argument("--mol2", required=True, type=Path)
    fragment.add_argument("--lib", required=True, type=Path)
    fragment.add_argument("--frcmod", required=True, type=Path)
    fragment.add_argument("--outdir", required=True, type=Path)
    fragment.add_argument("--config", type=Path, default=None)
    fragment.add_argument(
        "--nproc",
        type=int,
        default=None,
        help=(
            "Worker budget for torsion screening and per-fragment parmchk2/tleap "
            "writes (default: config nproc or 1)."
        ),
    )
    fragment.add_argument(
        "--acyclic-rotatable-only",
        action="store_true",
        help=(
            "Use the stricter legacy torsion definition and exclude amide-like "
            "acyclic single bonds."
        ),
    )
    fragment.add_argument(
        "--include-bond-smarts",
        action="append",
        default=[],
        help=(
            "SMARTS pattern that can nominate an otherwise excluded central "
            "bond as a torsion target. The pattern must map the central bond "
            "atoms as :1 and :2. May be repeated."
        ),
    )
    fragment.add_argument(
        "--restrict-bond-smarts",
        action="append",
        default=[],
        help=(
            "SMARTS pattern restricting fragmentation to only torsions whose "
            "central bond matches it (an allow-list). The pattern must map the "
            "central bond atoms as :1 and :2. May be repeated. When omitted, "
            "every rotatable torsion is fragmented."
        ),
    )
    pick = subparsers.add_parser(
        "pick-bond",
        help="Interactively pick a central bond and build a restrict SMARTS",
    )
    pick.add_argument("--mol2", required=True, type=Path)
    pick.add_argument(
        "--port",
        type=int,
        default=0,
        help="Port to serve on (0 = auto-pick a free port).",
    )
    pick.add_argument(
        "--host",
        default="127.0.0.1",
        help="Interface to bind (defaults to localhost only).",
    )
    pick.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not open a browser automatically; just print the URL.",
    )
    pick.add_argument(
        "--radius",
        type=int,
        default=1,
        help="Initial environment radius for the generated SMARTS.",
    )
    merge = subparsers.add_parser(
        "merge",
        help="Merge fitted fragment frcmods back into a parent frcmod",
    )
    merge.add_argument("--parent-frcmod", required=True, type=Path)
    merge.add_argument("--out", required=True, type=Path)
    merge.add_argument("--report", type=Path, default=None)
    merge.add_argument(
        "--fragments-root",
        type=Path,
        default=None,
        help="Directory containing fragment subdirectories with itX.frcmod files",
    )
    merge.add_argument(
        "--fragment-dir",
        type=Path,
        action="append",
        default=[],
        help="Specific fragment directory to merge; may be repeated",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CLI entry point and return a process exit code.

    Args:
        argv: Optional command-line argument list to parse.

    Returns:
        Process exit code suitable for ``SystemExit``.
    """

    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "fragment":
        config = _load_config(args.config)
        if args.acyclic_rotatable_only:
            config = replace(config, include_rigid_single_bonds=False)
        if args.include_bond_smarts:
            config = replace(
                config,
                rotatable_bond_smarts=(
                    config.rotatable_bond_smarts + tuple(args.include_bond_smarts)
                ),
            )
        if args.restrict_bond_smarts:
            config = replace(
                config,
                restrict_to_bond_smarts=(
                    config.restrict_to_bond_smarts + tuple(args.restrict_bond_smarts)
                ),
            )
        if args.nproc is not None:
            config = replace(config, nproc=max(1, int(args.nproc)))
        result = fragment_ligand(
            InputBundle(
                mol2_path=args.mol2,
                lib_path=args.lib,
                frcmod_path=args.frcmod,
            ),
            args.outdir,
            config,
        )
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
        return 0
    if args.command == "pick-bond":
        from .PickBond import run_pick_bond

        return run_pick_bond(
            mol2_path=args.mol2,
            host=args.host,
            port=args.port,
            open_browser=not args.no_browser,
            initial_radius=args.radius,
        )
    if args.command == "merge":
        fragment_dirs = [path.resolve() for path in args.fragment_dir]
        if args.fragments_root is not None:
            fragment_dirs.extend(
                path.resolve() for path in discover_fragment_dirs(args.fragments_root)
            )
        deduped_fragment_dirs = list(dict.fromkeys(fragment_dirs))
        if not args.fragment_dir and args.fragments_root is None:
            parser.error(
                "merge requires at least one --fragment-dir or a --fragments-root "
                "containing fragment directories"
            )
        # An empty list here means every fragment was skipped for lack of an
        # itXX.frcmod, which discover_fragment_dirs has already warned about.
        # Still write the output so downstream steps have a frcmod to consume.
        report = merge_fragment_frcmods(
            parent_frcmod_path=args.parent_frcmod,
            output_frcmod_path=args.out,
            fragment_dirs=deduped_fragment_dirs,
            report_path=args.report,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    parser.error(f"Unsupported command: {args.command}")
    return 2
