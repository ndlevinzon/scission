from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np

from .Capping import plan_caps
from .Graph import build_graph
from .Models import CandidateFragment, FragmentConfig, Ligand, SelectedFragment, TorsionDefinition
from .Screen import cap_direction, cap_site_scan_margin

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Draw
except ImportError:  # pragma: no cover
    Chem = None
    AllChem = None
    Draw = None


def safe_name(text: str) -> str:
    """Normalize free-form text into a filesystem-safe slug.

    Args:
        text: Source label to sanitize.

    Returns:
        A filename-safe version of ``text``.
    """

    return re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")


def _atom_name_map(mol: "Chem.Mol") -> dict[str, int]:
    """Map Tripos atom names to RDKit atom indices.

    Args:
        mol: RDKit molecule containing ``_TriposAtomName`` properties.

    Returns:
        Mapping from atom name to RDKit atom index.
    """

    mapping: dict[str, int] = {}
    for atom in mol.GetAtoms():
        name = atom.GetProp("_TriposAtomName") if atom.HasProp("_TriposAtomName") else atom.GetSymbol()
        mapping[name] = atom.GetIdx()
    return mapping


def _load_fragment_rdkit_mol(mol2_path: Path) -> "Chem.Mol | None":
    """Rebuild a fragment MOL2 file as an RDKit molecule for drawing.

    Args:
        mol2_path: Path to the fragment MOL2 file.

    Returns:
        An RDKit molecule when RDKit is available, otherwise ``None``.
    """

    if Chem is None:
        return None

    from .LigandIo import parse_mol2
    from .RdkitMol import build_rdkit_mol

    _, atoms, bonds = parse_mol2(mol2_path)
    return build_rdkit_mol(atoms, bonds)


def _highlight_for_torsion(mol: "Chem.Mol", torsion_label: str) -> tuple[list[int], list[int]]:
    """Locate atoms and bonds to highlight for a named torsion.

    Args:
        mol: RDKit molecule to inspect.
        torsion_label: Torsion label formatted as joined atom names.

    Returns:
        Atom indices and bond indices to highlight in the drawing.
    """

    atom_names = torsion_label.split("-")
    name_to_idx = _atom_name_map(mol)
    atom_ids = [name_to_idx[name] for name in atom_names if name in name_to_idx]
    bond_ids: list[int] = []
    for left, right in zip(atom_names, atom_names[1:]):
        if left not in name_to_idx or right not in name_to_idx:
            continue
        bond = mol.GetBondBetweenAtoms(name_to_idx[left], name_to_idx[right])
        if bond is not None:
            bond_ids.append(bond.GetIdx())
    return atom_ids, bond_ids


def _write_fragment_drawings(
    mol2_path: Path,
    fragment_dir: Path,
    fragment_id: str,
    assigned_torsions: list[str],
    fragment_net_charge: float,
    warnings: list[str] | None,
) -> tuple[Path | None, dict[str, Path]]:
    """Write overview and torsion-specific SVG drawings for a fragment.

    Args:
        mol2_path: Path to the fragment MOL2 file.
        fragment_dir: Output directory for the fragment.
        fragment_id: Human-readable fragment identifier.
        assigned_torsions: Torsion labels covered by the fragment.
        fragment_net_charge: Final fragment net charge used in legends.
        warnings: Optional warning list to append to.

    Returns:
        A tuple of the overview SVG path and a mapping of torsion label to SVG
        path. Paths may be missing when RDKit is unavailable.
    """

    if Chem is None or AllChem is None or Draw is None:
        if warnings is not None:
            warnings.append("RDKit not installed; skipping 2D fragment drawings.")
        return None, {}

    try:
        mol = _load_fragment_rdkit_mol(mol2_path)
        if mol is None:
            if warnings is not None:
                warnings.append(
                    f"Unable to load {mol2_path.name} into RDKit; skipping 2D fragment drawings."
                )
            return None, {}

        AllChem.Compute2DCoords(mol)
        overview_image_path = fragment_dir / "fragment_2d.svg"
        torsion_image_paths: dict[str, Path] = {}

        drawer = Draw.MolDraw2DSVG(900, 600)
        drawer.DrawMolecule(mol, legend=f"{fragment_id} | charge={fragment_net_charge:.3f}")
        drawer.FinishDrawing()
        overview_image_path.write_text(drawer.GetDrawingText())

        for torsion_label in assigned_torsions:
            atom_ids, bond_ids = _highlight_for_torsion(mol, torsion_label)
            drawer = Draw.MolDraw2DSVG(900, 600)
            options = drawer.drawOptions()
            options.addAtomIndices = False
            drawer.DrawMolecule(
                mol,
                highlightAtoms=atom_ids,
                highlightBonds=bond_ids,
                legend=f"{torsion_label} | charge={fragment_net_charge:.3f}",
            )
            drawer.FinishDrawing()
            out_path = fragment_dir / f"torsion_{safe_name(torsion_label)}.svg"
            out_path.write_text(drawer.GetDrawingText())
            torsion_image_paths[torsion_label] = out_path
    except Exception as exc:
        if warnings is not None:
            warnings.append(
                f"Skipping 2D fragment drawings for {mol2_path.name}: {exc}"
            )
        return None, {}

    return overview_image_path, torsion_image_paths


def write_fragment_outputs(
    ligand: Ligand,
    candidate: CandidateFragment,
    fragment_id: str,
    assigned_torsions: list[str],
    torsion_map: dict[str, TorsionDefinition],
    out_dir: Path,
    config: FragmentConfig,
    warnings: list[str] | None = None,
) -> SelectedFragment:
    """Write all on-disk artifacts for one selected fragment.

    Args:
        ligand: Parent ligand record.
        candidate: Selected candidate fragment to materialize.
        fragment_id: Human-friendly fragment identifier to use on disk.
        assigned_torsions: Torsion labels assigned to the fragment.
        torsion_map: Mapping from torsion label to torsion definition.
        out_dir: Parent output directory for the workflow.
        config: Fragmentation configuration used for the run.
        warnings: Optional warning list to append to.

    Returns:
        A :class:`SelectedFragment` populated with emitted artifact paths.
    """

    fragment_dir = out_dir / fragment_id
    fragment_dir.mkdir(parents=True, exist_ok=True)
    retained_atoms = sorted(candidate.retained_atoms)
    parent_atom_map = {parent_idx: local_idx for local_idx, parent_idx in enumerate(retained_atoms, start=1)}
    fit_torsions = _build_fit_torsions(assigned_torsions, torsion_map, parent_atom_map, ligand)
    existing_atom_names = {ligand.atom(parent_idx).name for parent_idx in retained_atoms}
    retained_net_charge = float(sum(ligand.atom(parent_idx).charge for parent_idx in retained_atoms))
    target_integer_charge = round(retained_net_charge)

    graph = build_graph(ligand)
    coordinates = ligand.coordinates

    # Atoms within ``torsion_neighborhood_radius`` bonds of the fitted dihedral:
    # their substituents set the steric/conjugation environment of the rotation,
    # so they must keep a matched cap rather than be stripped to a bare hydrogen.
    protected_atoms: set[int] = set()
    conjugated_atoms: set[int] = set()
    if config.preserve_torsion_neighborhood:
        for label in assigned_torsions:
            torsion = torsion_map.get(label)
            if torsion is not None:
                protected_atoms.update(torsion.atom_indices)
        frontier = set(protected_atoms)
        for _ in range(max(config.torsion_neighborhood_radius, 0)):
            nxt: set[int] = set()
            for atom_idx in frontier:
                nxt.update(graph.neighbors(atom_idx))
            frontier = nxt - protected_atoms
            protected_atoms |= nxt
        if config.preserve_conjugated_caps:
            conjugated_atoms = {
                atom_idx
                for bond in ligand.bonds
                for atom_idx in (bond.atom1, bond.atom2)
                if bond.bond_type.lower() in {"2", "2.0", "ar"}
            }

    def _force_matched(site) -> str | None:
        """Reason a carbon cut must keep a matched cap, or ``None`` if a bare H is fine."""

        if site.retained_atom in protected_atoms:
            return "near_fitted_torsion"
        if site.retained_atom in conjugated_atoms:
            return "conjugated_center"
        return None

    def _steric_rank(site) -> float:
        """Worst rotated-scan margin for a methyl-sized cap at ``site``."""

        margins = [
            cap_site_scan_margin(
                ligand,
                candidate,
                torsion_map[label],
                site.retained_atom,
                site.removed_atom,
                "C",
                config.angle_step,
                config.clash_thresholds,
            )
            for label in assigned_torsions
            if label in torsion_map
        ]
        return min(margins) if margins else float("inf")

    resolved_caps, fragment_net_charge = plan_caps(
        candidate.cap_sites,
        element_of=lambda idx: ligand.atom(idx).element,
        position_of=lambda idx: ligand.atom(idx).coords,
        direction_of=lambda retained, removed: cap_direction(graph, ligand, retained, removed, coordinates),
        strategy=config.cap_strategy,
        retained_net_charge=retained_net_charge,
        existing_names=existing_atom_names,
        h_min_charge=config.cap_h_min_charge,
        steric_rank_of=_steric_rank,
        force_matched_of=_force_matched,
    )
    cap_decisions = [
        {
            "parent_atom": cap.retained_atom,
            "removed_atom": cap.removed_atom,
            "bond_order": cap.parent_bond_order,
            "cap": cap.cap_label,
            "reason": cap.reason,
            "bare_h_charge": cap.bare_h_charge,
            "h_min_charge": config.cap_h_min_charge,
        }
        for cap in resolved_caps
    ]

    # Assign a fragment-local index to every cap atom (heavy atom first).
    cap_local_index: dict[int, int] = {}
    next_atom_index = len(retained_atoms) + 1
    for cap in resolved_caps:
        for cap_atom in cap.atoms:
            cap_local_index[id(cap_atom)] = next_atom_index
            next_atom_index += 1
    total_cap_atoms = len(cap_local_index)
    fragment_bonds = _fragment_bonds(ligand, candidate)

    mol2_lines = [
        "@<TRIPOS>MOLECULE",
        ligand.name,
        f"{len(retained_atoms) + total_cap_atoms} {len(fragment_bonds) + total_cap_atoms} 1 0 1",
        "SMALL",
        "USER_CHARGES",
        "",
        "@<TRIPOS>ATOM",
    ]
    for parent_idx in retained_atoms:
        atom = ligand.atom(parent_idx)
        x, y, z = atom.coords
        mol2_lines.append(
            f"{parent_atom_map[parent_idx]:>7} {atom.name:<8} {x:>9.4f} {y:>9.4f} {z:>9.4f} "
            f"{atom.atom_type:<6} {atom.residue_id:>3} {atom.residue_name:<8} {atom.charge:>8.4f}"
        )
    cap_atoms: list[dict[str, object]] = []
    for cap in resolved_caps:
        for cap_atom in cap.atoms:
            local = cap_local_index[id(cap_atom)]
            coords = np.asarray(cap_atom.coords, dtype=float)
            mol2_lines.append(
                f"{local:>7} {cap_atom.name:<8} {coords[0]:>9.4f} {coords[1]:>9.4f} {coords[2]:>9.4f} "
                f"{cap_atom.atom_type:<6} {1:>3} {'LIG':<8} {cap_atom.charge:>8.4f}"
            )
            cap_atoms.append(
                {
                    "name": cap_atom.name,
                    "element": cap_atom.element,
                    "parent_atom": cap.retained_atom,
                    "removed_atom": cap.removed_atom,
                    "coords": coords.tolist(),
                    "charge": cap_atom.charge,
                    "atom_type": cap_atom.atom_type,
                    "role": cap_atom.role,
                }
            )

    mol2_lines.extend(["@<TRIPOS>BOND"])
    bond_index = 1
    for atom1, atom2, bond_type in fragment_bonds:
        mol2_lines.append(
            f"{bond_index:>6} {atom1:>4} {atom2:>4} {bond_type}"
        )
        bond_index += 1
    for cap in resolved_caps:
        parent_local = parent_atom_map[cap.retained_atom]
        if cap.heavy is not None:
            heavy_local = cap_local_index[id(cap.heavy)]
            mol2_lines.append(f"{bond_index:>6} {parent_local:>4} {heavy_local:>4} {cap.parent_bond_order}")
            bond_index += 1
            for hydrogen in cap.hydrogens:
                hydrogen_local = cap_local_index[id(hydrogen)]
                mol2_lines.append(f"{bond_index:>6} {heavy_local:>4} {hydrogen_local:>4} 1")
                bond_index += 1
        else:
            hydrogen_local = cap_local_index[id(cap.hydrogens[0])]
            mol2_lines.append(f"{bond_index:>6} {parent_local:>4} {hydrogen_local:>4} {cap.parent_bond_order}")
            bond_index += 1
    mol2_lines.extend(
        [
            "@<TRIPOS>SUBSTRUCTURE",
            f"{1:>7} {'LIG':<8} {1:>7} {'RESIDUE':<10} {0:>4} {'****':<4} {'ROOT':<9} {0:>4}",
        ]
    )
    mol2_path = fragment_dir / "fragment.mol2"
    mol2_path.write_text("\n".join(mol2_lines) + "\n")

    xyz_lines = [
        str(len(retained_atoms) + len(cap_atoms)),
        f"{ligand.name} | charge={fragment_net_charge:.4f} | torsions={','.join(assigned_torsions)}",
    ]
    for parent_idx in retained_atoms:
        atom = ligand.atom(parent_idx)
        x, y, z = atom.coords
        xyz_lines.append(f"{atom.element:<2} {x:>12.6f} {y:>12.6f} {z:>12.6f}")
    for cap in cap_atoms:
        x, y, z = cap["coords"]
        xyz_lines.append(f"{cap['element']:<2} {x:>12.6f} {y:>12.6f} {z:>12.6f}")
    xyz_path = fragment_dir / "fragment.xyz"
    xyz_path.write_text("\n".join(xyz_lines) + "\n")

    lib_lines = [f"!entry.{ligand.name}.unit.atoms table  str name  str type"]
    for parent_idx in retained_atoms:
        atom = ligand.atom(parent_idx)
        lib_lines.append(f'"{atom.name}" "{atom.atom_type}"')
    for cap in cap_atoms:
        lib_lines.append(f'"{cap["name"]}" "{cap["atom_type"]}"')
    lib_path = fragment_dir / "fragment.lib"
    lib_path.write_text("\n".join(lib_lines) + "\n")

    frcmod_path = fragment_dir / "fragment.frcmod"
    shutil.copyfile(ligand.frcmod_path, frcmod_path)

    parm7_path, rst7_path = _write_amber_outputs(fragment_dir, warnings)
    overview_image_path, torsion_image_paths = _write_fragment_drawings(
        mol2_path,
        fragment_dir,
        fragment_id,
        assigned_torsions,
        fragment_net_charge,
        warnings,
    )

    manifest = {
        "fragment_id": fragment_id,
        "source_candidate_id": candidate.candidate_id,
        "parent_name": ligand.name,
        "retained_atoms": retained_atoms,
        "cut_bonds": [list(bond) for bond in sorted(candidate.cut_bonds)],
        "cap_atoms": cap_atoms,
        "cap_decisions": cap_decisions,
        "torsions": assigned_torsions,
        "fit_torsions": fit_torsions,
        "net_charge": fragment_net_charge,
        "retained_net_charge": retained_net_charge,
        "target_integer_charge": target_integer_charge,
        "is_parent_fallback": candidate.is_parent_fallback,
        "thresholds": {
            "angle_step": config.angle_step,
            "path3_scale": config.clash_thresholds.path3_scale,
            "far_scale": config.clash_thresholds.far_scale,
        },
        "parent_to_fragment_atom_map": parent_atom_map,
        "overview_image_path": str(overview_image_path) if overview_image_path is not None else None,
        "torsion_image_paths": {
            label: str(path) for label, path in torsion_image_paths.items()
        },
        "charge_type_caveats": [
            "Retained atoms preserve parent charges/types.",
            f"Caps were generated with the '{config.cap_strategy}' strategy.",
            "Chemistry-aware caps prefer a bare hydrogen and fall back to an "
            "element-matched group (e.g. CH3/OH/NH2) when the cap hydrogen "
            "would carry a negative/low charge or a heteroatom was severed.",
            "The charge correction needed to reach an integer fragment charge "
            "is carried by the bare-hydrogen caps, or, when none remain, "
            "distributed across the matched caps with most of the adjustment on "
            "the heavy atom.",
        ],
        "worst_margin": candidate.worst_margin,
    }
    manifest_path = fragment_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    fit_torsions_path = fragment_dir / "fit_torsions.json"
    fit_torsions_path.write_text(json.dumps(fit_torsions, indent=2, sort_keys=True) + "\n")

    return SelectedFragment(
        fragment_id=fragment_id,
        source_candidate_id=candidate.candidate_id,
        retained_atoms=retained_atoms,
        cut_bonds=sorted(candidate.cut_bonds),
        cap_atoms=cap_atoms,
        cap_decisions=cap_decisions,
        torsions=assigned_torsions,
        fit_torsions=fit_torsions,
        parent_atom_map=parent_atom_map,
        net_charge=fragment_net_charge,
        retained_net_charge=retained_net_charge,
        is_parent_fallback=candidate.is_parent_fallback,
        manifest_path=manifest_path,
        mol2_path=mol2_path,
        xyz_path=xyz_path,
        lib_path=lib_path,
        frcmod_path=frcmod_path,
        parm7_path=parm7_path,
        rst7_path=rst7_path,
        overview_image_path=overview_image_path,
        torsion_image_paths=torsion_image_paths,
        worst_margin=candidate.worst_margin,
    )


def write_fragment_index(selected_fragments: list[SelectedFragment], out_dir: Path) -> Path:
    """Write a compact top-level mapping for human-friendly fragment names.

    Args:
        selected_fragments: Materialized selected fragments.
        out_dir: Workflow output directory.

    Returns:
        Path to the written fragment index JSON file.
    """

    fragment_index = {
        "fragments": [
            {
                "fragment_id": fragment.fragment_id,
                "source_candidate_id": fragment.source_candidate_id,
                "directory": str(fragment.manifest_path.parent) if fragment.manifest_path else None,
                "manifest_path": str(fragment.manifest_path) if fragment.manifest_path else None,
                "retained_atoms": fragment.retained_atoms,
                "cut_bonds": [list(bond) for bond in fragment.cut_bonds],
                "torsions": fragment.torsions,
                "is_parent_fallback": fragment.is_parent_fallback,
            }
            for fragment in selected_fragments
        ]
    }
    path = out_dir / "fragment_index.json"
    path.write_text(json.dumps(fragment_index, indent=2, sort_keys=True) + "\n")
    return path


def write_summary(
    ligand: Ligand,
    selected_fragments: list[SelectedFragment],
    covered_torsions: dict[str, str],
    rejected_torsions: dict[str, str],
    warnings: list[str],
    out_dir: Path,
) -> Path:
    """Write the run-level summary JSON file.

    Args:
        ligand: Parent ligand record.
        selected_fragments: Materialized selected fragments.
        covered_torsions: Mapping from torsion label to fragment identifier.
        rejected_torsions: Mapping from torsion label to rejection details.
        warnings: Non-fatal warnings accumulated during the run.
        out_dir: Workflow output directory.

    Returns:
        Path to the written summary JSON file.
    """

    cap_decision_counts: dict[str, int] = {}
    for fragment in selected_fragments:
        for decision in fragment.cap_decisions:
            reason = decision["reason"]
            cap_decision_counts[reason] = cap_decision_counts.get(reason, 0) + 1

    summary = {
        "parent": {
            "name": ligand.name,
            "mol2_path": str(ligand.mol2_path),
            "lib_path": str(ligand.lib_path),
            "frcmod_path": str(ligand.frcmod_path),
        },
        "selected_fragments": [fragment.to_dict() for fragment in selected_fragments],
        "covered_torsions": covered_torsions,
        "rejected_torsions": rejected_torsions,
        "cap_decision_counts": cap_decision_counts,
        "warnings": warnings,
    }
    path = out_dir / "summary.json"
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return path


def _fragment_bonds(ligand: Ligand, candidate: CandidateFragment) -> list[tuple[int, int, str]]:
    """Project retained parent bonds into fragment-local atom numbering.

    Args:
        ligand: Parent ligand record.
        candidate: Candidate fragment whose retained atoms should be projected.

    Returns:
        Fragment-local bond tuples of ``(atom1, atom2, bond_type)``.
    """

    retained = candidate.retained_atoms
    parent_to_local = {parent_idx: local_idx for local_idx, parent_idx in enumerate(sorted(retained), start=1)}
    bonds: list[tuple[int, int, str]] = []
    for bond in ligand.bonds:
        if bond.atom1 in retained and bond.atom2 in retained:
            bonds.append((parent_to_local[bond.atom1], parent_to_local[bond.atom2], bond.bond_type))
    return bonds


def _build_fit_torsions(
    assigned_torsions: list[str],
    torsion_map: dict[str, TorsionDefinition],
    parent_atom_map: dict[int, int],
    ligand: Ligand,
) -> list[dict[str, object]]:
    """Build the serialized torsion mapping metadata for a fragment.

    Args:
        assigned_torsions: Torsion labels assigned to the fragment.
        torsion_map: Mapping from torsion label to torsion definition.
        parent_atom_map: Mapping from parent atom index to fragment-local index.
        ligand: Parent ligand record.

    Returns:
        Serialized torsion records describing parent-to-fragment mappings.
    """

    fit_torsions: list[dict[str, object]] = []
    for torsion_label in assigned_torsions:
        torsion = torsion_map[torsion_label]
        a, b, c, d = torsion.atom_indices
        fit_torsions.append(
            {
                "label": torsion.label,
                "parent_dihedral_atoms": [a, b, c, d],
                "fragment_dihedral_atoms": [
                    parent_atom_map[a],
                    parent_atom_map[b],
                    parent_atom_map[c],
                    parent_atom_map[d],
                ],
                "parent_rotatable_bond": list(torsion.bond),
                "fragment_rotatable_bond": [
                    parent_atom_map[torsion.bond[0]],
                    parent_atom_map[torsion.bond[1]],
                ],
                "parent_rotatable_bond_atom_names": [
                    ligand.atom(torsion.bond[0]).name,
                    ligand.atom(torsion.bond[1]).name,
                ],
                "fragment_rotatable_bond_atom_names": [
                    ligand.atom(torsion.bond[0]).name,
                    ligand.atom(torsion.bond[1]).name,
                ],
            }
        )
    return fit_torsions


def _write_amber_outputs(
    fragment_dir: Path,
    warnings: list[str] | None = None,
) -> tuple[Path | None, Path | None]:
    """Run ``parmchk2`` and ``tleap`` to build Amber topology files.

    Args:
        fragment_dir: Directory containing the fragment inputs.
        warnings: Optional warning list to append to.

    Returns:
        A tuple of ``(parm7_path, rst7_path)`` when generation succeeds, or
        ``(None, None)`` otherwise.
    """

    tleap = shutil.which("tleap")
    if tleap is None:
        if warnings is not None:
            warnings.append(
                f"tleap not found in PATH; skipping parm7/rst7 generation for {fragment_dir.name}."
            )
        return None, None

    parmchk2 = shutil.which("parmchk2")
    auto_frcmod = fragment_dir / "fragment.auto.frcmod"
    if parmchk2 is not None:
        parmchk = subprocess.run(
            [
                parmchk2,
                "-i",
                "fragment.mol2",
                "-f",
                "mol2",
                "-s",
                "gaff2",
                "-o",
                auto_frcmod.name,
            ],
            cwd=fragment_dir,
            text=True,
            capture_output=True,
        )
        (fragment_dir / "parmchk2.stdout.log").write_text(parmchk.stdout)
        (fragment_dir / "parmchk2.stderr.log").write_text(parmchk.stderr)
        if parmchk.returncode != 0:
            if warnings is not None:
                warnings.append(
                    f"parmchk2 failed for {fragment_dir.name}; tleap will use only fragment.frcmod."
                )
            auto_frcmod = None
    elif warnings is not None:
        warnings.append(
            f"parmchk2 not found in PATH; using only fragment.frcmod for {fragment_dir.name}."
        )
        auto_frcmod = None

    leaprc_path = _discover_leaprc_path(tleap)
    leap_lines = []
    if leaprc_path is not None:
        leap_lines.append(f"source {leaprc_path}")
    else:
        leap_lines.append("source leaprc.gaff2")
    leap_lines.append("loadamberparams fragment.frcmod")
    if auto_frcmod is not None and auto_frcmod.exists():
        leap_lines.append(f"loadamberparams {auto_frcmod.name}")
    leap_lines.extend(
        [
            "frag = loadmol2 fragment.mol2",
            "check frag",
            "saveamberparm frag fragment.parm7 fragment.rst7",
            "quit",
        ]
    )
    leap_input = fragment_dir / "tleap.in"
    leap_input.write_text("\n".join(leap_lines) + "\n")

    tleap_run = subprocess.run(
        [tleap, "-f", leap_input.name],
        cwd=fragment_dir,
        text=True,
        capture_output=True,
    )
    (fragment_dir / "tleap.stdout.log").write_text(tleap_run.stdout)
    (fragment_dir / "tleap.stderr.log").write_text(tleap_run.stderr)
    parm7_path = fragment_dir / "fragment.parm7"
    rst7_path = fragment_dir / "fragment.rst7"
    if tleap_run.returncode != 0 or not parm7_path.exists() or not rst7_path.exists():
        if warnings is not None:
            warnings.append(
                f"tleap failed to generate parm7/rst7 for {fragment_dir.name}; see tleap.stdout.log."
            )
        return None, None
    return parm7_path, rst7_path


def _discover_leaprc_path(tleap_path: str) -> str | None:
    """Infer the bundled ``leaprc.gaff2`` path next to a ``tleap`` install.

    Args:
        tleap_path: Resolved path to the ``tleap`` executable.

    Returns:
        Path to ``leaprc.gaff2`` when it can be inferred, otherwise ``None``.
    """

    prefix = Path(tleap_path).resolve().parent.parent
    candidate = prefix / "dat" / "leap" / "cmd" / "leaprc.gaff2"
    if candidate.exists():
        return str(candidate)
    return None
