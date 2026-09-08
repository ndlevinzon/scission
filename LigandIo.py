from __future__ import annotations

import re
from pathlib import Path

from .Models import Atom, Bond, InputBundle, Ligand

_ELEMENTS = {
    "H", "B", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I", "Si",
}


def _infer_element(name: str, atom_type: str) -> str:
    """Guess an element symbol from MOL2 or Amber atom naming.

    Args:
        name: Atom name from the structure file.
        atom_type: Force-field atom type label.

    Returns:
        The inferred element symbol.
    """

    stripped = re.sub(r"[^A-Za-z]", "", atom_type or name)
    if not stripped:
        return "C"
    if len(stripped) >= 2 and stripped[:2].title() in _ELEMENTS:
        return stripped[:2].title()
    return stripped[0].upper()


def parse_mol2(path: Path) -> tuple[str, list[Atom], list[Bond]]:
    """Parse a Tripos MOL2 file into ligand name, atoms, and bonds.

    Args:
        path: Path to the MOL2 file to parse.

    Returns:
        A tuple of ligand name, atom records, and bond records.

    Raises:
        ValueError: If the file does not contain both atoms and bonds.
    """

    text = path.read_text().splitlines()
    section = None
    ligand_name = path.stem
    atoms: list[Atom] = []
    bonds: list[Bond] = []
    for line in text:
        if not line.strip():
            continue
        if line.startswith("@<TRIPOS>"):
            section = line.strip()
            continue
        if section == "@<TRIPOS>MOLECULE":
            ligand_name = line.strip()
            section = "@<TRIPOS>MOLECULE_META"
            continue
        if section == "@<TRIPOS>ATOM":
            parts = line.split()
            atom_index = int(parts[0])
            name = parts[1]
            x, y, z = map(float, parts[2:5])
            atom_type = parts[5]
            residue_id = int(parts[6]) if len(parts) > 6 else 1
            residue_name = parts[7] if len(parts) > 7 else "LIG"
            charge = float(parts[8]) if len(parts) > 8 else 0.0
            atoms.append(
                Atom(
                    index=atom_index,
                    name=name,
                    element=_infer_element(name, atom_type),
                    atom_type=atom_type,
                    charge=charge,
                    coords=(x, y, z),
                    residue_name=residue_name,
                    residue_id=residue_id,
                )
            )
        elif section == "@<TRIPOS>BOND":
            parts = line.split()
            bonds.append(
                Bond(
                    index=int(parts[0]),
                    atom1=int(parts[1]),
                    atom2=int(parts[2]),
                    bond_type=parts[3],
                )
            )
    if not atoms or not bonds:
        raise ValueError(f"Failed to parse atoms/bonds from {path}")
    return ligand_name, atoms, bonds


def parse_lib(path: Path) -> tuple[list[str], dict[str, str]]:
    """Parse Amber ``.lib`` atom names and types while preserving order.

    Args:
        path: Path to the Amber LIB file.

    Returns:
        A tuple of ordered atom names and atom types keyed by name.

    Raises:
        ValueError: If the file does not contain any atom-name records.
    """

    atom_names: list[str] = []
    atom_types: dict[str, str] = {}
    in_atoms = False
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("!entry.") and ".unit.atoms table" in line:
            in_atoms = True
            continue
        if in_atoms and line.startswith("!entry."):
            in_atoms = False
        if in_atoms:
            fields = re.findall(r'"([^"]+)"', line)
            if len(fields) >= 2:
                atom_names.append(fields[0])
                atom_types[fields[0]] = fields[1]
    if not atom_names:
        for raw_line in path.read_text().splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                atom_names.append(parts[0])
                atom_types[parts[0]] = parts[1]
    if not atom_names:
        raise ValueError(f"Failed to parse atom names from {path}")
    return atom_names, atom_types


def load_ligand_from_mol2(mol2_path: Path, ligand_name: str | None = None) -> Ligand:
    """Build a :class:`Ligand` from a MOL2 file alone, for topology-only callers.

    Suitable for consumers that only need atoms and bonds (for example RDKit
    mol construction and central-bond SMARTS matching). The Amber ``.lib`` and
    ``.frcmod`` fields are filled with placeholders derived from the MOL2 and
    must not be relied upon.

    Args:
        mol2_path: Path to the MOL2 file describing the ligand topology.
        ligand_name: Optional override for the ligand name.

    Returns:
        A :class:`Ligand` populated from the MOL2 atoms and bonds.

    Raises:
        ValueError: If the MOL2 file does not contain both atoms and bonds.
    """

    parsed_name, atoms, bonds = parse_mol2(mol2_path)
    lib_atom_names = [atom.name for atom in atoms]
    lib_atom_types = {atom.name: atom.atom_type for atom in atoms}
    return Ligand(
        name=ligand_name or parsed_name,
        atoms=atoms,
        bonds=bonds,
        lib_atom_names=lib_atom_names,
        lib_atom_types=lib_atom_types,
        frcmod_text="",
        mol2_path=mol2_path,
        lib_path=mol2_path,
        frcmod_path=mol2_path,
    )


def load_ligand(bundle: InputBundle) -> Ligand:
    """Load and cross-check the MOL2, LIB, and FRCMOD ligand inputs.

    Args:
        bundle: Paths describing the input ligand files.

    Returns:
        A validated :class:`Ligand` record.

    Raises:
        ValueError: If atom counts or atom ordering differ between the MOL2 and
            LIB inputs.
    """

    ligand_name, atoms, bonds = parse_mol2(bundle.mol2_path)
    lib_atom_names, lib_atom_types = parse_lib(bundle.lib_path)
    if len(atoms) != len(lib_atom_names):
        raise ValueError(
            "mol2/lib atom count mismatch: "
            f"{len(atoms)} atoms in mol2 vs {len(lib_atom_names)} atoms in lib"
        )
    mol2_names = [atom.name for atom in atoms]
    if mol2_names != lib_atom_names:
        raise ValueError(
            "mol2/lib atom order mismatch: "
            f"{mol2_names} vs {lib_atom_names}"
        )
    frcmod_text = bundle.frcmod_path.read_text()
    return Ligand(
        name=bundle.ligand_name or ligand_name,
        atoms=atoms,
        bonds=bonds,
        lib_atom_names=lib_atom_names,
        lib_atom_types=lib_atom_types,
        frcmod_text=frcmod_text,
        mol2_path=bundle.mol2_path,
        lib_path=bundle.lib_path,
        frcmod_path=bundle.frcmod_path,
    )
