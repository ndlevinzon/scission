"""Build RDKit molecules from Amber/GAFF topologies for scission.

Amber MOL2 files carry partial charges and explicit hydrogens, but not RDKit
formal charges. Forcing formal charge 0 makes quaternary nitrogen (e.g. detergent
headgroups, GAFF ``n4``) fail sanitization with ``Explicit valence ... N, 4``.

Partial-charge heuristics must not mark sulfonyl/carbonyl oxygen as ``O-``
(double bond, degree 1) or amide nitrogen as ``N-`` (three bonds): RDKit then
logs ``Explicit valence ... is greater than permitted`` on stderr even when
Python later catches the failure.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Sequence

from .Models import Atom, Bond

try:
    from rdkit import Chem
except ImportError:  # pragma: no cover
    Chem = None


def rdkit_bond_type(bond_type: str) -> "Chem.BondType":
    """Translate stored bond labels into RDKit bond types."""

    if Chem is None:  # pragma: no cover
        raise ImportError("RDKit is required")
    normalized = bond_type.lower()
    if normalized in {"1", "1.0", "single", "am"}:
        return Chem.BondType.SINGLE
    if normalized in {"2", "2.0"}:
        return Chem.BondType.DOUBLE
    if normalized in {"3", "3.0"}:
        return Chem.BondType.TRIPLE
    if normalized == "ar":
        return Chem.BondType.AROMATIC
    return Chem.BondType.SINGLE


def _charge_fits_valence(
    element: str,
    degree: int,
    bond_order_sum: float | None,
    charge: int,
) -> bool:
    """True when ``charge`` does not overflow RDKit's permitted explicit valence."""

    valence = float(bond_order_sum) if bond_order_sum is not None else float(degree)
    # RDKit: adding a positive formal charge raises permitted valence
    # (O- = 1, O = 2, O+ = 3; N- = 2, N = 3, N+ = 4).
    if element == "O":
        return valence <= (2 + charge) + 0.1
    if element == "N":
        return valence <= (3 + charge) + 0.1
    if element == "S":
        return valence <= (6 + charge) + 0.1
    if element == "P":
        return valence <= (5 + charge) + 0.1
    return True


def _formal_charge_hint(
    atom_type: str,
    element: str,
    degree: int,
    partial_charge: float,
    bond_order_sum: float | None = None,
) -> int | None:
    """Return a formal charge when the FF type or connectivity is unambiguous.

    Degree-1 oxygen with a negative partial charge is ``O-`` only when the
    bond is single (carboxylate / alkoxide / sulfonate oxide). Sulfonyl and
    carbonyl oxygen are double-bonded (valence 2) and stay formally neutral.
    Nitrogen is ``N-`` only when it is not already trivalent (amide, aniline).
    """

    at = (atom_type or "").lower()
    if at in {"n4", "n+"} or at.startswith("n4"):
        return 1
    if element == "N" and degree >= 4:
        return 1
    if element == "N" and degree == 2 and partial_charge <= -0.4:
        if bond_order_sum is None or bond_order_sum <= 2.1:
            return -1
    if element == "O" and degree == 1 and partial_charge <= -0.4:
        if bond_order_sum is None or bond_order_sum <= 1.1:
            return -1
    if element == "O" and degree >= 3 and partial_charge >= 0.4:
        return 1
    if element == "S" and degree == 1 and partial_charge <= -0.4:
        if bond_order_sum is None or bond_order_sum <= 1.1:
            return -1
    # Near-integer partial charge on heteroatoms (RESP / BCC ionic sites)
    if element in {"N", "O", "S", "P"} and abs(partial_charge) >= 0.5:
        rounded = int(round(partial_charge))
        if rounded != 0 and abs(partial_charge - rounded) < 0.35:
            if _charge_fits_valence(element, degree, bond_order_sum, rounded):
                return rounded
    return None


def infer_formal_charge(
    element: str,
    degree: int,
    *,
    atom_type: str = "",
    partial_charge: float = 0.0,
    bond_order_sum: float | None = None,
) -> int:
    """Infer a formal charge for an Amber/GAFF atom.

    Quaternary nitrogen (four explicit bonds) is treated as ``N+`` even when the
    MOL2 partial charge is fractional (~0.6-1.0), which matches GAFF ``n4``.
    """

    hint = _formal_charge_hint(
        atom_type,
        element,
        degree,
        partial_charge,
        bond_order_sum=bond_order_sum,
    )
    if hint is not None:
        return hint

    if bond_order_sum is None:
        return 0

    valence = int(round(bond_order_sum))
    if element == "N":
        if valence >= 4:
            return 1
        if valence <= 2 and partial_charge < -0.3:
            return -1
        return 0
    if element == "O":
        if valence == 1:
            return -1
        if valence >= 3:
            return 1
        return 0
    return 0


@contextmanager
def _rdkit_app_logs_off() -> Iterator[None]:
    """Mute RDKit C++ ``rdApp`` logs so valence errors do not hit stderr."""

    if Chem is None:  # pragma: no cover
        yield
        return
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.error")
    RDLogger.DisableLog("rdApp.warning")
    try:
        yield
    finally:
        RDLogger.EnableLog("rdApp.error")
        RDLogger.EnableLog("rdApp.warning")


def build_rdkit_mol(
    atoms: Sequence[Atom],
    bonds: Sequence[Bond],
    *,
    sanitize: bool = True,
) -> "Chem.Mol":
    """Build an RDKit molecule from scission atom/bond records.

    Explicit hydrogens from the topology are preserved (``NoImplicit``), and
    formal charges are inferred so quaternary ``N`` sanitizes correctly.

    Args:
        atoms: Atom records (MOL2 order).
        bonds: Bond records with one-based atom indices.
        sanitize: When True, run ``Chem.SanitizeMol``.

    Returns:
        An RDKit molecule.

    Raises:
        ImportError: If RDKit is not installed.
        ValueError: If sanitization fails after charge assignment.
    """

    if Chem is None:
        raise ImportError("RDKit is required to build molecules for scission")

    editable = Chem.RWMol()
    for atom in atoms:
        rd_atom = Chem.Atom(atom.element)
        rd_atom.SetNoImplicit(True)
        rd_atom.SetFormalCharge(0)
        rd_idx = editable.AddAtom(rd_atom)
        editable.GetAtomWithIdx(rd_idx).SetProp("_TriposAtomName", atom.name)
        if atom.atom_type:
            editable.GetAtomWithIdx(rd_idx).SetProp("_TriposAtomType", atom.atom_type)

    aromatic_atoms: set[int] = set()
    for bond in bonds:
        atom1 = bond.atom1 - 1
        atom2 = bond.atom2 - 1
        editable.AddBond(atom1, atom2, rdkit_bond_type(bond.bond_type))
        if bond.bond_type.lower() == "ar":
            aromatic_atoms.add(atom1)
            aromatic_atoms.add(atom2)

    mol = editable.GetMol()
    for atom_idx in aromatic_atoms:
        mol.GetAtomWithIdx(atom_idx).SetIsAromatic(True)
    for bond in mol.GetBonds():
        if bond.GetBondType() == Chem.BondType.AROMATIC:
            bond.SetIsAromatic(True)

    mol.UpdatePropertyCache(strict=False)
    for atom, rd_atom in zip(atoms, mol.GetAtoms()):
        bond_order_sum = sum(b.GetBondTypeAsDouble() for b in rd_atom.GetBonds())
        charge = infer_formal_charge(
            atom.element,
            rd_atom.GetDegree(),
            atom_type=atom.atom_type,
            partial_charge=atom.charge,
            bond_order_sum=bond_order_sum,
        )
        rd_atom.SetFormalCharge(charge)
        rd_atom.SetNoImplicit(True)

    if sanitize:
        try:
            with _rdkit_app_logs_off():
                Chem.SanitizeMol(mol)
        except Exception as exc:
            raise ValueError(
                f"RDKit sanitization failed after formal-charge assignment: {exc}"
            ) from exc
    return mol
