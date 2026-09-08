from __future__ import annotations

import networkx as nx

from .Graph import build_graph, heavy_neighbors, ring_bond_set
from .Models import Ligand, TorsionDefinition

try:
    from rdkit import Chem
except ImportError:  # pragma: no cover
    Chem = None

from .RdkitMol import build_rdkit_mol


def _is_single_bond(bond_type: str) -> bool:
    """Return whether a bond label should count as a single bond.

    Args:
        bond_type: Bond type string from the source topology.

    Returns:
        ``True`` when the bond is treated as a single bond.
    """

    return bond_type.lower() in {"1", "1.0", "am", "single"}


def _is_conjugated_bond(bond_type: str) -> bool:
    """Return whether a bond label implies conjugation.

    Args:
        bond_type: Bond type string from the source topology.

    Returns:
        ``True`` when the bond is treated as conjugated.
    """

    return bond_type.lower() in {"2", "ar", "am"}


def _is_amide_like(graph: nx.Graph, b: int, c: int) -> bool:
    """Detect C-N bonds that should be treated as amide-like and rigid.

    Args:
        graph: Molecular graph containing element and bond annotations.
        b: First atom in the candidate central bond.
        c: Second atom in the candidate central bond.

    Returns:
        ``True`` when the bond matches the local pattern of an amide-like bond.
    """

    atom_b = graph.nodes[b]["atom"]
    atom_c = graph.nodes[c]["atom"]
    pair = {atom_b.element, atom_c.element}
    if pair != {"C", "N"}:
        return False
    carbon = b if atom_b.element == "C" else c
    for nbr in graph.neighbors(carbon):
        if nbr in {b, c}:
            continue
        edge = graph.edges[carbon, nbr]
        nbr_element = graph.nodes[nbr]["atom"].element
        if nbr_element in {"O", "S"} and _is_conjugated_bond(edge["bond_type"]):
            return True
    return False


def _neighbor_rank(graph: nx.Graph, center: int, candidate: int, exclude: int) -> tuple[int, int, int, int]:
    """Rank terminal-neighbor candidates for building a torsion definition.

    Args:
        graph: Molecular graph.
        center: Central atom adjacent to the torsion bond.
        candidate: Neighbor being ranked as a terminal torsion atom.
        exclude: The opposite central torsion atom to ignore.

    Returns:
        A tuple suitable for lexicographic sorting of candidate neighbors.
    """

    atom = graph.nodes[candidate]["atom"]
    heavy_degree = sum(
        1 for nbr in graph.neighbors(candidate)
        if nbr != center and graph.nodes[nbr]["atom"].element != "H"
    )
    rich = sum(
        1 for nbr in graph.neighbors(candidate)
        if nbr not in {center, exclude}
    )
    atomic_rank = {
        "H": 1, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9,
        "P": 15, "S": 16, "Cl": 17, "Br": 35, "I": 53,
    }.get(atom.element, 0)
    return (
        0 if atom.element != "H" else 1,
        -rich,
        -atomic_rank,
        -heavy_degree,
        atom.index,
    )


def _pick_terminal_neighbor(graph: nx.Graph, center: int, other: int) -> int | None:
    """Choose the best heavy-atom neighbor to anchor one end of a torsion.

    Args:
        graph: Molecular graph.
        center: Central torsion atom whose terminal neighbor is being chosen.
        other: The opposite central torsion atom to exclude.

    Returns:
        The chosen terminal neighbor index, or ``None`` if none is available.
    """

    candidates = [
        nbr for nbr in graph.neighbors(center)
        if nbr != other and graph.nodes[nbr]["atom"].element != "H"
    ]
    if not candidates:
        return None
    return sorted(candidates, key=lambda atom_idx: _neighbor_rank(graph, center, atom_idx, other))[0]


# Process-wide compiled SMARTS (pattern -> query + :1/:2 query indices).
_COMPILED_CENTRAL_BOND_SMARTS: dict[str, tuple] = {}


def _build_rdkit_mol(ligand: Ligand) -> "Chem.Mol":
    """Build an RDKit molecule from the in-memory ligand topology (cached)."""

    if Chem is None:
        raise ValueError(
            "RDKit is required when rotatable_bond_smarts are configured."
        )
    cached = getattr(ligand, "_rdkit_mol", None)
    if cached is not None:
        return cached
    mol = build_rdkit_mol(ligand.atoms, ligand.bonds)
    ligand._rdkit_mol = mol
    return mol


def _compiled_central_bond_smarts(pattern: str):
    """Compile and cache a ``:1``/``:2`` central-bond SMARTS query."""

    hit = _COMPILED_CENTRAL_BOND_SMARTS.get(pattern)
    if hit is not None:
        return hit
    if Chem is None:
        raise ValueError("RDKit is required for SMARTS matching.")
    query = Chem.MolFromSmarts(pattern)
    if query is None:
        raise ValueError(f"Invalid rotatable bond SMARTS pattern: {pattern}")
    atom_map_to_query_idx: dict[int, int] = {}
    for atom in query.GetAtoms():
        atom_map_num = atom.GetAtomMapNum()
        if atom_map_num:
            atom_map_to_query_idx[atom_map_num] = atom.GetIdx()
    if set(atom_map_to_query_idx) != {1, 2}:
        raise ValueError(
            "Rotatable bond SMARTS patterns must map the central bond atoms "
            f"as :1 and :2: {pattern}"
        )
    query_bond = query.GetBondBetweenAtoms(
        atom_map_to_query_idx[1],
        atom_map_to_query_idx[2],
    )
    if query_bond is None:
        raise ValueError(
            "Rotatable bond SMARTS mapped atoms :1 and :2 must be directly bonded: "
            f"{pattern}"
        )
    hit = (query, atom_map_to_query_idx[1], atom_map_to_query_idx[2])
    _COMPILED_CENTRAL_BOND_SMARTS[pattern] = hit
    return hit


def _match_rotatable_bond_smarts(
    ligand: Ligand,
    rotatable_bond_smarts: tuple[str, ...],
) -> set[tuple[int, int]]:
    """Return central bonds nominated by SMARTS override patterns."""

    if not rotatable_bond_smarts:
        return set()
    mol = _build_rdkit_mol(ligand)
    matched_bonds: set[tuple[int, int]] = set()
    for pattern in rotatable_bond_smarts:
        query, q1, q2 = _compiled_central_bond_smarts(pattern)
        for match in mol.GetSubstructMatches(query, uniquify=True):
            atom1 = match[q1] + 1
            atom2 = match[q2] + 1
            matched_bonds.add(tuple(sorted((atom1, atom2))))
    return matched_bonds



def match_central_bond_smarts(
    ligand: Ligand,
    bond_smarts: tuple[str, ...],
) -> set[tuple[int, int]]:
    """Return the parent central bonds matched by ``:1``/``:2`` SMARTS patterns.

    Public wrapper around the internal matcher so callers (and the
    fragmentation pipeline's ``restrict_to_bond_smarts`` allow-list) can map
    central-bond SMARTS patterns onto normalized one-based parent atom pairs.
    Each pattern must mark the two central bond atoms with atom-map numbers
    ``:1`` and ``:2``, exactly like ``rotatable_bond_smarts``.

    Args:
        ligand: Parent ligand whose topology is matched against the patterns.
        bond_smarts: SMARTS patterns marking the central bond atoms ``:1``/``:2``.

    Returns:
        A set of normalized ``(low, high)`` one-based parent atom index pairs.
    """

    return _match_rotatable_bond_smarts(ligand, tuple(bond_smarts))


def find_rotatable_bonds(
    ligand: Ligand,
    include_rigid_single_bonds: bool = True,
    rotatable_bond_smarts: tuple[str, ...] = (),
    graph: nx.Graph | None = None,
) -> list[tuple[int, int]]:
    """Identify acyclic single bonds eligible for torsion scans.

    Args:
        ligand: Parent ligand to analyze.
        include_rigid_single_bonds: Whether to include acyclic single bonds
            that would otherwise be excluded as amide-like and treated as
            rigid by default.
        rotatable_bond_smarts: SMARTS override patterns that may nominate
            otherwise excluded bonds as torsion targets.
        graph: Optional prebuilt molecular graph.

    Returns:
        Sorted normalized bond pairs considered rotatable.
    """

    graph = graph if graph is not None else build_graph(ligand)
    cache_key = (bool(include_rigid_single_bonds), tuple(rotatable_bond_smarts))
    cached = getattr(ligand, "_rotatable_bonds_cache", None)
    if cached is not None and cached[0] == cache_key:
        return list(cached[1])

    ring_edges = ring_bond_set(graph, ligand)
    smarts_matched_bonds = _match_rotatable_bond_smarts(ligand, rotatable_bond_smarts)
    rotatable: list[tuple[int, int]] = []
    for bond in ligand.bonds:
        edge = tuple(sorted((bond.atom1, bond.atom2)))
        if edge in ring_edges:
            continue
        passes_default = _is_single_bond(bond.bond_type)
        if passes_default and not include_rigid_single_bonds and _is_amide_like(graph, bond.atom1, bond.atom2):
            passes_default = False
        if not passes_default and edge not in smarts_matched_bonds:
            continue
        if len(heavy_neighbors(graph, bond.atom1)) < 2 or len(heavy_neighbors(graph, bond.atom2)) < 2:
            continue
        rotatable.append(edge)
    rotatable.sort()
    ligand._rotatable_bonds_cache = (cache_key, tuple(rotatable))
    return rotatable


def enumerate_torsions(
    ligand: Ligand,
    include_rigid_single_bonds: bool = True,
    rotatable_bond_smarts: tuple[str, ...] = (),
) -> list[TorsionDefinition]:
    """Build named four-atom torsions for each rotatable bond.

    Args:
        ligand: Parent ligand to analyze.
        include_rigid_single_bonds: Whether to include torsions centered on
            acyclic single bonds that are normally treated as amide-like and
            rigid.
        rotatable_bond_smarts: SMARTS override patterns that may nominate
            otherwise excluded bonds as torsion targets.

    Returns:
        Sorted torsion definitions suitable for fragment generation.
    """

    graph = build_graph(ligand)
    rotatable = set(
        find_rotatable_bonds(
            ligand,
            include_rigid_single_bonds=include_rigid_single_bonds,
            rotatable_bond_smarts=rotatable_bond_smarts,
            graph=graph,
        )
    )
    torsions: list[TorsionDefinition] = []
    for bond in ligand.bonds:
        edge = tuple(sorted((bond.atom1, bond.atom2)))
        if edge not in rotatable:
            continue
        a = _pick_terminal_neighbor(graph, bond.atom1, bond.atom2)
        d = _pick_terminal_neighbor(graph, bond.atom2, bond.atom1)
        if a is None or d is None:
            continue
        torsion_atoms = (a, bond.atom1, bond.atom2, d)
        label = "-".join(ligand.atom(atom_idx).name for atom_idx in torsion_atoms)
        torsions.append(
            TorsionDefinition(
                atom_indices=torsion_atoms,
                bond=edge,
                label=label,
            )
        )
    torsions.sort(key=lambda item: item.atom_indices)
    return torsions
