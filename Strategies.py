"""Named fragmentation strategies built around a rotatable bond.

The default ``scission`` scheme is the existing rigid-domain shell enumerator.
``pfizer`` and ``wbo`` follow the protocols described in:

    Stern CD, Bayly CI, Smith DGA, Fass J, Wang L-P, Mobley DL, Chodera JD.
    Capturing non-local through-bond effects in molecular mechanics force
    fields I: Fragmenting molecules for quantum chemical torsion scans.
    bioRxiv 2020.08.27.270934v2.
    https://www.biorxiv.org/content/10.1101/2020.08.27.270934v2
    doi:10.1101/2020.08.27.270934

Pfizer (Rai et al., J. Chem. Inf. Model. 2019, doi:10.1021/acs.jcim.9b00373)
keeps the torsion quartet, fused ring systems, listed functional groups, and
ortho ring substituents, then methyl-caps N/O/S open valences.

WBO / FBO grows that seed one substituent at a time along shortest path from
the target bond (the paper's default ``path_length`` heuristic). AM1 ELF10
Wiberg orders are **not** computed here — that would add OpenFF / extra QM
engines. Clash screening plus fragment selection still pick the smallest
passing candidate. If ``wbo_max_growth`` is set, growth stops after that many
additions.

Register extra schemes with :func:`register_strategy`. They must return
:class:`~scission.Models.CandidateFragment` objects so capping, screening,
and Amber writes stay unchanged.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING

import networkx as nx

from .Fragments import build_candidate_fragments, candidate_from_retained_atoms
from .Graph import build_graph, ring_bond_set, sorted_cut_bond
from .Models import CandidateFragment, FragmentConfig, Ligand, TorsionDefinition
from .Torsions import find_rotatable_bonds

if TYPE_CHECKING:
    from rdkit import Chem as _Chem

try:
    from rdkit import Chem
except ImportError:  # pragma: no cover
    Chem = None

from .RdkitMol import build_rdkit_mol

# Paper Table 1 / OpenFF fragmenter defaults: groups with >1 heteroatom
# (or equivalent) that should not be split. SMARTS are RDKit-compatible.
# Source: Stern et al., bioRxiv 2020.08.27.270934v2.
DEFAULT_FUNCTIONAL_GROUPS: dict[str, str] = {
    "hydrazine": "[NX3:1][NX3:2]",
    "hydrazone": "[NX3:1][NX2:2]",
    "nitrile": "[NX1:1]#[CX2:2]",
    "enamine": "[NX3:1][#6:2]=[#6:3]",
    "ketone": "[#6X3:1][#6:2](=[#8:3])",
    "amide": "[#7:1][#6:2](=[#8:3])",
    "urea": "[#7:1][#6:2](=[#8:3])[#7:4]",
    "alcohol": "[#6:1]-[#8X2H1:2]",
    "sulfoxide": "[#16X3:1]=[OX1:2]",
    "sulfonyl": "[#16X4:1](=[OX1:2])=[OX1:3]",
    "sulfinic_acid": "[#16X3:1](=[OX1:2])[OX2H,OX1H0-:3]",
    "sulfonamide": "[#16X4:1](=[OX1:2])(=[OX1:3])[NX3:4]",
    "sulfonic_acid": "[#16X4:1](=[OX1:2])(=[OX1:3])[OX2H,OX1H0-:4]",
    "phosphine_oxide": "[PX4:1](=[OX1:2])([#6:3])([#6:4])([#6:5])",
    "phosphonate": "[P:1](=[OX1:2])([OX2H,OX1-:3])([OX2H,OX1-:4])",
    "phosphate": "[PX4:1](=[OX1:2])([#8:3])([#8:4])([#8:5])",
    "carboxylic_acid": "[CX3:1](=[O:2])[OX1H0-,OX2H1:3]",
    "nitro": "[NX3:1](=[O:2])=[O:3]",
    "nitro_charged": "[NX3+:1](=[O:2])[O-:3]",
    "ester": "[CX3:1](=[O:2])[OX2H0:3]",
    "tri_halide": "[#6:1]([F,Cl,I,Br:2])([F,Cl,I,Br:3])([F,Cl,I,Br:4])",
}

_ORTHO_SMARTS = "[!#1:1]~&!@[*:2]@[*:3]~&!@[!#1*:4]"

StrategyFn = Callable[[Ligand, TorsionDefinition, FragmentConfig], list[CandidateFragment]]

_REGISTRY: dict[str, StrategyFn] = {}

STRATEGY_SCISSION = "scission"
STRATEGY_PFIZER = "pfizer"
STRATEGY_WBO = "wbo"


def register_strategy(name: str) -> Callable[[StrategyFn], StrategyFn]:
    """Register ``fn(ligand, torsion, config) -> list[CandidateFragment]``.

    The name is what ``FragmentConfig.strategy`` / ``--strategy`` accept.
    Re-registering the same name replaces the previous function.
    """

    key = str(name).strip().lower()
    if not key:
        raise ValueError("strategy name must be non-empty")

    def decorator(fn: StrategyFn) -> StrategyFn:
        _REGISTRY[key] = fn
        return fn

    return decorator


def available_strategies() -> tuple[str, ...]:
    """Return registered strategy names in sorted order."""

    return tuple(sorted(_REGISTRY))


def resolve_strategy(name: str) -> StrategyFn:
    """Look up a registered strategy by name."""

    key = str(name).strip().lower()
    fn = _REGISTRY.get(key)
    if fn is None:
        known = ", ".join(available_strategies()) or "(none)"
        raise ValueError(f"Unknown fragmentation strategy {name!r}. Known: {known}")
    return fn


def build_candidates(
    ligand: Ligand,
    torsion: TorsionDefinition,
    config: FragmentConfig,
) -> list[CandidateFragment]:
    """Dispatch candidate enumeration for one torsion."""

    return resolve_strategy(config.strategy)(ligand, torsion, config)


def functional_groups_for_config(config: FragmentConfig) -> dict[str, str]:
    """Return SMARTS groups to keep intact (paper defaults when unset)."""

    if config.functional_groups is None:
        return dict(DEFAULT_FUNCTIONAL_GROUPS)
    return dict(config.functional_groups)


@register_strategy(STRATEGY_SCISSION)
def scission_strategy(
    ligand: Ligand,
    torsion: TorsionDefinition,
    config: FragmentConfig,
) -> list[CandidateFragment]:
    """Existing scission rigid-domain shells (backwards-compatible default)."""

    return build_candidate_fragments(
        ligand,
        torsion,
        include_rigid_single_bonds=config.include_rigid_single_bonds,
        rotatable_bond_smarts=config.rotatable_bond_smarts,
    )


@register_strategy(STRATEGY_PFIZER)
def pfizer_strategy(
    ligand: Ligand,
    torsion: TorsionDefinition,
    config: FragmentConfig,
) -> list[CandidateFragment]:
    """One fragment per torsion using the Pfizer / Rai protocol."""

    atoms, bonds = _pfizer_atom_bond_set(ligand, torsion, config)
    candidate = candidate_from_retained_atoms(ligand, torsion, atoms, shell_level=0)
    return [candidate]


@register_strategy(STRATEGY_WBO)
def wbo_strategy(
    ligand: Ligand,
    torsion: TorsionDefinition,
    config: FragmentConfig,
) -> list[CandidateFragment]:
    """Grow the Pfizer seed along shortest path from the target bond.

    Stern et al. stop when fragment WBO matches the parent within
    ``wbo_threshold``. This port emits each growth step as a candidate and
    lets scission's clash screen pick; set ``wbo_max_growth`` to cap steps.
    """

    atoms, bonds = _pfizer_atom_bond_set(ligand, torsion, config)
    groups = _find_functional_groups(ligand, functional_groups_for_config(config))
    rings = _ring_systems(ligand)
    if config.keep_non_rotor_ring_substituents:
        atoms, bonds = _add_non_rotor_ring_substituents(ligand, atoms, bonds, rings)

    candidates_by_id: dict[str, CandidateFragment] = {}
    seed = candidate_from_retained_atoms(ligand, torsion, atoms, shell_level=0)
    candidates_by_id[seed.candidate_id] = seed

    max_growth = config.wbo_max_growth
    step = 0
    while True:
        if max_growth is not None and step >= max_growth:
            break
        grown = _add_next_substituent(ligand, torsion, atoms, bonds, groups, rings)
        if grown is None:
            break
        atoms, bonds = grown
        step += 1
        cand = candidate_from_retained_atoms(
            ligand, torsion, atoms, shell_level=step
        )
        candidates_by_id.setdefault(cand.candidate_id, cand)

    parent_atoms = {atom.index for atom in ligand.atoms}
    if atoms < parent_atoms:
        fallback = candidate_from_retained_atoms(
            ligand,
            torsion,
            parent_atoms,
            shell_level=step + 1,
            is_parent_fallback=True,
        )
        candidates_by_id.setdefault(fallback.candidate_id, fallback)

    return sorted(
        candidates_by_id.values(),
        key=lambda candidate: (
            len(candidate.retained_atoms),
            candidate.shell_level,
            candidate.candidate_id,
        ),
    )


def _pfizer_atom_bond_set(
    ligand: Ligand,
    torsion: TorsionDefinition,
    config: FragmentConfig,
) -> tuple[set[int], set[tuple[int, int]]]:
    """Torsion quartet + rings + functional groups + ortho + N/O/S methyl cap."""

    atoms, bonds = _torsion_quartet(ligand, torsion.bond)
    groups = _find_functional_groups(ligand, functional_groups_for_config(config))
    rings = _ring_systems(ligand)
    atoms, bonds = _include_rings_and_groups(atoms, bonds, groups, rings)
    atoms, bonds = _include_ortho_substituents(ligand, atoms, bonds)
    atoms, bonds = _include_rings_and_groups(atoms, bonds, groups, rings)
    if config.keep_non_rotor_ring_substituents:
        atoms, bonds = _add_non_rotor_ring_substituents(ligand, atoms, bonds, rings)
    atoms, bonds = _cap_open_valence(ligand, atoms, bonds, groups)
    return atoms, bonds


def _ligand_rdkit(ligand: Ligand) -> "_Chem.Mol | None":
    if Chem is None:
        return None
    cached = getattr(ligand, "_rdkit_mol", None)
    if cached is not None:
        return cached
    mol = build_rdkit_mol(ligand.atoms, ligand.bonds, sanitize=True)
    ligand._rdkit_mol = mol
    return mol


def _torsion_quartet(
    ligand: Ligand, bond: tuple[int, int]
) -> tuple[set[int], set[tuple[int, int]]]:
    """Bond atoms plus two neighbor hops (Stern / OpenFF torsion quartet)."""

    graph = build_graph(ligand)
    atoms = {bond[0], bond[1]}
    bonds: set[tuple[int, int]] = {sorted_cut_bond(*bond)}
    for atom_idx in (bond[0], bond[1]):
        for nbr in graph.neighbors(atom_idx):
            atoms.add(nbr)
            bonds.add(sorted_cut_bond(atom_idx, nbr))
            for nxt in graph.neighbors(nbr):
                atoms.add(nxt)
                bonds.add(sorted_cut_bond(nbr, nxt))
    return atoms, bonds


def _find_functional_groups(
    ligand: Ligand, smarts_map: Mapping[str, str]
) -> dict[str, tuple[set[int], set[tuple[int, int]]]]:
    """Match functional-group SMARTS; keys are ``name_i`` like the paper code."""

    mol = _ligand_rdkit(ligand)
    found: dict[str, tuple[set[int], set[tuple[int, int]]]] = {}
    if mol is None or Chem is None:
        return found
    graph = build_graph(ligand)
    for name, smarts in smarts_map.items():
        query = Chem.MolFromSmarts(smarts)
        if query is None:
            continue
        unique = {tuple(sorted(match)) for match in mol.GetSubstructMatches(query)}
        for i, match in enumerate(unique):
            atoms = {idx + 1 for idx in match}
            bonds = {
                sorted_cut_bond(a, b)
                for a, b in graph.edges
                if a in atoms and b in atoms
            }
            found[f"{name}_{i}"] = (atoms, bonds)
    return found


def _ring_systems(ligand: Ligand) -> list[tuple[set[int], set[tuple[int, int]]]]:
    """Fused-ring systems as connected components of ring edges."""

    graph = build_graph(ligand)
    ring_edges = ring_bond_set(graph, ligand)
    if not ring_edges:
        return []
    ring_graph = nx.Graph()
    ring_graph.add_edges_from(ring_edges)
    systems: list[tuple[set[int], set[tuple[int, int]]]] = []
    for component in nx.connected_components(ring_graph):
        atoms = set(component)
        bonds = {
            sorted_cut_bond(a, b)
            for a, b in ring_graph.subgraph(component).edges
        }
        systems.append((atoms, bonds))
    return systems


def _include_rings_and_groups(
    atoms: set[int],
    bonds: set[tuple[int, int]],
    groups: Mapping[str, tuple[set[int], set[tuple[int, int]]]],
    rings: list[tuple[set[int], set[tuple[int, int]]]],
) -> tuple[set[int], set[tuple[int, int]]]:
    """If any atom is in a ring or functional group, keep the whole group."""

    for group_atoms, group_bonds in groups.values():
        if atoms & group_atoms:
            atoms.update(group_atoms)
            bonds.update(group_bonds)
    for ring_atoms, ring_bonds in rings:
        if atoms & ring_atoms:
            atoms.update(ring_atoms)
            bonds.update(ring_bonds)
    return atoms, bonds


def _include_ortho_substituents(
    ligand: Ligand,
    atoms: set[int],
    bonds: set[tuple[int, int]],
) -> tuple[set[int], set[tuple[int, int]]]:
    """Keep heavy ortho substituents of a rotatable bond on a ring."""

    mol = _ligand_rdkit(ligand)
    if mol is None or Chem is None:
        return atoms, bonds
    query = Chem.MolFromSmarts(_ORTHO_SMARTS)
    if query is None:
        return atoms, bonds
    for match in mol.GetSubstructMatches(query):
        mapped = tuple(idx + 1 for idx in match)
        end = sorted_cut_bond(mapped[0], mapped[1])
        if end not in bonds:
            continue
        atoms.update((mapped[0], mapped[3]))
        bonds.add(end)
        bonds.add(sorted_cut_bond(mapped[2], mapped[3]))
    return atoms, bonds


def _add_non_rotor_ring_substituents(
    ligand: Ligand,
    atoms: set[int],
    bonds: set[tuple[int, int]],
    rings: list[tuple[set[int], set[tuple[int, int]]]],
) -> tuple[set[int], set[tuple[int, int]]]:
    """Keep non-rotatable heavy substituents on included ring systems."""

    graph = build_graph(ligand)
    rotatable = set(
        find_rotatable_bonds(
            ligand,
            include_rigid_single_bonds=True,
            graph=graph,
        )
    )
    ring_atoms = set()
    for system_atoms, _ in rings:
        if atoms & system_atoms:
            ring_atoms.update(system_atoms)
    for atom_idx in list(ring_atoms):
        for nbr in graph.neighbors(atom_idx):
            if graph.nodes[nbr]["atom"].element == "H":
                continue
            edge = sorted_cut_bond(atom_idx, nbr)
            if edge in rotatable:
                continue
            if (atom_idx in ring_atoms) == (nbr in ring_atoms):
                continue
            atoms.update((atom_idx, nbr))
            bonds.add(edge)
    return atoms, bonds


def _cap_open_valence(
    ligand: Ligand,
    atoms: set[int],
    bonds: set[tuple[int, int]],
    groups: Mapping[str, tuple[set[int], set[tuple[int, int]]]],
) -> tuple[set[int], set[tuple[int, int]]]:
    """Keep a neighboring carbon when the fragment ends at N, O, or S.

    OpenFF / the paper call this a methyl cap; scission then still applies
    its own capper on any remaining cut bonds.
    """

    graph = build_graph(ligand)
    in_group = {atom for group_atoms, _ in groups.values() for atom in group_atoms}
    extra_atoms: set[int] = set()
    extra_bonds: set[tuple[int, int]] = set()
    for atom_idx in list(atoms):
        element = graph.nodes[atom_idx]["atom"].element
        if element not in {"N", "O", "S"} and atom_idx not in in_group:
            continue
        needs_cap = False
        for nbr in graph.neighbors(atom_idx):
            if graph.nodes[nbr]["atom"].element == "H" or nbr in atoms:
                continue
            needs_cap = True
            break
        if not needs_cap:
            continue
        for nbr in graph.neighbors(atom_idx):
            if graph.nodes[nbr]["atom"].element != "C":
                continue
            extra_atoms.add(nbr)
            extra_bonds.add(sorted_cut_bond(atom_idx, nbr))
    atoms.update(extra_atoms)
    bonds.update(extra_bonds)
    return atoms, bonds


def _add_next_substituent(
    ligand: Ligand,
    torsion: TorsionDefinition,
    atoms: set[int],
    bonds: set[tuple[int, int]],
    groups: Mapping[str, tuple[set[int], set[tuple[int, int]]]],
    rings: list[tuple[set[int], set[tuple[int, int]]]],
) -> tuple[set[int], set[tuple[int, int]]] | None:
    """Add the nearest unused heavy neighbor (paper ``path_length`` heuristic)."""

    graph = build_graph(ligand)
    target = torsion.bond
    ranked: list[tuple[int, int, int, int]] = []
    for atom_idx in atoms:
        if graph.nodes[atom_idx]["atom"].element == "H":
            continue
        for nbr in graph.neighbors(atom_idx):
            if graph.nodes[nbr]["atom"].element == "H" or nbr in atoms:
                continue
            d1 = nx.shortest_path_length(graph, target[0], nbr)
            d2 = nx.shortest_path_length(graph, target[1], nbr)
            ranked.append((min(d1, d2), max(d1, d2), nbr, atom_idx))
    if not ranked:
        return None
    ranked.sort()
    neighbour, from_atom = ranked[0][2], ranked[0][3]
    atoms.add(neighbour)
    bonds.add(sorted_cut_bond(from_atom, neighbour))
    for group_atoms, group_bonds in groups.values():
        if neighbour in group_atoms:
            atoms.update(group_atoms)
            bonds.update(group_bonds)
    for ring_atoms, ring_bonds in rings:
        if neighbour in ring_atoms:
            atoms.update(ring_atoms)
            bonds.update(ring_bonds)
    return atoms, bonds
