from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import networkx as nx

from .Graph import build_graph, ring_bond_set, sorted_cut_bond
from .Models import CandidateFragment, CapSite, Ligand, TorsionDefinition
from .Torsions import find_rotatable_bonds


@dataclass(frozen=True)
class FragmentationTopology:
    """Ligand-level rigid-domain decomposition reused across torsions."""

    graph: nx.Graph
    heavy_graph: nx.Graph
    ring_edges: frozenset[tuple[int, int]]
    domains: tuple[frozenset[int], ...]
    atom_to_domain: dict[int, int]
    domain_graph: nx.Graph
    domain_has_ring: dict[int, bool]


def get_fragmentation_topology(
    ligand: Ligand,
    include_rigid_single_bonds: bool = True,
    rotatable_bond_smarts: tuple[str, ...] = (),
) -> FragmentationTopology:
    """Return domains / rings / rotatable cuts, cached once per ligand+config.

    ``build_candidate_fragments`` used to rebuild this stack for every torsion.
    """

    cache_key = (bool(include_rigid_single_bonds), tuple(rotatable_bond_smarts))
    cached = getattr(ligand, "_frag_topology", None)
    if cached is not None and cached[0] == cache_key:
        return cached[1]

    graph = build_graph(ligand)
    heavy_graph = _build_heavy_graph(graph)
    ring_edges = frozenset(ring_bond_set(graph, ligand))
    domains, atom_to_domain, domain_graph, domain_has_ring = _build_domains(
        ligand,
        heavy_graph,
        ring_edges=ring_edges,
        include_rigid_single_bonds=include_rigid_single_bonds,
        rotatable_bond_smarts=rotatable_bond_smarts,
    )
    topo = FragmentationTopology(
        graph=graph,
        heavy_graph=heavy_graph,
        ring_edges=ring_edges,
        domains=tuple(frozenset(d) for d in domains),
        atom_to_domain=atom_to_domain,
        domain_graph=domain_graph,
        domain_has_ring=domain_has_ring,
    )
    ligand._frag_topology = (cache_key, topo)
    return topo


def _is_heavy(atom) -> bool:
    """Return whether an atom should be treated as heavy.

    Args:
        atom: Atom-like object exposing an ``element`` attribute.

    Returns:
        ``True`` when the atom is not hydrogen.
    """

    return atom.element != "H"


def _candidate_id(retained_atoms: set[int], cut_bonds: set[tuple[int, int]]) -> str:
    """Build a stable identifier from retained atoms and cut bonds.

    Args:
        retained_atoms: Parent atom indices kept in the candidate.
        cut_bonds: Normalized parent bond pairs cut for the candidate.

    Returns:
        A deterministic fragment identifier string.
    """

    atom_key = ",".join(map(str, sorted(retained_atoms)))
    cut_key = ",".join(f"{a}-{b}" for a, b in sorted(cut_bonds))
    return f"atoms[{atom_key}]|cuts[{cut_key}]"


def _candidate_charge(ligand: Ligand, retained_atoms: set[int]) -> float:
    """Sum the parent partial charges for a retained atom set.

    Args:
        ligand: Parent ligand record.
        retained_atoms: Parent atom indices retained in the fragment.

    Returns:
        The net charge of the retained atoms.
    """

    return float(sum(ligand.atom(atom_idx).charge for atom_idx in retained_atoms))


def _add_attached_hydrogens(graph: nx.Graph, heavy_atoms: set[int]) -> set[int]:
    """Expand a heavy-atom selection to include directly attached hydrogens.

    Args:
        graph: Molecular graph containing hydrogen atoms.
        heavy_atoms: Parent heavy-atom indices to retain.

    Returns:
        A retained atom set containing the heavy atoms and their bonded
        hydrogens.
    """

    retained = set(heavy_atoms)
    for atom_idx in list(heavy_atoms):
        retained.update(
            nbr for nbr in graph.neighbors(atom_idx)
            if graph.nodes[nbr]["atom"].element == "H"
        )
    return retained


def _build_heavy_graph(graph: nx.Graph) -> nx.Graph:
    """Return a copy of the molecular graph with hydrogens removed.

    Args:
        graph: Full molecular graph.

    Returns:
        A graph containing only heavy atoms and heavy-heavy bonds.
    """

    heavy = nx.Graph()
    for node, data in graph.nodes(data=True):
        if data["atom"].element == "H":
            continue
        heavy.add_node(node, **data)
    for atom1, atom2, data in graph.edges(data=True):
        if atom1 in heavy and atom2 in heavy:
            heavy.add_edge(atom1, atom2, **data)
    return heavy


def _build_domains(
    ligand: Ligand,
    heavy_graph: nx.Graph,
    ring_edges: frozenset[tuple[int, int]] | set[tuple[int, int]] | None = None,
    include_rigid_single_bonds: bool = True,
    rotatable_bond_smarts: tuple[str, ...] = (),
) -> tuple[list[set[int]], dict[int, int], nx.Graph, dict[int, bool]]:
    """Collapse rigid heavy-atom regions into a domain graph.

    Args:
        ligand: Parent ligand record used to identify rotatable bonds.
        heavy_graph: Heavy-atom-only molecular graph.
        ring_edges: Optional precomputed ring-bond set (avoids a second
            ``cycle_basis``). When omitted, computed from ``heavy_graph``.
        include_rigid_single_bonds: Whether to split rigid domains across
            amide-like acyclic single bonds in addition to the default set of
            acyclic single-bond torsions.
        rotatable_bond_smarts: SMARTS override patterns that may nominate
            otherwise excluded bonds as torsion targets.

    Returns:
        A tuple containing the rigid-domain atom sets, the atom-to-domain map,
        the domain connectivity graph, and per-domain ring flags.
    """

    rotatable = set(
        find_rotatable_bonds(
            ligand,
            include_rigid_single_bonds=include_rigid_single_bonds,
            rotatable_bond_smarts=rotatable_bond_smarts,
        )
    )
    rigid_graph = heavy_graph.copy()
    rigid_graph.remove_edges_from([edge for edge in rotatable if rigid_graph.has_edge(*edge)])
    if ring_edges is None:
        ring_edges = frozenset(ring_bond_set(heavy_graph))
    ring_atoms = {atom for edge in ring_edges for atom in edge}

    domains = [set(component) for component in nx.connected_components(rigid_graph)]
    atom_to_domain: dict[int, int] = {}
    domain_has_ring: dict[int, bool] = {}
    for domain_id, atoms in enumerate(domains):
        for atom_idx in atoms:
            atom_to_domain[atom_idx] = domain_id
        domain_has_ring[domain_id] = bool(atoms & ring_atoms)

    domain_graph = nx.Graph()
    for domain_id, atoms in enumerate(domains):
        domain_graph.add_node(domain_id, atoms=atoms, has_ring=domain_has_ring[domain_id])
    for atom1, atom2 in rotatable:
        left = atom_to_domain[atom1]
        right = atom_to_domain[atom2]
        if left == right:
            continue
        bond = tuple(sorted((atom1, atom2)))
        domain_graph.add_edge(left, right, bond=bond)
    return domains, atom_to_domain, domain_graph, domain_has_ring


def _side_distances(domain_graph: nx.Graph, start: int, blocked: int) -> dict[int, int]:
    """Measure shell depths on one side of a blocked domain-domain edge.

    Args:
        domain_graph: Domain connectivity graph.
        start: Domain from which to start the breadth-first search.
        blocked: Adjacent domain to exclude from traversal.

    Returns:
        A mapping from reachable domain id to shell depth.
    """

    queue: deque[tuple[int, int]] = deque([(start, 0)])
    seen = {blocked}
    distances: dict[int, int] = {}
    while queue:
        node, depth = queue.popleft()
        if node in seen:
            continue
        seen.add(node)
        distances[node] = depth
        for nbr in domain_graph.neighbors(node):
            if nbr not in seen:
                queue.append((nbr, depth + 1))
    return distances


def _cut_bonds_for_retained(
    graph: nx.Graph,
    retained_heavy: set[int],
    ring_edges: set[tuple[int, int]],
) -> tuple[set[tuple[int, int]], list[CapSite]]:
    """Find cuttable outgoing single bonds and record their cap sites.

    Args:
        graph: Full molecular graph.
        retained_heavy: Heavy atoms kept in the candidate.
        ring_edges: Bond pairs that belong to rings and must not be cut.

    Returns:
        A tuple of normalized cut bonds and one :class:`CapSite` per cut bond
        (a retained atom may carry several caps when it has multiple cut
        neighbors).
    """

    cut_bonds: set[tuple[int, int]] = set()
    cap_sites: list[CapSite] = []
    for atom_idx in sorted(retained_heavy):
        for nbr in graph.neighbors(atom_idx):
            edge = sorted_cut_bond(atom_idx, nbr)
            if nbr in retained_heavy or edge in ring_edges:
                continue
            if graph.nodes[nbr]["atom"].element == "H":
                continue
            bond_type = graph.edges[atom_idx, nbr]["bond_type"]
            if bond_type.lower() in {"1", "1.0", "single"}:
                cut_bonds.add(edge)
                cap_sites.append(CapSite(retained_atom=atom_idx, removed_atom=nbr, bond_type=bond_type))
    return cut_bonds, cap_sites


def _candidate_from_domains(
    ligand: Ligand,
    graph: nx.Graph,
    ring_edges: set[tuple[int, int]],
    domain_atoms: list[set[int]],
    atom_to_domain: dict[int, int],
    domain_has_ring: dict[int, bool],
    included_domains: set[int],
    torsion: TorsionDefinition,
    shell_level: int,
    is_parent_fallback: bool = False,
) -> CandidateFragment:
    """Build a candidate fragment from a chosen set of rigid domains.

    Args:
        ligand: Parent ligand record.
        graph: Full molecular graph.
        ring_edges: Ring bond set that must be preserved.
        domain_atoms: Atom membership for each rigid domain.
        atom_to_domain: Mapping from heavy atom index to domain id.
        domain_has_ring: Mapping from domain id to ring-presence flag.
        included_domains: Domain ids to include in the fragment.
        torsion: Torsion the fragment is intended to preserve.
        shell_level: Shell expansion depth used to generate the candidate.
        is_parent_fallback: Whether this candidate represents the full parent.

    Returns:
        A populated :class:`CandidateFragment`.
    """

    retained_heavy = set()
    for domain_id in included_domains:
        retained_heavy.update(domain_atoms[domain_id])
    cut_bonds, cap_sites = _cut_bonds_for_retained(graph, retained_heavy, ring_edges)
    retained_atoms = _add_attached_hydrogens(graph, retained_heavy)
    ring_cap_count = sum(
        1
        for site in cap_sites
        if domain_has_ring.get(atom_to_domain[site.retained_atom], False)
    )
    return CandidateFragment(
        candidate_id=_candidate_id(retained_atoms, cut_bonds),
        retained_atoms=retained_atoms,
        cut_bonds=cut_bonds,
        cap_sites=cap_sites,
        shell_level=shell_level,
        net_charge=_candidate_charge(ligand, retained_atoms),
        ring_count=sum(1 for domain_id in included_domains if domain_has_ring.get(domain_id, False)),
        ring_cap_count=ring_cap_count,
        is_parent_fallback=is_parent_fallback,
        torsion_labels={torsion.label},
    )


def build_candidate_fragments(
    ligand: Ligand,
    torsion: TorsionDefinition,
    include_rigid_single_bonds: bool = True,
    rotatable_bond_smarts: tuple[str, ...] = (),
) -> list[CandidateFragment]:
    """Enumerate fragment candidates that preserve the requested torsion.

    Args:
        ligand: Parent ligand to fragment.
        torsion: Torsion that must remain intact in every candidate.
        include_rigid_single_bonds: Whether to allow domain splitting across
            amide-like acyclic single bonds as well as the default rotatable
            single bonds.
        rotatable_bond_smarts: SMARTS override patterns that may nominate
            otherwise excluded bonds as torsion targets.

    Returns:
        Candidate fragments sorted from smaller shells to larger fallbacks.
    """

    topo = get_fragmentation_topology(
        ligand,
        include_rigid_single_bonds=include_rigid_single_bonds,
        rotatable_bond_smarts=rotatable_bond_smarts,
    )
    graph = topo.graph
    ring_edges = set(topo.ring_edges)
    domains = [set(d) for d in topo.domains]
    atom_to_domain = topo.atom_to_domain
    domain_graph = topo.domain_graph
    domain_has_ring = topo.domain_has_ring

    a, b, c, d = torsion.atom_indices
    left_domain = atom_to_domain[b]
    right_domain = atom_to_domain[c]
    if left_domain == right_domain:
        full = _candidate_from_domains(
            ligand,
            graph,
            ring_edges,
            domains,
            atom_to_domain,
            domain_has_ring,
            set(range(len(domains))),
            torsion,
            shell_level=0,
            is_parent_fallback=True,
        )
        return [full]

    left_distances = _side_distances(domain_graph, left_domain, right_domain)
    right_distances = _side_distances(domain_graph, right_domain, left_domain)
    left_required = {
        atom_to_domain[atom_idx]
        for atom_idx in (a, b)
        if atom_to_domain[atom_idx] in left_distances
    }
    right_required = {
        atom_to_domain[atom_idx]
        for atom_idx in (c, d)
        if atom_to_domain[atom_idx] in right_distances
    }
    min_left = max((left_distances[domain_id] for domain_id in left_required), default=0)
    min_right = max((right_distances[domain_id] for domain_id in right_required), default=0)
    max_left = max(left_distances.values(), default=0)
    max_right = max(right_distances.values(), default=0)

    # Cumulative shells: only keep depths that change the domain set so the
    # leftxright product does not rebuild identical fragments.
    def _unique_shells(
        distances: dict[int, int], min_depth: int, max_depth: int
    ) -> list[tuple[int, frozenset[int]]]:
        shells: list[tuple[int, frozenset[int]]] = []
        seen: set[frozenset[int]] = set()
        for depth in range(min_depth, max_depth + 1):
            domain_set = frozenset(
                domain_id for domain_id, d in distances.items() if d <= depth
            )
            if domain_set in seen:
                continue
            seen.add(domain_set)
            shells.append((depth, domain_set))
        return shells

    left_shells = _unique_shells(left_distances, min_left, max_left)
    right_shells = _unique_shells(right_distances, min_right, max_right)

    # Keyed by included domain frozenset -> (min shell_level, domains)
    domain_combos: dict[frozenset[int], int] = {}
    for left_depth, left_set in left_shells:
        for right_depth, right_set in right_shells:
            included = left_set | right_set
            if len(included) == len(domains):
                continue
            shell_level = left_depth + right_depth
            prev = domain_combos.get(included)
            if prev is None or shell_level < prev:
                domain_combos[included] = shell_level

    candidates_by_id: dict[str, CandidateFragment] = {}
    for included_domains, shell_level in domain_combos.items():
        candidate = _candidate_from_domains(
            ligand,
            graph,
            ring_edges,
            domains,
            atom_to_domain,
            domain_has_ring,
            set(included_domains),
            torsion,
            shell_level=shell_level,
        )
        candidates_by_id.setdefault(candidate.candidate_id, candidate)

    full_candidate = _candidate_from_domains(
        ligand,
        graph,
        ring_edges,
        domains,
        atom_to_domain,
        domain_has_ring,
        set(range(len(domains))),
        torsion,
        shell_level=max_left + max_right + 1,
        is_parent_fallback=True,
    )
    candidates_by_id.setdefault(full_candidate.candidate_id, full_candidate)

    return sorted(
        candidates_by_id.values(),
        key=lambda candidate: (
            len(candidate.retained_atoms),
            candidate.shell_level,
            candidate.candidate_id,
        ),
    )
