from __future__ import annotations

import math
from dataclasses import dataclass

import networkx as nx
import numpy as np

from .Capping import bond_type_to_order, heavy_cap_bond_length
from .Graph import build_graph, retained_distance_map
from .Models import CandidateFragment, ClashThresholds, Ligand, TorsionDefinition

VDW_RADII = {
    "H": 1.20,
    "C": 1.70,
    "N": 1.55,
    "O": 1.52,
    "F": 1.47,
    "P": 1.80,
    "S": 1.80,
    "Cl": 1.75,
    "Br": 1.85,
    "I": 1.98,
}

CAP_BOND_LENGTHS = {
    "C": 1.09,
    "N": 1.01,
    "O": 0.98,
    "S": 1.34,
    "P": 1.42,
}


@dataclass(frozen=True)
class ScreenResult:
    """Outcome of clash-screening a candidate over sampled torsion angles.

    Attributes:
        accepted: Whether the candidate passed all sampled angles.
        worst_margin: Smallest nonbonded margin observed during screening.
        reason: Short failure code when the candidate is rejected.
        violation: Optional details describing the worst clash encountered.
    """

    accepted: bool
    worst_margin: float
    reason: str | None = None
    violation: dict[str, object] | None = None


def _rotation_matrix(axis: np.ndarray, theta_rad: float) -> np.ndarray:
    """Return a 3D rotation matrix around ``axis`` by ``theta_rad`` radians.

    Args:
        axis: Rotation axis vector.
        theta_rad: Rotation angle in radians.

    Returns:
        A ``3 x 3`` rotation matrix.
    """

    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    cos_t = math.cos(theta_rad)
    sin_t = math.sin(theta_rad)
    return np.array(
        [
            [cos_t + x * x * (1 - cos_t), x * y * (1 - cos_t) - z * sin_t, x * z * (1 - cos_t) + y * sin_t],
            [y * x * (1 - cos_t) + z * sin_t, cos_t + y * y * (1 - cos_t), y * z * (1 - cos_t) - x * sin_t],
            [z * x * (1 - cos_t) - y * sin_t, z * y * (1 - cos_t) + x * sin_t, cos_t + z * z * (1 - cos_t)],
        ]
    )


def _dihedral_angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray) -> float:
    """Compute the signed dihedral angle defined by four points.

    Args:
        p1: First point.
        p2: Second point.
        p3: Third point.
        p4: Fourth point.

    Returns:
        The dihedral angle in degrees.
    """

    b0 = p2 - p1
    b1 = p3 - p2
    b2 = p4 - p3
    b1 = b1 / np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)
    return math.degrees(math.atan2(y, x))


def _descendants(graph: nx.Graph, start: int, blocked: int, allowed: set[int]) -> set[int]:
    """Traverse one side of a bond while staying inside the allowed atoms.

    Args:
        graph: Molecular graph.
        start: Atom index from which to begin traversal.
        blocked: Atom index on the opposite side of the bond.
        allowed: Atom indices allowed to remain in the traversal.

    Returns:
        Reachable atoms on the chosen side of the bond.
    """

    stack = [start]
    seen = {blocked}
    descendants: set[int] = set()
    while stack:
        node = stack.pop()
        if node in seen or node not in allowed:
            continue
        seen.add(node)
        descendants.add(node)
        for nbr in graph.neighbors(node):
            if nbr not in seen:
                stack.append(nbr)
    return descendants


def cap_direction(
    graph: nx.Graph,
    ligand: Ligand,
    retained_atom: int,
    removed_atom: int,
    coordinates: dict[int, np.ndarray],
) -> np.ndarray:
    """Choose an outward cap direction for a cut bond.

    Args:
        graph: Molecular graph.
        ligand: Parent ligand record.
        retained_atom: Atom that stays in the fragment.
        removed_atom: Bonded atom removed from the fragment.
        coordinates: Coordinate map for the current conformation.

    Returns:
        A unit vector pointing in the preferred cap direction.
    """

    origin = coordinates[retained_atom]
    neighbor_vectors: list[np.ndarray] = []
    for nbr in graph.neighbors(retained_atom):
        if nbr == removed_atom:
            continue
        nbr_coord = coordinates.get(nbr)
        if nbr_coord is None:
            nbr_coord = np.array(ligand.atom(nbr).coords, dtype=float)
        vector = nbr_coord - origin
        norm = np.linalg.norm(vector)
        if norm > 1.0e-8:
            neighbor_vectors.append(vector / norm)

    if len(neighbor_vectors) >= 2:
        direction = -np.sum(neighbor_vectors, axis=0)
        norm = np.linalg.norm(direction)
        if norm > 1.0e-8:
            return direction / norm

    removed = np.array(ligand.atom(removed_atom).coords, dtype=float)
    fallback = origin - removed
    fallback_norm = np.linalg.norm(fallback)
    if fallback_norm > 1.0e-8:
        return fallback / fallback_norm
    return np.array([1.0, 0.0, 0.0], dtype=float)


def _build_cap_coordinates(
    ligand: Ligand,
    candidate: CandidateFragment,
    coordinates: dict[int, np.ndarray],
    graph: nx.Graph,
) -> dict[str, np.ndarray]:
    """Generate trial cap coordinates for a screened fragment conformation.

    Args:
        ligand: Parent ligand record.
        candidate: Candidate fragment being screened.
        coordinates: Current coordinates for retained parent atoms.
        graph: Prebuilt molecular graph (topology is fixed during a scan).

    Returns:
        Mapping from cap identifier to proposed cap coordinate.
    """

    cap_coords: dict[str, np.ndarray] = {}
    for site in sorted(candidate.cap_sites, key=lambda s: (s.retained_atom, s.removed_atom)):
        retained = ligand.atom(site.retained_atom)
        removed = ligand.atom(site.removed_atom)
        origin = coordinates[site.retained_atom]
        direction = cap_direction(graph, ligand, site.retained_atom, site.removed_atom, coordinates)
        # Place the cap heavy atom (the dominant clash contributor) at the
        # element-matched bond length, conservatively assuming the removed atom
        # is recreated even when a bare hydrogen might ultimately be used.
        bond_length = heavy_cap_bond_length(
            retained.element, removed.element, bond_type_to_order(site.bond_type)
        )
        cap_coords[f"cap_{site.retained_atom}_{site.removed_atom}"] = origin + direction * bond_length
    return cap_coords


def _heavy_clash_tables(
    ligand: Ligand,
    retained_atoms: set[int],
    distances: dict[int, dict[int, int]],
    thresholds: ClashThresholds,
) -> tuple[list[int], np.ndarray, np.ndarray, np.ndarray]:
    """Precompute heavy-atom clash pairs and VDW allowances (caps unused in scoring).

    Caps were previously built each angle then skipped in ``_minimum_margin``
    (string cap ids fail the ``isinstance(..., int)`` gate). Screening only
    scores retained heavy-heavy contacts with graph distance > 2.
    """

    heavy_idxs = [
        idx
        for idx in sorted(retained_atoms)
        if ligand.atom(idx).element != "H"
    ]
    n = len(heavy_idxs)
    left_rows: list[int] = []
    right_rows: list[int] = []
    allowed: list[float] = []
    for i in range(n):
        li = heavy_idxs[i]
        left_element = ligand.atom(li).element
        left_vdw = VDW_RADII.get(left_element, 1.7)
        for j in range(i + 1, n):
            rj = heavy_idxs[j]
            path_length = distances.get(li, {}).get(rj, 99)
            if path_length <= 2:
                continue
            scale = thresholds.path3_scale if path_length == 3 else thresholds.far_scale
            right_element = ligand.atom(rj).element
            allowed.append(
                scale * (left_vdw + VDW_RADII.get(right_element, 1.2))
            )
            left_rows.append(i)
            right_rows.append(j)
    return (
        heavy_idxs,
        np.asarray(left_rows, dtype=np.intp),
        np.asarray(right_rows, dtype=np.intp),
        np.asarray(allowed, dtype=float),
    )


def _vectorized_worst_margin(
    heavy_coords: np.ndarray,
    left_rows: np.ndarray,
    right_rows: np.ndarray,
    allowed: np.ndarray,
) -> tuple[float, int]:
    """Return (worst_margin, pair_index) for precomputed heavy-atom pairs."""

    if left_rows.size == 0:
        return 0.0, -1
    delta = heavy_coords[left_rows] - heavy_coords[right_rows]
    observed = np.linalg.norm(delta, axis=1)
    margins = observed - allowed
    worst_i = int(np.argmin(margins))
    return float(margins[worst_i]), worst_i


def _minimum_margin(
    ligand: Ligand,
    retained_atoms: set[int],
    coordinates: dict[int, np.ndarray],
    caps: dict[str, np.ndarray],
    thresholds: ClashThresholds,
    distances: dict[int, dict[int, int]],
) -> tuple[float, dict[str, object] | None]:
    """Measure the worst nonbonded margin among retained heavy atoms.

    ``caps`` is accepted for API compatibility but ignored: historical scoring
    never counted cap-atom pairs (cap ids are non-integers).

    Args:
        ligand: Parent ligand record.
        retained_atoms: Parent atom indices retained in the fragment.
        coordinates: Coordinates for retained atoms in the current conformer.
        caps: Unused; retained for call-site compatibility.
        thresholds: Clash-threshold parameters.
        distances: Precomputed shortest-path lengths on the retained subgraph.

    Returns:
        A tuple of the worst observed margin and optional detail for the most
        restrictive atom pair.
    """

    del caps  # built historically but never scored
    heavy_idxs, left_rows, right_rows, allowed = _heavy_clash_tables(
        ligand, retained_atoms, distances, thresholds
    )
    if not heavy_idxs or left_rows.size == 0:
        return 0.0, None
    heavy_coords = np.asarray([coordinates[idx] for idx in heavy_idxs], dtype=float)
    worst_margin, worst_i = _vectorized_worst_margin(
        heavy_coords, left_rows, right_rows, allowed
    )
    if worst_i < 0:
        return 0.0, None
    li = heavy_idxs[int(left_rows[worst_i])]
    rj = heavy_idxs[int(right_rows[worst_i])]
    path_length = distances.get(li, {}).get(rj, 99)
    observed = float(np.linalg.norm(heavy_coords[int(left_rows[worst_i])] - heavy_coords[int(right_rows[worst_i])]))
    worst_detail = {
        "left_atom_index": li,
        "left_atom_name": ligand.atom(li).name,
        "left_element": ligand.atom(li).element,
        "right_atom_index": rj,
        "right_atom_name": ligand.atom(rj).name,
        "right_element": ligand.atom(rj).element,
        "graph_distance": path_length,
        "observed_distance": observed,
        "allowed_distance": float(allowed[worst_i]),
        "margin": worst_margin,
    }
    return worst_margin, worst_detail


def screen_candidate(
    ligand: Ligand,
    torsion: TorsionDefinition,
    candidate: CandidateFragment,
    angle_step: int,
    thresholds: ClashThresholds,
) -> ScreenResult:
    """Reject fragments whose sampled torsions introduce steric clashes.

    Topology (graph / bond-path distances) is fixed for a candidate; only
    Cartesian coordinates change across the torsion scan. Cap coordinates are
    not built: clash scoring only uses retained heavy atoms.

    Args:
        ligand: Parent ligand record.
        torsion: Torsion being preserved by the candidate.
        candidate: Candidate fragment to evaluate.
        angle_step: Sampling interval in degrees for the torsion scan.
        thresholds: Clash-threshold parameters.

    Returns:
        A :class:`ScreenResult` describing whether the candidate passed.
    """

    graph = build_graph(ligand)
    retained = candidate.retained_atoms
    a, b, c, d = torsion.atom_indices
    if not {a, b, c, d}.issubset(retained):
        return ScreenResult(False, -999.0, "candidate_missing_torsion_atoms", None)
    if not candidate.cut_bonds:
        return ScreenResult(True, 0.0, None, None)

    side_from_c = _descendants(graph, d, c, retained)
    side_from_b = _descendants(graph, a, b, retained)
    rotating_side = side_from_c if len(side_from_c) <= len(side_from_b) else side_from_b
    distances = retained_distance_map(ligand, retained)
    original = ligand.coordinates
    pivot1 = original[b]
    pivot2 = original[c]
    axis = pivot2 - pivot1
    if np.linalg.norm(axis) == 0:
        return ScreenResult(False, -999.0, "zero_length_torsion_axis", None)

    current_dihedral = _dihedral_angle(
        original[a],
        original[b],
        original[c],
        original[d],
    )

    retained_list = sorted(retained)
    row_of = {idx: i for i, idx in enumerate(retained_list)}
    base_coords = np.asarray([original[idx] for idx in retained_list], dtype=float)
    rotating_rows = np.asarray(
        [row_of[idx] for idx in rotating_side if idx in row_of],
        dtype=np.intp,
    )
    heavy_idxs, left_rows, right_rows, allowed = _heavy_clash_tables(
        ligand, retained, distances, thresholds
    )
    heavy_rows = np.asarray([row_of[idx] for idx in heavy_idxs], dtype=np.intp)
    pivot = np.asarray(pivot1, dtype=float)

    worst = float("inf")
    for target_angle in range(-180, 180, angle_step):
        coords = base_coords.copy()
        delta_angle = ((target_angle - current_dihedral + 180.0) % 360.0) - 180.0
        if rotating_rows.size:
            rotation = _rotation_matrix(axis, math.radians(delta_angle))
            shifted = coords[rotating_rows] - pivot
            coords[rotating_rows] = (rotation @ shifted.T).T + pivot
        if heavy_rows.size == 0 or left_rows.size == 0:
            margin, worst_i = 0.0, -1
            detail = None
        else:
            heavy_coords = coords[heavy_rows]
            margin, worst_i = _vectorized_worst_margin(
                heavy_coords, left_rows, right_rows, allowed
            )
            detail = None
            if worst_i >= 0:
                li = heavy_idxs[int(left_rows[worst_i])]
                rj = heavy_idxs[int(right_rows[worst_i])]
                path_length = distances.get(li, {}).get(rj, 99)
                observed = float(
                    np.linalg.norm(
                        heavy_coords[int(left_rows[worst_i])]
                        - heavy_coords[int(right_rows[worst_i])]
                    )
                )
                detail = {
                    "left_atom_index": li,
                    "left_atom_name": ligand.atom(li).name,
                    "left_element": ligand.atom(li).element,
                    "right_atom_index": rj,
                    "right_atom_name": ligand.atom(rj).name,
                    "right_element": ligand.atom(rj).element,
                    "graph_distance": path_length,
                    "observed_distance": observed,
                    "allowed_distance": float(allowed[worst_i]),
                    "margin": margin,
                }
        worst = min(worst, margin)
        if margin < 0:
            if detail is not None:
                detail = {**detail, "target_angle": target_angle}
            return ScreenResult(False, worst, f"clash_at_{target_angle}", detail)
    return ScreenResult(True, worst if worst != float("inf") else 0.0, None, None)


def cap_site_scan_margin(
    ligand: Ligand,
    candidate: CandidateFragment,
    torsion: TorsionDefinition,
    retained_atom: int,
    removed_atom: int,
    cap_element: str,
    angle_step: int,
    thresholds: ClashThresholds,
) -> float:
    """Worst clash margin for a heavy cap at one site over the torsion scan.

    Used to decide which carbon cut can carry a bulky (methyl) cap: a roomy site
    yields a large margin, a crowded site a small or negative one. The heavy cap
    atom is placed at the cut site and the torsion is rotated through a full turn;
    the smallest margin between that cap atom and any retained heavy atom is
    returned.

    Topology is fixed for the candidate; coordinates are rotated with the same
    vectorized path as :func:`screen_candidate`. Cap placement still follows
    :func:`cap_direction` each angle (neighbor geometry can rotate).

    Args:
        ligand: Parent ligand record.
        candidate: Candidate fragment being capped.
        torsion: Torsion that will be scanned during fitting.
        retained_atom: Parent index of the atom the cap attaches to.
        removed_atom: Parent index of the removed atom defining the cap direction.
        cap_element: Element of the trial heavy cap (typically ``"C"``).
        angle_step: Torsion sampling interval in degrees.
        thresholds: Clash-threshold parameters.

    Returns:
        The worst (smallest) nonbonded margin observed for the cap heavy atom, or
        ``inf`` when the torsion cannot be scanned in this candidate.
    """

    graph = build_graph(ligand)
    retained = candidate.retained_atoms
    a, b, c, d = torsion.atom_indices
    if not {a, b, c, d, retained_atom}.issubset(retained):
        return float("inf")

    distances = retained_distance_map(ligand, retained)
    # Through-bond separation from the cap (one bond past the retained atom).
    sub_distances = distances.get(retained_atom, {})
    retained_element = ligand.atom(retained_atom).element
    bond_length = heavy_cap_bond_length(retained_element, cap_element, 1)

    original = ligand.coordinates
    retained_list = sorted(retained)
    row_of = {idx: i for i, idx in enumerate(retained_list)}
    base_coords = np.asarray([original[idx] for idx in retained_list], dtype=float)
    attach_row = row_of[retained_atom]

    partner_idxs: list[int] = []
    allowed_list: list[float] = []
    for atom_idx in retained_list:
        element = ligand.atom(atom_idx).element
        if element == "H":
            continue
        path_length = 1 + sub_distances.get(atom_idx, 99)
        if path_length <= 2:
            continue
        scale = thresholds.path3_scale if path_length == 3 else thresholds.far_scale
        allowed_list.append(
            scale * (VDW_RADII.get(cap_element, 1.7) + VDW_RADII.get(element, 1.7))
        )
        partner_idxs.append(atom_idx)
    partner_rows = np.asarray(
        [row_of[idx] for idx in partner_idxs], dtype=np.intp
    )
    allowed = np.asarray(allowed_list, dtype=float)

    if candidate.cut_bonds:
        side_from_c = _descendants(graph, d, c, retained)
        side_from_b = _descendants(graph, a, b, retained)
        rotating_side = side_from_c if len(side_from_c) <= len(side_from_b) else side_from_b
        pivot1 = original[b]
        axis = original[c] - pivot1
        if np.linalg.norm(axis) == 0:
            return 0.0
        current = _dihedral_angle(original[a], original[b], original[c], original[d])
        angles = list(range(-180, 180, angle_step))
        rotating_rows = np.asarray(
            [row_of[idx] for idx in rotating_side if idx in row_of],
            dtype=np.intp,
        )
        pivot = np.asarray(pivot1, dtype=float)
    else:
        rotating_rows = np.asarray([], dtype=np.intp)
        axis = np.zeros(3)
        current = 0.0
        angles = [0]
        pivot = np.zeros(3)

    if partner_rows.size == 0:
        return 0.0

    worst = float("inf")
    for target_angle in angles:
        coords = base_coords.copy()
        if rotating_rows.size and np.linalg.norm(axis) > 1.0e-8:
            delta = ((target_angle - current + 180.0) % 360.0) - 180.0
            rotation = _rotation_matrix(axis, math.radians(delta))
            shifted = coords[rotating_rows] - pivot
            coords[rotating_rows] = (rotation @ shifted.T).T + pivot
        # Neighbor dict for cap_direction (small; only attach neighborhood).
        coord_map = {idx: coords[row_of[idx]] for idx in retained}
        direction = cap_direction(
            graph, ligand, retained_atom, removed_atom, coord_map
        )
        cap_pos = coords[attach_row] + direction * bond_length
        delta = cap_pos - coords[partner_rows]
        observed = np.linalg.norm(delta, axis=1)
        margins = observed - allowed
        worst = min(worst, float(np.min(margins)))
    return worst if worst != float("inf") else 0.0
