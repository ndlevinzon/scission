"""Chemistry-aware capping of severed fragment bonds.

When a bond ``R-X`` is cut to isolate a fragment, the dangling valence on the
retained atom ``R`` must be capped. This module decides *what* to place there:

- Prefer a bare hydrogen (``R-H``) -- it is minimal and introduces no spurious
  hydrogen-bond donor/acceptor.
- Never let that cap hydrogen carry a negative (or very low) partial charge.
  When the integer-charge balancing would push it below the configured
  threshold, escalate to an element-matched cap instead.
- A bare hydrogen is only appropriate when ``R`` is carbon *and* the removed
  atom ``X`` is carbon/hydrogen. Otherwise recreate the removed atom as a
  hydrogen-saturated group that matches its element and the cut bond order:
  ``C-C -> C-CH3`` / ``C=CH2`` / ``C#CH``, ``C-O -> C-OH`` / ``C=O``,
  ``C-N -> C-NH2`` / ``C=NH`` / ``C#N``, ``C-S -> C-SH``.

Charges follow the existing scheme: parent charges are preserved and only the
integer-charge correction is distributed across caps (no charge model is
invoked here). Cap coordinates are idealized, not minimized -- they are local
scan artifacts that downstream Amber tooling re-parameterizes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# Standard heavy-atom valences, used to decide how many hydrogens saturate a
# matched cap once the cut bond order is accounted for.
STANDARD_VALENCE = {
    "C": 4,
    "N": 3,
    "O": 2,
    "S": 2,
    "P": 3,
    "F": 1,
    "Cl": 1,
    "Br": 1,
    "I": 1,
}

# GAFF2 atom type for a matched cap heavy atom, keyed by element and the bond
# order to the retained atom (which fixes the hybridization).
CAP_HEAVY_TYPES = {
    "C": {1: "c3", 2: "c2", 3: "c1"},
    "N": {1: "n3", 2: "n2", 3: "n1"},
    "O": {1: "oh", 2: "o"},
    "S": {1: "sh", 2: "s"},
    "P": {1: "p3"},
    "F": {1: "f"},
    "Cl": {1: "cl"},
    "Br": {1: "br"},
    "I": {1: "i"},
}

# GAFF2 type for a hydrogen bonded to a heavy atom of the given element. Used
# both for the hydrogens that saturate a matched cap and for a bare-H cap that
# replaces the removed atom directly on the retained atom.
CAP_H_TYPE_FOR_HEAVY = {
    "C": "hc",
    "N": "hn",
    "O": "ho",
    "S": "hs",
    "P": "hp",
}

# Representative base charge for each cap atom type, drawn from typical
# GAFF2/AM1-BCC values. Unlike the legacy scheme, matched cap groups are NOT
# forced to be net neutral -- a polar cap (``OH``, ``NH2``) keeps its real
# polarity and the small integer-balancing residual is routed to whichever cap
# atom can chemically absorb it (see :func:`plan_caps`). A methyl is naturally
# near-neutral (``c3`` + 3x``hc`` ~= 0), so it makes a good charge sink.
CAP_BASE_CHARGE = {
    # hydrogens
    "hc": 0.06,
    "hn": 0.36,
    "ho": 0.43,
    "hs": 0.16,
    "hp": 0.10,
    # carbon caps
    "c3": -0.18,
    "c2": -0.10,
    "c1": -0.10,
    # oxygen caps
    "oh": -0.60,
    "o": -0.45,
    # nitrogen caps
    "n3": -0.90,
    "n2": -0.50,
    "n1": -0.40,
    # sulfur / phosphorus
    "sh": -0.30,
    "s": -0.20,
    "p3": -0.20,
    # halogens
    "f": -0.20,
    "cl": -0.12,
    "br": -0.10,
    "i": -0.08,
}

# Single-bond reference lengths (Angstrom) keyed by an unordered element pair.
# Higher bond orders scale these down via ``_ORDER_SCALE``.
_SINGLE_BOND_LENGTHS = {
    frozenset({"C", "C"}): 1.53,
    frozenset({"C", "N"}): 1.47,
    frozenset({"C", "O"}): 1.43,
    frozenset({"C", "S"}): 1.81,
    frozenset({"C", "P"}): 1.84,
    frozenset({"C", "F"}): 1.35,
    frozenset({"C", "Cl"}): 1.77,
    frozenset({"C", "Br"}): 1.94,
    frozenset({"C", "I"}): 2.14,
    frozenset({"N", "N"}): 1.45,
    frozenset({"N", "O"}): 1.40,
    frozenset({"O", "O"}): 1.48,
    frozenset({"N", "S"}): 1.68,
    frozenset({"O", "S"}): 1.58,
    frozenset({"S", "S"}): 2.05,
}
_ORDER_SCALE = {1: 1.0, 2: 0.87, 3: 0.78}

# X-H bond lengths (Angstrom) by the element the hydrogen attaches to.
_H_BOND_LENGTHS = {"C": 1.09, "N": 1.01, "O": 0.96, "S": 1.34, "P": 1.42}

# Legacy hydroxyl cap parameters (preserved for the ``"hydroxyl"`` strategy).
LEGACY_OH_O_BOND_LENGTHS = {"C": 1.43, "N": 1.40, "O": 1.41, "S": 1.58, "P": 1.58}
LEGACY_OH_BOND_LENGTH = 0.96
LEGACY_OH_O_CHARGE = -0.54
LEGACY_OH_H_CHARGE = 0.54
LEGACY_OH_O_WEIGHT = 0.85
LEGACY_OH_H_WEIGHT = 0.15

# Elements whose removal allows a bare-hydrogen cap (carbon-like). Removing a
# more electronegative heteroatom should recreate it instead.
_H_CAPPABLE_REMOVED = {"C", "H"}

CAP_STRATEGIES = ("chemistry_aware", "hydroxyl", "hydrogen")


def bond_type_to_order(bond_type: str) -> int:
    """Map a stored bond-type label to an integer bond order.

    Args:
        bond_type: Bond label from the source topology (e.g. ``"1"``, ``"2"``,
            ``"ar"``, ``"am"``).

    Returns:
        ``1``, ``2``, or ``3``. Aromatic/amide labels collapse to ``1`` because
        only single bonds are ever cut today.
    """

    normalized = str(bond_type).lower()
    if normalized in {"2", "2.0"}:
        return 2
    if normalized in {"3", "3.0"}:
        return 3
    return 1


def bare_hydrogen_allowed(retained_element: str, removed_element: str) -> bool:
    """Return whether a cut may be capped with a bare hydrogen.

    A bare hydrogen keeps the chemistry faithful only when the retained atom is
    carbon (so the new bond is ``C-H`` and adds no donor) and the removed atom
    is carbon/hydrogen (so no heteroatom electronics are stripped).

    Args:
        retained_element: Element of the atom kept in the fragment.
        removed_element: Element of the atom removed across the cut bond.

    Returns:
        ``True`` when a bare hydrogen is an appropriate cap.
    """

    return retained_element == "C" and removed_element in _H_CAPPABLE_REMOVED


def heavy_cap_bond_length(retained_element: str, cap_element: str, order: int) -> float:
    """Length of the bond from the retained atom to a matched cap heavy atom.

    Args:
        retained_element: Element of the retained atom.
        cap_element: Element of the cap heavy atom (the recreated removed atom).
        order: Bond order of the cut bond.

    Returns:
        A bond length in Angstrom.
    """

    base = _SINGLE_BOND_LENGTHS.get(frozenset({retained_element, cap_element}), 1.50)
    return base * _ORDER_SCALE.get(order, 1.0)


def hydrogen_bond_length(element: str) -> float:
    """Length of a bond from ``element`` to a hydrogen.

    Args:
        element: Element the hydrogen attaches to.

    Returns:
        A bond length in Angstrom.
    """

    return _H_BOND_LENGTHS.get(element, 1.00)


@dataclass
class CapAtom:
    """A single concrete cap atom to be emitted into a fragment.

    Attributes:
        element: Element symbol.
        atom_type: GAFF2 atom type.
        role: ``"heavy"`` for a recreated heavy atom, ``"hydrogen"`` otherwise.
        coords: Cartesian coordinates in Angstrom.
        base_charge: Charge before any integer-balancing delta is applied.
        charge: Final charge (set by :func:`plan_caps`).
        name: Unique atom name within the fragment (set by :func:`plan_caps`).
    """

    element: str
    atom_type: str
    role: str
    coords: np.ndarray
    base_charge: float
    charge: float = 0.0
    name: str = ""


@dataclass
class ResolvedCap:
    """All cap atoms produced for one cut bond, with bonding topology.

    Attributes:
        retained_atom: Parent index of the retained atom this cap attaches to.
        removed_atom: Parent index of the removed atom this cap stands in for.
        parent_bond_order: Order of the bond from the retained atom to the cap
            atom that attaches to it (the heavy atom, or the bare hydrogen).
        heavy: The recreated heavy atom, or ``None`` for a bare-hydrogen cap.
        hydrogens: Hydrogen atoms. They bond to ``heavy`` when present, else the
            single hydrogen bonds directly to the retained atom.
        cap_label: Human-readable cap formula (e.g. ``"bare_hydrogen"``,
            ``"CH3"``, ``"OH"``, ``"=NH"``).
        reason: Why this cap was chosen (see :func:`plan_caps`).
        bare_h_charge: The bare-hydrogen charge that drove the decision -- the
            actual charge when a bare hydrogen was used, the hypothetical charge
            that triggered escalation, or ``None`` when a bare hydrogen was never
            eligible.
    """

    retained_atom: int
    removed_atom: int
    parent_bond_order: int
    heavy: CapAtom | None
    hydrogens: list[CapAtom] = field(default_factory=list)
    cap_label: str = ""
    reason: str = ""
    bare_h_charge: float | None = None

    @property
    def atoms(self) -> list[CapAtom]:
        """Return all cap atoms, heavy atom first when present."""

        return ([self.heavy] if self.heavy is not None else []) + self.hydrogens


def _orthonormal_frame(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a right-handed orthonormal frame whose first vector is ``axis``.

    Args:
        axis: A non-zero direction vector.

    Returns:
        A tuple ``(a, u, v)`` of unit vectors with ``a`` parallel to ``axis``.
    """

    norm = np.linalg.norm(axis)
    a = axis / norm if norm > 1.0e-8 else np.array([1.0, 0.0, 0.0])
    reference = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = reference - np.dot(reference, a) * a
    u_norm = np.linalg.norm(u)
    u = u / u_norm if u_norm > 1.0e-8 else np.array([0.0, 1.0, 0.0])
    v = np.cross(a, u)
    return a, u, v


def place_cap_hydrogens(
    parent_pos: np.ndarray,
    heavy_pos: np.ndarray,
    order: int,
    n_hydrogens: int,
    h_length: float,
) -> list[np.ndarray]:
    """Place saturating hydrogens around a matched cap heavy atom.

    Idealized geometry is used: tetrahedral (~109.5 deg) for a single bond to
    the retained atom, trigonal (120 deg) for a double bond, and linear for a
    triple bond. Coordinates are not minimized.

    Args:
        parent_pos: Coordinate of the retained atom.
        heavy_pos: Coordinate of the cap heavy atom.
        order: Bond order between the retained atom and the cap heavy atom.
        n_hydrogens: Number of hydrogens to place.
        h_length: Heavy-atom-to-hydrogen bond length.

    Returns:
        A list of hydrogen coordinates.
    """

    if n_hydrogens <= 0:
        return []
    a, u, v = _orthonormal_frame(parent_pos - heavy_pos)
    directions: list[np.ndarray] = []
    if order >= 3:
        directions = [-a for _ in range(n_hydrogens)]
    elif order == 2:
        cos_t, sin_t = -0.5, math.sqrt(3.0) / 2.0
        signs = [1.0, -1.0, 1.0]
        for k in range(n_hydrogens):
            directions.append(cos_t * a + sin_t * signs[k % len(signs)] * u)
    else:
        cos_t = -1.0 / 3.0
        sin_t = math.sqrt(1.0 - cos_t * cos_t)
        for k in range(n_hydrogens):
            phi = math.pi + k * 2.0 * math.pi / 3.0
            directions.append(cos_t * a + sin_t * (math.cos(phi) * u + math.sin(phi) * v))
    coords: list[np.ndarray] = []
    for direction in directions:
        norm = np.linalg.norm(direction)
        unit = direction / norm if norm > 1.0e-8 else a
        coords.append(heavy_pos + unit * h_length)
    return coords


def _build_bare_hydrogen(
    retained_atom: int,
    removed_atom: int,
    retained_element: str,
    parent_pos: np.ndarray,
    direction: np.ndarray,
) -> ResolvedCap:
    """Build a single-hydrogen cap attached directly to the retained atom."""

    h_type = CAP_H_TYPE_FOR_HEAVY.get(retained_element, "hc")
    coord = parent_pos + direction * hydrogen_bond_length(retained_element)
    base = CAP_BASE_CHARGE.get(h_type, 0.06)
    hydrogen = CapAtom(element="H", atom_type=h_type, role="hydrogen", coords=coord, base_charge=base)
    return ResolvedCap(retained_atom, removed_atom, 1, heavy=None, hydrogens=[hydrogen])


def _build_matched(
    retained_atom: int,
    removed_atom: int,
    retained_element: str,
    removed_element: str,
    order: int,
    parent_pos: np.ndarray,
    direction: np.ndarray,
) -> ResolvedCap:
    """Build an element-matched cap recreating the removed atom plus hydrogens."""

    heavy_types = CAP_HEAVY_TYPES.get(removed_element, {1: "du"})
    heavy_type = heavy_types.get(order, heavy_types[min(heavy_types)])
    heavy_pos = parent_pos + direction * heavy_cap_bond_length(retained_element, removed_element, order)
    n_hydrogens = max(STANDARD_VALENCE.get(removed_element, 1) - order, 0)
    h_type = CAP_H_TYPE_FOR_HEAVY.get(removed_element, "hc")
    h_base = CAP_BASE_CHARGE.get(h_type, 0.06)
    heavy = CapAtom(
        element=removed_element,
        atom_type=heavy_type,
        role="heavy",
        coords=heavy_pos,
        base_charge=CAP_BASE_CHARGE.get(heavy_type, 0.0),
    )
    h_coords = place_cap_hydrogens(parent_pos, heavy_pos, order, n_hydrogens, hydrogen_bond_length(removed_element))
    hydrogens = [
        CapAtom(element="H", atom_type=h_type, role="hydrogen", coords=coord, base_charge=h_base)
        for coord in h_coords
    ]
    return ResolvedCap(retained_atom, removed_atom, order, heavy=heavy, hydrogens=hydrogens)


def _build_legacy_oh(
    retained_atom: int,
    removed_atom: int,
    retained_element: str,
    parent_pos: np.ndarray,
    direction: np.ndarray,
) -> ResolvedCap:
    """Build a legacy ``-OH`` cap with the original geometry and base charges."""

    oxygen_pos = parent_pos + direction * LEGACY_OH_O_BOND_LENGTHS.get(retained_element, 1.43)
    hydrogen_pos = oxygen_pos + direction * LEGACY_OH_BOND_LENGTH
    oxygen = CapAtom("O", "oh", "heavy", oxygen_pos, base_charge=LEGACY_OH_O_CHARGE)
    hydrogen = CapAtom("H", "ho", "hydrogen", hydrogen_pos, base_charge=LEGACY_OH_H_CHARGE)
    return ResolvedCap(retained_atom, removed_atom, 1, heavy=oxygen, hydrogens=[hydrogen])


def _assign_unique_name(cap_atom: CapAtom, existing_names: set[str], counters: dict[str, int]) -> None:
    """Assign a unique fragment-local name to ``cap_atom`` in place."""

    prefix = "HX" if cap_atom.role == "hydrogen" else f"{cap_atom.element.upper()}X"
    counter = counters.get(prefix, 0) + 1
    while True:
        name = f"{prefix}{counter:02d}"
        if name not in existing_names:
            break
        counter += 1
    counters[prefix] = counter
    existing_names.add(name)
    cap_atom.name = name


def _cap_label(element: str, n_hydrogens: int, order: int) -> str:
    """Build a human-readable cap formula, e.g. ``CH3``, ``OH``, ``=NH``.

    Args:
        element: Element of the cap heavy atom.
        n_hydrogens: Number of saturating hydrogens.
        order: Bond order from the retained atom to the cap heavy atom.

    Returns:
        A short formula string, prefixed with ``=`` or ``#`` for double/triple
        bonds.
    """

    prefix = {2: "=", 3: "#"}.get(order, "")
    if n_hydrogens <= 0:
        suffix = ""
    elif n_hydrogens == 1:
        suffix = "H"
    else:
        suffix = f"H{n_hydrogens}"
    return f"{prefix}{element}{suffix}"


def _distribute(total: float, count: int) -> list[float]:
    """Split ``total`` into ``count`` near-equal parts (remainder on the last)."""

    if count <= 0:
        return []
    parts = [total / count] * count
    parts[-1] += total - sum(parts)
    return parts


def plan_caps(
    cap_sites,
    element_of,
    position_of,
    direction_of,
    strategy: str,
    retained_net_charge: float,
    existing_names: set[str],
    h_min_charge: float = 0.0,
    steric_rank_of=None,
    force_matched_of=None,
) -> tuple[list[ResolvedCap], float]:
    """Resolve every cut site into concrete, charged, named cap atoms.

    Cap atoms are given representative base charges (see :data:`CAP_BASE_CHARGE`)
    and the small residual needed for an integer fragment charge is routed to the
    cap that can best absorb it: heteroatom caps first, then carbon caps, and
    hydrogens only as a last resort. A carbon cut defaults to a bare hydrogen and
    is escalated to a methyl charge sink only when no other sink exists and a bare
    hydrogen would fall below ``h_min_charge``. When several carbon cuts could
    become the sink, the sterically roomiest one (highest ``steric_rank_of``) is
    chosen so bulky caps land where there is room and hydrogens stay where space
    is tight. A cut may also be forced to a matched cap by ``force_matched_of`` to
    preserve the chemistry around the fitted torsion (e.g. an atom in or next to
    the dihedral, or an sp2/conjugated centre, must keep its substituent rather
    than be stripped to an H).

    Args:
        cap_sites: Iterable of objects exposing ``retained_atom``,
            ``removed_atom`` and ``bond_type`` (typically
            :class:`~scission.Models.CapSite`).
        element_of: Callable mapping a parent atom index to its element.
        position_of: Callable mapping a parent atom index to its coordinate.
        direction_of: Callable mapping ``(retained, removed)`` to the unit cap
            direction (typically :func:`scission.Screen.cap_direction`).
        strategy: One of :data:`CAP_STRATEGIES`.
        retained_net_charge: Net charge of the retained parent atoms.
        existing_names: Atom names already used in the fragment; updated in place
            as cap names are assigned.
        h_min_charge: Minimum allowed charge for a bare-hydrogen cap.
        steric_rank_of: Optional callable mapping a cap site to a steric score
            (higher = roomier) used to pick which carbon cut becomes a methyl
            sink. When omitted, sites are escalated in index order.
        force_matched_of: Optional callable mapping a cap site to a reason string
            (or ``None``). When it returns a reason, an otherwise bare-hydrogen
            cut is forced to a matched cap and the reason is recorded.

    Returns:
        A tuple of the resolved caps and the resulting fragment net charge
        (guaranteed to equal ``round(retained_net_charge)``). Each
        :class:`ResolvedCap` records why it was chosen via its ``reason`` field,
        one of: ``"bare_hydrogen"``, ``"charge_escalation"`` (a carbon cut made a
        methyl sink so an integer charge is reached without a negative hydrogen),
        ``"heteroatom_severed"``, ``"retained_not_carbon"``,
        ``"near_fitted_torsion"`` / ``"conjugated_center"`` (forced matched to
        preserve the torsion's environment), ``"forced_hydrogen_strategy"``, or
        ``"legacy_hydroxyl_strategy"``.
    """

    sites = sorted(cap_sites, key=lambda site: (site.retained_atom, site.removed_atom))
    target = round(retained_net_charge)
    total_delta = target - retained_net_charge

    # Per-site kind: "legacy_oh", "bare", "matched" (mandatory heteroatom or
    # forced to preserve the torsion environment), or "sink" (a carbon cut
    # escalated to a methyl charge sink).
    escalation_charge: dict[int, float] = {}
    forced_reason: dict[int, str] = {}
    if strategy == "hydroxyl":
        kinds = ["legacy_oh"] * len(sites)
    elif strategy == "hydrogen":
        kinds = ["bare"] * len(sites)
    else:  # chemistry_aware
        kinds = []
        for i, s in enumerate(sites):
            eligible = bare_hydrogen_allowed(element_of(s.retained_atom), element_of(s.removed_atom))
            reason = force_matched_of(s) if (eligible and force_matched_of is not None) else None
            if eligible and reason:
                kinds.append("matched")
                forced_reason[i] = reason
            elif eligible:
                kinds.append("bare")
            else:
                kinds.append("matched")
        kinds = _resolve_chemistry_aware_kinds(
            sites, kinds, element_of, total_delta, h_min_charge, steric_rank_of, escalation_charge
        )

    resolved: list[ResolvedCap] = []
    for kind, site in zip(kinds, sites):
        retained_element = element_of(site.retained_atom)
        parent_pos = np.asarray(position_of(site.retained_atom), dtype=float)
        direction = np.asarray(direction_of(site.retained_atom, site.removed_atom), dtype=float)
        if kind == "legacy_oh":
            cap = _build_legacy_oh(site.retained_atom, site.removed_atom, retained_element, parent_pos, direction)
        elif kind == "bare":
            cap = _build_bare_hydrogen(site.retained_atom, site.removed_atom, retained_element, parent_pos, direction)
        else:  # "matched" or "sink" -- both recreate the removed atom
            cap = _build_matched(
                site.retained_atom,
                site.removed_atom,
                retained_element,
                element_of(site.removed_atom),
                bond_type_to_order(site.bond_type),
                parent_pos,
                direction,
            )
        resolved.append(cap)

    if strategy == "hydroxyl":
        _assign_charges_hydroxyl(resolved, total_delta)
    else:
        _assign_charges_sink(resolved, total_delta)

    counters: dict[str, int] = {}
    for cap in resolved:
        for atom in cap.atoms:
            _assign_unique_name(atom, existing_names, counters)

    # Record the decision rationale on each cap.
    for i, (site, cap, kind) in enumerate(zip(sites, resolved, kinds)):
        if kind == "bare":
            cap.cap_label = "bare_hydrogen"
            cap.bare_h_charge = cap.hydrogens[0].charge
            cap.reason = "forced_hydrogen_strategy" if strategy == "hydrogen" else "bare_hydrogen"
        else:
            cap.cap_label = _cap_label(cap.heavy.element, len(cap.hydrogens), cap.parent_bond_order)
            if kind == "legacy_oh":
                cap.reason = "legacy_hydroxyl_strategy"
            elif kind == "sink":
                cap.reason = "charge_escalation"
                cap.bare_h_charge = escalation_charge.get(i)
            elif i in forced_reason:
                cap.reason = forced_reason[i]
            elif element_of(site.retained_atom) != "C":
                cap.reason = "retained_not_carbon"
            else:
                cap.reason = "heteroatom_severed"

    fragment_net = retained_net_charge + sum(atom.charge for cap in resolved for atom in cap.atoms)
    return resolved, fragment_net


def _kind_base_sum(sites, kinds, element_of) -> float:
    """Sum the representative base charges implied by a tentative cap config."""

    total = 0.0
    for kind, site in zip(kinds, sites):
        if kind == "bare":
            total += CAP_BASE_CHARGE.get(CAP_H_TYPE_FOR_HEAVY.get(element_of(site.retained_atom), "hc"), 0.06)
            continue
        # "matched" (heteroatom) or "sink" (methyl): heavy atom plus hydrogens.
        removed = element_of(site.removed_atom)
        order = bond_type_to_order(site.bond_type)
        heavy_types = CAP_HEAVY_TYPES.get(removed, {1: "du"})
        heavy_type = heavy_types.get(order, heavy_types[min(heavy_types)])
        n_h = max(STANDARD_VALENCE.get(removed, 1) - order, 0)
        h_type = CAP_H_TYPE_FOR_HEAVY.get(removed, "hc")
        total += CAP_BASE_CHARGE.get(heavy_type, 0.0) + n_h * CAP_BASE_CHARGE.get(h_type, 0.06)
    return total


def _resolve_chemistry_aware_kinds(
    sites, kinds, element_of, total_delta, h_min_charge, steric_rank_of, escalation_charge
):
    """Decide which carbon cuts stay bare H and which become methyl charge sinks.

    Heteroatom cuts are always matched. A carbon cut is escalated to a methyl
    sink only when no heavy sink exists and the residual would drag a bare
    hydrogen below ``h_min_charge``; the sterically roomiest carbon cut is chosen
    each time.
    """

    kinds = list(kinds)
    steric_cache: dict[int, float] = {}

    def rank(i: int) -> float:
        if steric_rank_of is None:
            return -i  # deterministic: prefer earliest site
        if i not in steric_cache:
            steric_cache[i] = steric_rank_of(sites[i])
        return steric_cache[i]

    while True:
        if any(kind in ("matched", "sink") for kind in kinds):
            break  # a heavy sink exists -> the residual lands there, H's stay at base
        bare_indices = [i for i, kind in enumerate(kinds) if kind == "bare"]
        if not bare_indices:
            break
        residual = total_delta - _kind_base_sum(sites, kinds, element_of)
        would_be = CAP_BASE_CHARGE.get("hc", 0.06) + residual / len(bare_indices)
        if would_be >= h_min_charge:
            break  # bare hydrogens stay non-negative; keep them all
        pick = max(bare_indices, key=rank)
        escalation_charge[pick] = would_be
        kinds[pick] = "sink"
    return kinds


def _assign_charges_hydroxyl(resolved: list[ResolvedCap], total_delta: float) -> None:
    """Legacy OH charge assignment: even per-pair split, oxygen-weighted."""

    deltas = _distribute(total_delta, len(resolved))
    for cap, delta in zip(resolved, deltas):
        cap.heavy.charge = cap.heavy.base_charge + LEGACY_OH_O_WEIGHT * delta
        cap.hydrogens[0].charge = cap.hydrogens[0].base_charge + LEGACY_OH_H_WEIGHT * delta


def _assign_charges_sink(resolved: list[ResolvedCap], total_delta: float) -> None:
    """Assign representative base charges, then route the residual to the best sink.

    Every cap atom starts at its representative base charge. The residual needed
    to reach an integer fragment charge is distributed across the highest-priority
    available sink tier -- heteroatom heavy atoms, else carbon heavy atoms, else
    hydrogens -- so polar caps keep their charge and hydrogens stay reasonable.
    """

    for cap in resolved:
        for atom in cap.atoms:
            atom.charge = atom.base_charge

    base_sum = sum(atom.charge for cap in resolved for atom in cap.atoms)
    residual = total_delta - base_sum

    heavy_hetero = [a for cap in resolved for a in cap.atoms if a.role == "heavy" and a.element in {"N", "O", "S", "P"}]
    heavy_carbon = [a for cap in resolved for a in cap.atoms if a.role == "heavy" and a.element == "C"]
    hydrogens = [a for cap in resolved for a in cap.atoms if a.role == "hydrogen"]
    sinks = heavy_hetero or heavy_carbon or hydrogens
    if not sinks:
        return
    for atom, share in zip(sinks, _distribute(residual, len(sinks))):
        atom.charge += share
