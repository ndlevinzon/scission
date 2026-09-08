from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Atom:
    """Simple atom record parsed from the ligand input files.

    Attributes:
        index: One-based atom index from the parent MOL2 file.
        name: Atom name used in the parent topology.
        element: Inferred element symbol.
        atom_type: Force-field atom type label.
        charge: Partial atomic charge.
        coords: Cartesian coordinates in Angstrom.
        residue_name: Residue name associated with the atom.
        residue_id: Residue index associated with the atom.
        is_cap: Whether the atom was introduced as a fragment cap.
    """

    index: int
    name: str
    element: str
    atom_type: str
    charge: float
    coords: tuple[float, float, float]
    residue_name: str = "LIG"
    residue_id: int = 1
    is_cap: bool = False


@dataclass(frozen=True)
class Bond:
    """Bond record using the parent ligand atom indices.

    Attributes:
        index: One-based bond index from the source file.
        atom1: One-based index of the first bonded atom.
        atom2: One-based index of the second bonded atom.
        bond_type: Bond order/type label as read from the input.
    """

    index: int
    atom1: int
    atom2: int
    bond_type: str


@dataclass(frozen=True)
class CapSite:
    """A single cut bond that must be capped on the retained side.

    Attributes:
        retained_atom: Parent index of the atom kept in the fragment.
        removed_atom: Parent index of the bonded atom removed across the cut.
        bond_type: Bond-type label of the cut bond, used to choose the cap
            element and bond order.
    """

    retained_atom: int
    removed_atom: int
    bond_type: str


@dataclass(frozen=True)
class TorsionDefinition:
    """Named four-atom torsion anchored to a rotatable bond.

    Attributes:
        atom_indices: Parent atom indices defining the dihedral.
        bond: The central rotatable bond as a normalized atom pair.
        label: Human-readable torsion label built from atom names.
    """

    atom_indices: tuple[int, int, int, int]
    bond: tuple[int, int]
    label: str


@dataclass
class Ligand:
    """In-memory ligand model plus the source metadata needed for outputs.

    Attributes:
        name: Ligand name used for emitted files.
        atoms: Parent atom records in source order.
        bonds: Parent bond records in source order.
        lib_atom_names: Atom names parsed from the Amber ``.lib`` file.
        lib_atom_types: Amber atom types keyed by atom name.
        frcmod_text: Raw ``.frcmod`` contents.
        mol2_path: Source MOL2 path.
        lib_path: Source Amber LIB path.
        frcmod_path: Source FRCMOD path.
    """

    name: str
    atoms: list[Atom]
    bonds: list[Bond]
    lib_atom_names: list[str]
    lib_atom_types: dict[str, str]
    frcmod_text: str
    mol2_path: Path
    lib_path: Path
    frcmod_path: Path
    # Topology caches (filled lazily; not part of construction / equality).
    _coordinates: dict[int, np.ndarray] | None = field(
        default=None, init=False, repr=False, compare=False
    )
    _graph: Any = field(default=None, init=False, repr=False, compare=False)
    _frag_topology: Any = field(default=None, init=False, repr=False, compare=False)
    _rotatable_bonds_cache: Any = field(default=None, init=False, repr=False, compare=False)
    _ring_edges: Any = field(default=None, init=False, repr=False, compare=False)
    _rdkit_mol: Any = field(default=None, init=False, repr=False, compare=False)
    _distance_maps: Any = field(default=None, init=False, repr=False, compare=False)

    def atom(self, atom_index: int) -> Atom:
        """Return a parent atom record by its one-based index.

        Args:
            atom_index: One-based atom index from the ligand topology.

        Returns:
            The matching :class:`Atom` record.
        """

        return self.atoms[atom_index - 1]

    @property
    def coordinates(self) -> dict[int, np.ndarray]:
        """Return parent coordinates keyed by one-based atom index.

        The mapping is built once and reused. Callers that mutate coordinates
        (e.g. torsion screening) must ``.copy()`` individual arrays.
        """

        if self._coordinates is None:
            self._coordinates = {
                atom.index: np.asarray(atom.coords, dtype=float) for atom in self.atoms
            }
        return self._coordinates

    def clear_geometry_caches(self) -> None:
        """Drop cached coordinates / graph after topology or coords change."""

        self._coordinates = None
        self._graph = None
        self._frag_topology = None
        self._rotatable_bonds_cache = None
        self._ring_edges = None
        self._rdkit_mol = None
        self._distance_maps = None


@dataclass(frozen=True)
class ClashThresholds:
    """Scale factors used when screening fragments for steric clashes.

    Attributes:
        path3_scale: Scale factor for atom pairs separated by three bonds.
        far_scale: Scale factor for more distant nonbonded pairs.
    """

    path3_scale: float = 0.60
    far_scale: float = 0.60


@dataclass(frozen=True)
class FragmentConfig:
    """Configuration options that control fragment generation and selection.

    Attributes:
        angle_step: Torsion sampling interval in degrees.
        torsion_scope: Label describing which torsions to consider.
        include_rigid_single_bonds: Whether to include acyclic single bonds
            that would normally be excluded as amide-like and treated as
            rigid by stricter torsion heuristics.
        rotatable_bond_smarts: SMARTS patterns that may nominate otherwise
            excluded bonds as torsion targets. Each pattern must mark the
            central bond atoms with atom-map numbers ``:1`` and ``:2``.
        restrict_to_bond_smarts: SMARTS patterns that restrict fragmentation
            to only the torsions whose central bond matches one of the
            patterns. Each pattern must mark the central bond atoms with
            atom-map numbers ``:1`` and ``:2``. Empty (the default) means no
            restriction - every enumerated rotatable torsion is fragmented.
            When non-empty, rotatable torsions whose central bond matches no
            pattern are dropped before fragment selection and reported under
            ``rejected_torsions``. This is an allow-list applied *after*
            ``rotatable_bond_smarts`` nomination, so the two compose: extra
            bonds added by ``rotatable_bond_smarts`` are still subject to the
            restriction.
        cap_strategy: Strategy used to cap cut bonds. One of
            ``"chemistry_aware"`` (default; prefer a bare hydrogen, fall back to
            an element-matched group such as ``CH3``/``OH``/``NH2`` when the cap
            hydrogen would carry a negative/low charge or a heteroatom was
            severed), ``"hydroxyl"`` (legacy ``-OH`` everywhere), or
            ``"hydrogen"`` (always a bare hydrogen).
        cap_h_min_charge: Minimum allowed partial charge for a bare-hydrogen
            cap under ``"chemistry_aware"``. A bare hydrogen whose balanced
            charge would fall below this escalates to an element-matched cap.
        preserve_torsion_neighborhood: When true (default), a carbon cut whose
            retained atom is in, or within ``torsion_neighborhood_radius`` bonds
            of, the fitted dihedral is forced to a matched cap instead of a bare
            hydrogen, so the steric/conjugation environment that sets the
            torsional barrier is preserved. Only affects ``"chemistry_aware"``.
        torsion_neighborhood_radius: Bond radius of the protected shell around
            the fitted dihedral atoms (``0`` = the dihedral atoms only, ``1`` =
            also their direct neighbors, ``2`` = the next shell, ...).
        preserve_conjugated_caps: When true (default), a carbon cut whose
            retained atom is an sp2/conjugated centre (has a double or aromatic
            bond) is also forced to a matched cap. Gated by
            ``preserve_torsion_neighborhood``.
        optimization_target: Selection heuristic name.
        clash_thresholds: Clash-screening thresholds.
        preserve_conjugated_neighbors: Whether to bias toward preserving
            conjugated neighborhoods near torsions.
        use_parent_fallback: Whether full-parent fallback fragments count as
            valid coverage during selection.
        nproc: Worker budget for torsion screening (process pool) and
            per-fragment Amber writes (thread pool). Default 1 (serial).
    """

    angle_step: int = 30
    torsion_scope: str = "acyclic_rotatable"
    include_rigid_single_bonds: bool = True
    rotatable_bond_smarts: tuple[str, ...] = ()
    restrict_to_bond_smarts: tuple[str, ...] = ()
    cap_strategy: str = "chemistry_aware"
    cap_h_min_charge: float = 0.0
    preserve_torsion_neighborhood: bool = True
    torsion_neighborhood_radius: int = 1
    preserve_conjugated_caps: bool = True
    optimization_target: str = "fewest_total_fragments"
    clash_thresholds: ClashThresholds = field(default_factory=ClashThresholds)
    preserve_conjugated_neighbors: bool = True
    use_parent_fallback: bool = False
    nproc: int = 1

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FragmentConfig":
        """Build a configuration object from a plain mapping.

        Args:
            payload: Parsed configuration dictionary, typically loaded from
                YAML or JSON.

        Returns:
            A populated :class:`FragmentConfig` instance.
        """

        thresholds = payload.get("clash_thresholds", {})
        rotatable_bond_smarts = payload.get("rotatable_bond_smarts", [])
        if isinstance(rotatable_bond_smarts, str):
            rotatable_bond_smarts = [rotatable_bond_smarts]
        restrict_to_bond_smarts = payload.get("restrict_to_bond_smarts", [])
        if isinstance(restrict_to_bond_smarts, str):
            restrict_to_bond_smarts = [restrict_to_bond_smarts]
        return cls(
            angle_step=payload.get("angle_step", 30),
            torsion_scope=payload.get("torsion_scope", "acyclic_rotatable"),
            include_rigid_single_bonds=payload.get(
                "include_rigid_single_bonds",
                True,
            ),
            rotatable_bond_smarts=tuple(str(pattern) for pattern in rotatable_bond_smarts),
            restrict_to_bond_smarts=tuple(str(pattern) for pattern in restrict_to_bond_smarts),
            cap_strategy=payload.get("cap_strategy", "chemistry_aware"),
            cap_h_min_charge=payload.get("cap_h_min_charge", 0.0),
            preserve_torsion_neighborhood=payload.get("preserve_torsion_neighborhood", True),
            torsion_neighborhood_radius=payload.get("torsion_neighborhood_radius", 1),
            preserve_conjugated_caps=payload.get("preserve_conjugated_caps", True),
            optimization_target=payload.get(
                "optimization_target",
                "fewest_total_fragments",
            ),
            clash_thresholds=ClashThresholds(
                path3_scale=thresholds.get("path3_scale", 0.60),
                far_scale=thresholds.get("far_scale", 0.60),
            ),
            preserve_conjugated_neighbors=payload.get(
                "preserve_conjugated_neighbors",
                True,
            ),
            use_parent_fallback=payload.get("use_parent_fallback", False),
            nproc=int(payload.get("nproc", 1)),
        )


@dataclass
class CandidateFragment:
    """Intermediate fragment candidate before any files are written.

    Attributes:
        candidate_id: Stable identifier derived from retained atoms and cuts.
        retained_atoms: Parent atom indices kept in the fragment.
        cut_bonds: Parent heavy-atom bonds cut to isolate the fragment.
        cap_sites: Cut bonds to cap, each recording the retained atom, the
            removed bonded atom (which defines the cap direction and element),
            and the cut bond type.
        shell_level: Breadth of the domain-shell expansion used to create the
            candidate.
        net_charge: Net charge of the retained parent atoms.
        ring_count: Number of retained rigid domains containing ring atoms.
        ring_cap_count: Number of cap sites attached to ring-containing domains.
        is_parent_fallback: Whether the candidate is the full parent fallback.
        torsion_labels: Torsion labels associated with this candidate.
        worst_margin: Worst nonbonded margin observed during screening.
        accepted_torsions: Torsions that passed screening for this candidate.
    """

    candidate_id: str
    retained_atoms: set[int]
    cut_bonds: set[tuple[int, int]]
    cap_sites: list[CapSite]
    shell_level: int
    net_charge: float = 0.0
    ring_count: int = 0
    ring_cap_count: int = 0
    is_parent_fallback: bool = False
    torsion_labels: set[str] = field(default_factory=set)
    worst_margin: float | None = None
    accepted_torsions: set[str] = field(default_factory=set)


@dataclass
class SelectedFragment:
    """Materialized fragment together with generated file paths and metadata.

    Attributes:
        fragment_id: Stable identifier for the selected fragment.
        source_candidate_id: Original internal candidate identifier derived from
            retained atoms and cut bonds.
        retained_atoms: Retained parent atom indices in sorted order.
        cut_bonds: Parent cut bonds represented as normalized pairs.
        cap_atoms: Serialized cap atom records emitted to the fragment files.
        cap_decisions: Per-cut records describing the cap chosen and why.
        torsions: Torsion labels assigned to this fragment.
        fit_torsions: Serialized torsion-mapping records for downstream fitting.
        parent_atom_map: Mapping from parent atom index to fragment-local atom
            index.
        net_charge: Final fragment net charge after capping.
        retained_net_charge: Charge of retained parent atoms before capping.
        is_parent_fallback: Whether the selected fragment is the parent
            fallback.
        manifest_path: Path to the fragment manifest JSON.
        mol2_path: Path to the fragment MOL2 file.
        xyz_path: Path to the fragment XYZ file.
        lib_path: Path to the fragment Amber LIB file.
        frcmod_path: Path to the fragment FRCMOD file.
        parm7_path: Path to the generated Amber topology, if available.
        rst7_path: Path to the generated Amber restart file, if available.
        overview_image_path: Path to the overview SVG drawing, if available.
        torsion_image_paths: Mapping of torsion label to SVG path.
        worst_margin: Worst clash-screen margin observed for the fragment.
    """

    fragment_id: str
    source_candidate_id: str
    retained_atoms: list[int]
    cut_bonds: list[tuple[int, int]]
    cap_atoms: list[dict[str, Any]]
    torsions: list[str]
    fit_torsions: list[dict[str, Any]]
    parent_atom_map: dict[int, int]
    cap_decisions: list[dict[str, Any]] = field(default_factory=list)
    net_charge: float = 0.0
    retained_net_charge: float = 0.0
    is_parent_fallback: bool = False
    manifest_path: Path | None = None
    mol2_path: Path | None = None
    xyz_path: Path | None = None
    lib_path: Path | None = None
    frcmod_path: Path | None = None
    parm7_path: Path | None = None
    rst7_path: Path | None = None
    overview_image_path: Path | None = None
    torsion_image_paths: dict[str, Path] = field(default_factory=dict)
    worst_margin: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize the fragment and normalize any paths to strings.

        Returns:
            A JSON-friendly dictionary representation of the fragment.
        """

        payload = asdict(self)
        for key in (
            "manifest_path",
            "mol2_path",
            "xyz_path",
            "lib_path",
            "frcmod_path",
            "parm7_path",
            "rst7_path",
            "overview_image_path",
        ):
            if payload[key] is not None:
                payload[key] = str(payload[key])
        payload["torsion_image_paths"] = {
            label: str(path) for label, path in self.torsion_image_paths.items()
        }
        return payload


@dataclass
class FragmentationResult:
    """Top-level result bundle produced by the fragmentation pipeline.

    Attributes:
        selected_fragments: Fragments chosen to cover the accepted torsions.
        covered_torsions: Mapping from torsion label to fragment identifier.
        rejected_torsions: Mapping from torsion label to rejection details.
        warnings: Non-fatal warnings encountered during the run.
        output_paths: Mapping from fragment identifier to its output directory.
        summary_path: Path to the run-level summary JSON, if written.
        fragment_index_path: Path to the top-level fragment index JSON, if
            written.
    """

    selected_fragments: list[SelectedFragment]
    covered_torsions: dict[str, str]
    rejected_torsions: dict[str, str]
    warnings: list[str]
    output_paths: dict[str, str]
    summary_path: Path | None = None
    fragment_index_path: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize the pipeline result into JSON-friendly primitives.

        Returns:
            A JSON-serializable dictionary representation of the run result.
        """

        return {
            "selected_fragments": [fragment.to_dict() for fragment in self.selected_fragments],
            "covered_torsions": self.covered_torsions,
            "rejected_torsions": self.rejected_torsions,
            "warnings": self.warnings,
            "output_paths": self.output_paths,
            "summary_path": str(self.summary_path) if self.summary_path else None,
            "fragment_index_path": str(self.fragment_index_path) if self.fragment_index_path else None,
        }


@dataclass(frozen=True)
class InputBundle:
    """Paths describing a ligand input triplet to process.

    Attributes:
        mol2_path: Path to the parent MOL2 structure.
        lib_path: Path to the Amber LIB file.
        frcmod_path: Path to the Amber FRCMOD file.
        ligand_name: Optional override for the ligand name.
    """

    mol2_path: Path
    lib_path: Path
    frcmod_path: Path
    ligand_name: str | None = None
