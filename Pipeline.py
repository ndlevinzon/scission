from __future__ import annotations

from collections import defaultdict
from pathlib import Path

try:
    import parmed  # noqa: F401
except ImportError:  # pragma: no cover
    parmed = None

try:
    import rdkit  # noqa: F401
except ImportError:  # pragma: no cover
    rdkit = None

from .LigandIo import load_ligand
from .Models import FragmentConfig, FragmentationResult, InputBundle
from .Optimize import select_fragments
from .Parallel import screen_torsions, write_selected_fragments
from .Torsions import enumerate_torsions, match_central_bond_smarts
from .Writers import write_fragment_index, write_summary


def fragment_ligand(
    bundle: InputBundle,
    out_dir: Path,
    config: FragmentConfig,
) -> FragmentationResult:
    """Run the full ligand fragmentation workflow for one input bundle.

    When ``config.nproc > 1``, torsion screening is pooled over torsions and
    per-fragment ``parmchk2`` / ``tleap`` writes use a thread pool. Selection
    and summary writes remain serial.

    Args:
        bundle: Input ligand file bundle to load and validate.
        out_dir: Output directory where fragment artifacts should be written.
        config: Fragmentation and screening configuration.

    Returns:
        A :class:`FragmentationResult` describing the selected fragments and
        any rejected torsions.
    """

    ligand = load_ligand(bundle)
    out_dir.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []
    if parmed is None:
        warnings.append("ParmEd not installed; using pure-Python parser/writer fallback.")
    if rdkit is None:
        warnings.append("RDKit not installed; using pure-Python torsion heuristics.")

    torsions = enumerate_torsions(
        ligand,
        include_rigid_single_bonds=config.include_rigid_single_bonds,
        rotatable_bond_smarts=config.rotatable_bond_smarts,
    )
    torsion_map = {torsion.label: torsion for torsion in torsions}
    rejected_torsions: dict[str, object] = {}

    # When restrict_to_bond_smarts is set, keep only the torsions whose central
    # bond matches one of the allow-list patterns; the rest are dropped before
    # any fragment is built and reported as rejected. Restriction composes with
    # rotatable_bond_smarts (which nominates extra bonds): nomination decides
    # what is rotatable, restriction decides which of those to actually fit.
    if config.restrict_to_bond_smarts:
        allowed_bonds = match_central_bond_smarts(
            ligand, config.restrict_to_bond_smarts
        )
        kept_torsions = []
        for torsion in torsions:
            if torsion.bond in allowed_bonds:
                kept_torsions.append(torsion)
            else:
                rejected_torsions[torsion.label] = "excluded_by_restrict_to_bond_smarts"
        if not kept_torsions:
            warnings.append(
                "restrict_to_bond_smarts matched no rotatable torsions; "
                "no fragments will be generated."
            )
        torsions = kept_torsions
        torsion_map = {torsion.label: torsion for torsion in torsions}

    candidate_pool, accepted_by_torsion, screen_rejected = screen_torsions(
        ligand, torsions, config
    )
    rejected_torsions.update(screen_rejected)

    preferred_candidate_ids: set[str] = set()
    for torsion_label, candidates in accepted_by_torsion.items():
        preferred = min(
            candidates,
            key=lambda candidate: (
                -candidate.ring_cap_count,
                len(candidate.retained_atoms),
                candidate.ring_count,
                -(candidate.worst_margin if candidate.worst_margin is not None else -999.0),
                len(candidate.cut_bonds),
                candidate.candidate_id,
            ),
        )
        preferred_candidate_ids.add(preferred.candidate_id)

    selected_candidates, covered_torsions, uncovered = select_fragments(
        candidate_pool,
        [torsion.label for torsion in torsions if torsion.label not in rejected_torsions],
        allow_parent_fallback=config.use_parent_fallback,
        preferred_candidate_ids=preferred_candidate_ids,
    )
    for torsion_label in sorted(uncovered):
        rejected_torsions.setdefault(torsion_label, "no_valid_fragment_found")

    assigned: dict[str, list[str]] = defaultdict(list)
    for torsion_label, candidate_id in covered_torsions.items():
        assigned[candidate_id].append(torsion_label)

    selected_fragments = write_selected_fragments(
        ligand,
        selected_candidates,
        assigned,
        torsion_map,
        out_dir,
        config,
        warnings,
    )
    fragment_index_path = write_fragment_index(selected_fragments, out_dir)
    summary_path = write_summary(
        ligand,
        selected_fragments,
        covered_torsions,
        rejected_torsions,
        warnings,
        out_dir,
    )
    output_paths = {fragment.fragment_id: str(fragment.manifest_path.parent) for fragment in selected_fragments}
    return FragmentationResult(
        selected_fragments=selected_fragments,
        covered_torsions=covered_torsions,
        rejected_torsions=rejected_torsions,
        warnings=warnings,
        output_paths=output_paths,
        summary_path=summary_path,
        fragment_index_path=fragment_index_path,
    )
