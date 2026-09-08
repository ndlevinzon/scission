from __future__ import annotations

from collections import defaultdict

from .Models import CandidateFragment


def _charge_penalty(candidate: CandidateFragment) -> float:
    """Score how far a fragment's net charge is from the nearest integer.

    Args:
        candidate: Candidate fragment being ranked.

    Returns:
        Absolute distance between the fragment charge and the nearest integer.
    """

    nearest_integer = round(candidate.net_charge)
    return abs(candidate.net_charge - nearest_integer)


def select_fragments(
    candidate_pool: dict[str, CandidateFragment],
    torsion_labels: list[str],
    allow_parent_fallback: bool = False,
    preferred_candidate_ids: set[str] | None = None,
) -> tuple[list[CandidateFragment], dict[str, str], set[str]]:
    """Greedily choose fragments that maximize torsion coverage.

    Args:
        candidate_pool: Candidate fragments keyed by candidate id.
        torsion_labels: Torsion labels that still need coverage.
        allow_parent_fallback: Whether full-parent fallback fragments may be
            selected to cover otherwise uncovered torsions.
        preferred_candidate_ids: Optional subset of non-parent candidates to
            consider during the main greedy pass.

    Returns:
        A tuple of selected candidates, torsion-to-fragment assignments, and
        torsions left uncovered.
    """

    coverage: dict[str, set[str]] = defaultdict(set)
    for candidate_id, candidate in candidate_pool.items():
        for torsion in candidate.accepted_torsions:
            coverage[candidate_id].add(torsion)

    uncovered = set(torsion_labels)
    selected: list[CandidateFragment] = []
    assignment: dict[str, str] = {}
    non_parent_pool = {
        candidate_id: candidate
        for candidate_id, candidate in candidate_pool.items()
        if not candidate.is_parent_fallback
    }
    if preferred_candidate_ids is not None:
        non_parent_pool = {
            candidate_id: candidate
            for candidate_id, candidate in non_parent_pool.items()
            if candidate_id in preferred_candidate_ids
        }
    while uncovered:
        ranked = [
            (
                len(coverage[candidate_id] & uncovered),
                -int(candidate.is_parent_fallback),
                candidate.ring_cap_count,
                -len(candidate.retained_atoms),
                -_charge_penalty(candidate),
                candidate.worst_margin if candidate.worst_margin is not None else -999.0,
                -len(candidate.cut_bonds),
                candidate_id,
                candidate,
            )
            for candidate_id, candidate in non_parent_pool.items()
            if coverage[candidate_id] & uncovered
        ]
        if not ranked:
            break
        _, _, _, _, _, _, _, _, chosen = max(ranked)
        selected.append(chosen)
        newly_covered = coverage[chosen.candidate_id] & uncovered
        for torsion in sorted(newly_covered):
            assignment[torsion] = chosen.candidate_id
        uncovered -= newly_covered

    if allow_parent_fallback:
        parent_candidates = {
            candidate_id: candidate
            for candidate_id, candidate in candidate_pool.items()
            if candidate.is_parent_fallback
        }
        while uncovered:
            ranked = [
                (
                    len(coverage[candidate_id] & uncovered),
                    -len(candidate.retained_atoms),
                    -_charge_penalty(candidate),
                    candidate_id,
                    candidate,
                )
                for candidate_id, candidate in parent_candidates.items()
                if coverage[candidate_id] & uncovered
            ]
            if not ranked:
                break
            _, _, _, _, chosen = max(ranked)
            selected.append(chosen)
            newly_covered = coverage[chosen.candidate_id] & uncovered
            for torsion in sorted(newly_covered):
                assignment[torsion] = chosen.candidate_id
            uncovered -= newly_covered

    return selected, assignment, uncovered
