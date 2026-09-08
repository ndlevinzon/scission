"""Parallel helpers for scission screening and per-fragment Amber writes."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Sequence

from .Fragments import build_candidate_fragments
from .Models import (
    CandidateFragment,
    FragmentConfig,
    Ligand,
    SelectedFragment,
    TorsionDefinition,
)
from .Screen import screen_candidate
from .Writers import write_fragment_outputs

_LOG = logging.getLogger("scission.Parallel")


def split_core_budget(total_cores: int, n_jobs: int) -> tuple[int, int]:
    """Prefer as many workers as possible with ``workers * per_job <= total``.

    Keep algorithm in sync with ``ffpopt.runtime.FastWavefront.split_core_budget``
    (scission stays free of ffpopt imports).
    """
    total = max(1, int(total_cores))
    n_jobs = max(1, int(n_jobs))
    if n_jobs == 1:
        return 1, total
    n_workers = min(total, n_jobs)
    per_job = max(1, total // n_workers)
    return n_workers, per_job


def _screen_one_torsion(job: dict[str, Any]) -> dict[str, Any]:
    """Screen all candidates for one torsion (spawn-pool worker)."""
    ligand: Ligand = job["ligand"]
    torsion: TorsionDefinition = job["torsion"]
    include_rigid = bool(job["include_rigid_single_bonds"])
    rotatable_smarts = tuple(job["rotatable_bond_smarts"])
    angle_step = int(job["angle_step"])
    thresholds = job["thresholds"]
    use_parent_fallback = bool(job["use_parent_fallback"])

    valid_for_torsion = False
    best_failure: tuple[float, dict[str, object]] | None = None
    evaluations: list[dict[str, Any]] = []

    for candidate in build_candidate_fragments(
        ligand,
        torsion,
        include_rigid_single_bonds=include_rigid,
        rotatable_bond_smarts=rotatable_smarts,
    ):
        screen = screen_candidate(
            ligand,
            torsion,
            candidate,
            angle_step=angle_step,
            thresholds=thresholds,
        )
        counts_as_success = screen.accepted and (
            not candidate.is_parent_fallback or use_parent_fallback
        )
        evaluations.append(
            {
                "candidate": candidate,
                "counts_as_success": counts_as_success,
                "worst_margin": screen.worst_margin,
                "reason": screen.reason,
                "violation": screen.violation,
            }
        )
        if counts_as_success:
            valid_for_torsion = True
        elif screen.violation is not None and not candidate.is_parent_fallback:
            failure_payload = {
                "reason": screen.reason,
                "candidate_id": candidate.candidate_id,
                "retained_atom_count": len(candidate.retained_atoms),
                "cut_bonds": [list(bond) for bond in sorted(candidate.cut_bonds)],
                "violation": screen.violation,
            }
            if best_failure is None or screen.worst_margin > best_failure[0]:
                best_failure = (screen.worst_margin, failure_payload)

    return {
        "torsion_label": torsion.label,
        "evaluations": evaluations,
        "valid_for_torsion": valid_for_torsion,
        "best_failure": best_failure[1] if best_failure is not None else None,
    }


def screen_torsions(
    ligand: Ligand,
    torsions: Sequence[TorsionDefinition],
    config: FragmentConfig,
    *,
    logger: logging.Logger | None = None,
) -> tuple[
    dict[str, CandidateFragment],
    dict[str, list[CandidateFragment]],
    dict[str, object],
]:
    """Screen every (torsion, candidate) pair; pool over torsions when ``nproc>1``.

    Returns
    -------
    candidate_pool, accepted_by_torsion, rejected_torsions
        Same structures previously built inline in :func:`fragment_ligand`.
    """
    log = logger or _LOG
    torsions = list(torsions)
    candidate_pool: dict[str, CandidateFragment] = {}
    accepted_by_torsion: dict[str, list[CandidateFragment]] = {}
    rejected_torsions: dict[str, object] = {}

    if not torsions:
        return candidate_pool, accepted_by_torsion, rejected_torsions

    jobs = [
        {
            "ligand": ligand,
            "torsion": torsion,
            "include_rigid_single_bonds": config.include_rigid_single_bonds,
            "rotatable_bond_smarts": tuple(config.rotatable_bond_smarts),
            "angle_step": config.angle_step,
            "thresholds": config.clash_thresholds,
            "use_parent_fallback": config.use_parent_fallback,
        }
        for torsion in torsions
    ]

    n_workers, _ = split_core_budget(getattr(config, "nproc", 1) or 1, len(jobs))
    log.info(
        "[scission] screening %s torsion(s) with %s worker(s) (nproc=%s)",
        len(jobs),
        n_workers,
        getattr(config, "nproc", 1),
    )

    if n_workers == 1:
        results = [_screen_one_torsion(job) for job in jobs]
    else:
        import multiprocessing as mp

        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=n_workers) as pool:
            results = pool.map(_screen_one_torsion, jobs)

    for result in results:
        torsion_label = result["torsion_label"]
        accepted_candidates: list[CandidateFragment] = []
        for evaluation in result["evaluations"]:
            candidate: CandidateFragment = evaluation["candidate"]
            candidate_pool.setdefault(candidate.candidate_id, candidate)
            if evaluation["counts_as_success"]:
                pooled = candidate_pool[candidate.candidate_id]
                pooled.torsion_labels.update(candidate.torsion_labels)
                pooled.accepted_torsions.add(torsion_label)
                margin = evaluation["worst_margin"]
                pooled.worst_margin = (
                    margin
                    if pooled.worst_margin is None
                    else min(pooled.worst_margin, margin)
                )
                accepted_candidates.append(pooled)
        if not result["valid_for_torsion"]:
            rejected_torsions[torsion_label] = (
                result["best_failure"]
                if result["best_failure"] is not None
                else "no_valid_fragment_found"
            )
        else:
            accepted_by_torsion[torsion_label] = accepted_candidates

    return candidate_pool, accepted_by_torsion, rejected_torsions


def _write_one_fragment(job: dict[str, Any]) -> dict[str, Any]:
    """Write one selected fragment (thread-pool worker; Amber tools are subprocesses)."""
    local_warnings: list[str] = []
    fragment = write_fragment_outputs(
        job["ligand"],
        job["candidate"],
        job["fragment_id"],
        job["assigned_torsions"],
        job["torsion_map"],
        job["out_dir"],
        job["config"],
        local_warnings,
    )
    return {
        "index": job["index"],
        "fragment": fragment,
        "warnings": local_warnings,
    }


def write_selected_fragments(
    ligand: Ligand,
    selected_candidates: Sequence[CandidateFragment],
    assigned: dict[str, list[str]],
    torsion_map: dict[str, TorsionDefinition],
    out_dir: Path,
    config: FragmentConfig,
    warnings: list[str],
    *,
    logger: logging.Logger | None = None,
) -> list[SelectedFragment]:
    """Materialize selected fragments; pool parmchk2/tleap writes when ``nproc>1``."""
    log = logger or _LOG
    jobs = [
        {
            "index": index,
            "ligand": ligand,
            "candidate": candidate,
            "fragment_id": f"fragment_{index}",
            "assigned_torsions": sorted(assigned[candidate.candidate_id]),
            "torsion_map": torsion_map,
            "out_dir": out_dir,
            "config": config,
        }
        for index, candidate in enumerate(selected_candidates, start=1)
    ]
    if not jobs:
        return []

    n_workers, _ = split_core_budget(getattr(config, "nproc", 1) or 1, len(jobs))
    log.info(
        "[scission] writing %s fragment(s) with %s worker(s) (nproc=%s)",
        len(jobs),
        n_workers,
        getattr(config, "nproc", 1),
    )

    if n_workers == 1:
        raw = [_write_one_fragment(job) for job in jobs]
    else:
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            raw = list(pool.map(_write_one_fragment, jobs))

    raw.sort(key=lambda item: item["index"])
    for item in raw:
        warnings.extend(item["warnings"])
    return [item["fragment"] for item in raw]
