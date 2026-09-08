from __future__ import annotations

import json
import re
import warnings
from pathlib import Path
from typing import Any

from .Frcmod import FrcmodFile, _normalize_param_name_to_key, normalize_dihe_line

_ITERATION_FRCMOD_RE = re.compile(r"it(\d+)\.frcmod$")
_FRAGMENT_SUFFIX_RE = re.compile(r"(\d+)$")


class MergeWarning(UserWarning):
    """Raised when a fragment contributes no fitted torsions to a merge.

    A fragment legitimately produces no iteration output when its starting
    parameters were already good enough or the high-level profile came back
    flat, so this is a warning rather than an error: the remaining fragments
    still merge and the parent keeps its existing terms for that fragment.
    """


def _fragment_sort_key(path: Path) -> tuple[int, str]:
    """Build a sortable key for iteration frcmod filenames.

    Args:
        path: Candidate iteration frcmod path.

    Returns:
        Numeric iteration number and filename as a tiebreaker.
    """

    match = _ITERATION_FRCMOD_RE.fullmatch(path.name)
    if match is None:
        return (-1, path.name)
    return (int(match.group(1)), path.name)


def find_latest_iteration_frcmod(fragment_dir: Path) -> Path:
    """Return the highest-numbered ``itX.frcmod`` in a fragment directory.

    Args:
        fragment_dir: Fragment run directory containing ``ffpopt`` outputs.

    Returns:
        Path to the latest iteration frcmod file.

    Raises:
        FileNotFoundError: If the directory contains no ``itX.frcmod`` files.
    """

    candidates = list_iteration_frcmods(fragment_dir)
    if not candidates:
        raise FileNotFoundError(
            f"No itX.frcmod files found in fragment directory: {fragment_dir}"
        )
    return candidates[-1]


def list_iteration_frcmods(fragment_dir: Path) -> list[Path]:
    """Return all ``itX.frcmod`` files in ascending iteration order.

    Prefer targeted ``stat`` probes (``it01.frcmod``, ``it02.frcmod``, ...)
    over a full directory listing. Fragment run dirs often contain thousands
    of wavefront / geomopt artifacts; listing them on shared filesystems
    (VAST / Lustre) dominates merge wall time.
    """

    found: list[Path] = []
    seen: set[str] = set()
    consecutive_misses = 0
    # Generous upper bound: bond-batched fragments can promote many itXX files.
    for i in range(1, 256):
        hit = None
        for name in (f"it{i:02d}.frcmod", f"it{i}.frcmod"):
            path = fragment_dir / name
            try:
                if path.is_file():
                    hit = path
                    break
            except OSError:
                continue
        if hit is not None:
            key = hit.name.lower()
            if key not in seen:
                seen.add(key)
                found.append(hit)
            consecutive_misses = 0
            continue
        consecutive_misses += 1
        # Stop after a short gap once we have seen at least one file, or
        # after a few misses when the directory has no itXX at all.
        if consecutive_misses >= 3:
            break

    if found:
        return sorted(found, key=_fragment_sort_key)

    # Fallback for unusual names (still cheaper than listing when empty).
    try:
        candidates = [
            path
            for path in fragment_dir.glob("it*.frcmod")
            if path.is_file() and _ITERATION_FRCMOD_RE.fullmatch(path.name)
        ]
    except OSError:
        candidates = []
    return sorted(candidates, key=_fragment_sort_key)


def _has_iteration_frcmod(directory: Path) -> bool:
    """Return whether a directory directly contains an ``itX.frcmod`` file.

    Args:
        directory: Directory to inspect (non-recursively).

    Returns:
        True if at least one direct child is an ``itX.frcmod`` file.
    """

    return bool(list_iteration_frcmods(directory))


def discover_fragment_dirs(root: Path) -> list[Path]:
    """Discover mergeable fragment run directories under a root directory.

    Handles both run layouts produced by ALPS
    ``run_fragmented_dihed_twist_workflow``: a single-coupling-group fragment
    whose ``itX.frcmod`` files live directly in the fragment directory, and a
    multi-coupling-group fragment whose fits live in ``coupling_NN/``
    subdirectories of the fragment directory. For the latter, the
    ``coupling_NN`` subdirectories - not the fragment directory itself - are
    the mergeable run directories.

    Args:
        root: Parent directory containing fragment subdirectories.

    A fragment that produced no ``itX.frcmod`` at all - because its starting
    parameters were already good or its high-level profile was flat - is
    skipped with a :class:`MergeWarning` rather than silently dropped.

    Returns:
        Sorted list of run directories that contain at least one
        ``itX.frcmod`` file (fragment directories and/or their
        ``coupling_NN`` subdirectories).
    """

    fragment_dirs: list[Path] = []
    for child in sorted(path for path in root.iterdir() if path.is_dir()):
        if _has_iteration_frcmod(child):
            # Single-coupling-group fragment: fits live in the dir itself.
            fragment_dirs.append(child)
            continue
        # Multi-coupling-group fragment: fits live one level down, in
        # coupling_NN/ subdirectories. Recurse one level to find them.
        run_dirs = [
            sub
            for sub in sorted(path for path in child.iterdir() if path.is_dir())
            if _has_iteration_frcmod(sub)
        ]
        if not run_dirs:
            warnings.warn(
                f"No itXX.frcmod found under fragment directory {child}; "
                "skipping it. This is expected when the fragment needed no "
                "refit (already-good parameters or a flat high-level profile).",
                MergeWarning,
                stacklevel=2,
            )
            continue
        fragment_dirs.extend(run_dirs)
    return fragment_dirs


def _format_fragment_source_label(fragment_dir: Path) -> str:
    """Build a compact source tag for merged DIHE comments.

    Args:
        fragment_dir: Fragment run directory contributing fitted parameters.

    Returns:
        A stable source label suitable for trailing frcmod comment text.
    """

    match = _FRAGMENT_SUFFIX_RE.search(fragment_dir.name)
    if match is not None:
        return f"fragment={match.group(1)}"
    return f"fragment={fragment_dir.name}"


def _annotate_dihe_lines_with_source(lines: list[str], source_label: str) -> list[str]:
    """Append a fragment source tag to fitted DIHE lines.

    Args:
        lines: Normalized DIHE lines for one fitted parameter family.
        source_label: Fragment identifier to append.

    Returns:
        DIHE lines annotated with the fragment source label.
    """

    return [f"{line}  {source_label}" for line in lines]


def _load_fit_torsions(path: Path) -> list[dict[str, Any]]:
    """Load torsion-to-parent mapping metadata when present.

    Args:
        path: Path to the fragment ``fit_torsions.json`` file.

    Returns:
        Parsed fit-torsion records, or an empty list when the file is absent.
    """

    if not path.exists():
        return []
    payload = json.loads(path.read_text())
    if not isinstance(payload, list):
        raise ValueError(f"Expected a list in {path}")
    return payload


def _collect_param_keys(param_names) -> set[tuple[str, str, str, str]]:
    """Normalize an iterable of ``ffpopt`` parameter names into DIHE keys.

    Args:
        param_names: Parameter family names such as ``LIG_ca-ca-c-o``.

    Returns:
        The subset of names that describe four-atom dihedral families, as
        normalized atom-type keys.
    """

    keys: set[tuple[str, str, str, str]] = set()
    for param_name in param_names:
        key = _normalize_param_name_to_key(param_name)
        if key is not None:
            keys.add(key)
    return keys


def _load_fit_param_keys(
    path: Path,
) -> tuple[set[tuple[str, str, str, str]] | None, set[tuple[str, str, str, str]]]:
    """Load the DIHE families optimized and scanned by an iteration fit file.

    A fragment fit rarely touches only the torsions that were scanned: the
    optimizer relaxes the whole coupled family of dihedral types around the
    scanned bonds, and ``itXX.fit.json`` records that family in its top-level
    ``params`` block. Merging only the scanned (plotted) subset leaves the
    parent ligand with half a fitted torsional surface and half the old generic
    parameters, so the fitted set - not the plotted set - is what gates the
    merge.

    Args:
        path: Path to an ``itXX.fit.json`` file.

    Returns:
        A ``(fitted_keys, scanned_keys)`` pair. ``fitted_keys`` is the union of
        the top-level ``params`` families, the per-system ``params`` families,
        and the plotted families; it is ``None`` when the file is absent or
        names no dihedral families, letting callers fall back to every DIHE
        group in the frcmod. ``scanned_keys`` holds only the families listed in
        ``systems[].profiles[].plots``, i.e. the torsions the scan targeted
        directly.
    """

    if not path.exists():
        return None, set()

    payload = json.loads(path.read_text())
    fitted_keys = _collect_param_keys(payload.get("params", {}))
    scanned_keys: set[tuple[str, str, str, str]] = set()
    for system in payload.get("systems", []):
        fitted_keys |= _collect_param_keys(system.get("params", {}))
        for profile in system.get("profiles", []):
            scanned_keys |= _collect_param_keys(profile.get("plots", []))
    fitted_keys |= scanned_keys
    return (fitted_keys or None), scanned_keys


def _load_fragment_update(fragment_dir: Path) -> dict[str, Any]:
    """Collect merge-ready torsion parameters from one fragment directory.

    Accumulates DIHE groups from **all** ``itX.frcmod`` files in order so
    drop-mode survivors fitted in earlier iterations are not lost when a later
    iteration omits them. Later iterations overwrite the same DIHE key when they
    explicitly refit it.

    Args:
        fragment_dir: Fragment run directory containing ``ffpopt`` outputs.

    Returns:
        Structured update payload including the chosen iteration frcmod, parsed
        DIHE groups, and fit-torsion metadata.
    """

    iterations = list_iteration_frcmods(fragment_dir)
    if not iterations:
        raise FileNotFoundError(
            f"No itX.frcmod files found in fragment directory: {fragment_dir}"
        )

    dihe_groups: dict[tuple[str, str, str, str], list[str]] = {}
    fitted_param_keys: set[tuple[str, str, str, str]] = set()
    scanned_param_keys: set[tuple[str, str, str, str]] = set()
    saw_fit_filter = False

    for iteration_frcmod in iterations:
        parsed = FrcmodFile.read(iteration_frcmod)
        groups = parsed.dihe_groups()
        fitted_keys, scanned_keys = _load_fit_param_keys(
            fragment_dir / f"{iteration_frcmod.stem}.fit.json"
        )
        if fitted_keys is not None:
            saw_fit_filter = True
            groups = {
                key: lines
                for key, lines in groups.items()
                if key in fitted_keys
            }
            fitted_param_keys |= fitted_keys
        else:
            fitted_param_keys |= set(groups.keys())
        scanned_param_keys |= scanned_keys
        # Later iterations win on key collision (explicit refit).
        dihe_groups.update(groups)

    fit_torsions = _load_fit_torsions(fragment_dir / "fit_torsions.json")
    return {
        "fragment_dir": fragment_dir,
        "iteration_frcmod": iterations[-1],
        "iteration_frcmods": iterations,
        "fit_torsions": fit_torsions,
        "dihe_groups": dihe_groups,
        "fitted_param_keys": fitted_param_keys if saw_fit_filter else None,
        "scanned_param_keys": scanned_param_keys,
    }


def _dihe_body_fingerprint(lines: list[str]) -> tuple[str, ...]:
    """Stable fingerprint of DIHE lines ignoring trailing source annotations."""
    bodies: list[str] = []
    for line in lines:
        # Source tags are appended as ``  fragment=N`` (see annotate helper).
        body = line.split("  fragment=", 1)[0].rstrip()
        bodies.append(normalize_dihe_line(body) if body.strip() else body)
    return tuple(bodies)


def _safe_load_fragment_update(
    fragment_dir: Path,
) -> tuple[Path, dict[str, Any] | None, str | None]:
    """Load one fragment update; return ``(dir, update|None, skip_reason|None)``."""
    try:
        update = _load_fragment_update(fragment_dir)
    except FileNotFoundError:
        return fragment_dir, None, "no itXX.frcmod in the fragment directory"
    if not update["dihe_groups"]:
        return (
            fragment_dir,
            None,
            f"{update['iteration_frcmod'].name} defines no fitted DIHE terms",
        )
    return fragment_dir, update, None


def merge_fragment_frcmods(
    parent_frcmod_path: Path,
    output_frcmod_path: Path,
    fragment_dirs: list[Path],
    report_path: Path | None = None,
) -> dict[str, Any]:
    """Merge fitted fragment torsion terms back into a parent frcmod.

    Args:
        parent_frcmod_path: Parent ligand frcmod to update.
        output_frcmod_path: Path for the merged parent frcmod.
        fragment_dirs: Fragment directories to merge. Each directory contributes
            DIHE terms accumulated from **all** ``itX.frcmod`` files in order
            (drop-mode survivors from earlier iterations are retained unless a
            later iteration explicitly refits the same key). A directory with no
            iteration frcmod, or one whose iteration frcmods define no ``DIHE``
            terms, is skipped with a :class:`MergeWarning` and recorded under
            ``skipped_fragments`` in the report; the remaining fragments still
            merge.
        report_path: Optional JSON report path summarizing what was merged.

    Returns:
        A JSON-serializable report describing the merge.

    Notes
    -----
    With ``bytype`` fits, distinct rotors often share one DIHE atom-type key
    (e.g. ``c3-c3-n4-c3``). When two fragments both directly scanned that
    family, the merge keeps the first scanned contributor in ``fragment_dirs``
    order (or the identical parameter set if both agree) and records the
    collision under ``conflicts`` with a :class:`MergeWarning` - it does not
    abort the run.
    """
    from concurrent.futures import ThreadPoolExecutor

    parent = FrcmodFile.read(parent_frcmod_path)
    replacements: dict[tuple[str, str, str, str], list[str]] = {}
    replacement_sources: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    fragment_reports: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    dirs = [Path(p) for p in fragment_dirs]
    # Parallelize per-fragment I/O (frcmod + fit.json reads). Merge logic below
    # stays single-threaded so conflict resolution remains deterministic.
    n_workers = min(8, max(1, len(dirs)))
    if len(dirs) <= 1:
        loaded = [_safe_load_fragment_update(d) for d in dirs]
    else:
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            loaded = list(pool.map(_safe_load_fragment_update, dirs))

    for fragment_dir, update, skip_reason in loaded:
        if skip_reason is not None:
            warnings.warn(
                f"Skipping fragment {fragment_dir}: {skip_reason}. This is expected "
                "when the fragment needed no refit (already-good parameters or "
                "a flat high-level profile); the parent keeps its existing "
                "terms for it.",
                MergeWarning,
                stacklevel=2,
            )
            skipped.append({"fragment_dir": str(fragment_dir), "reason": skip_reason})
            continue
        assert update is not None

        dihe_groups = update["dihe_groups"]
        fit_torsions = update["fit_torsions"]
        scanned_param_keys = update["scanned_param_keys"]
        source_label = _format_fragment_source_label(fragment_dir)

        fragment_report = {
            "fragment_dir": str(fragment_dir),
            "iteration_frcmod": str(update["iteration_frcmod"]),
            "fit_torsions": fit_torsions,
            "dihedral_keys": ["-".join(key) for key in sorted(dihe_groups)],
            "dihedral_group_count": len(dihe_groups),
            "fitted_param_keys": (
                ["-".join(key) for key in sorted(update["fitted_param_keys"])]
                if update["fitted_param_keys"] is not None
                else None
            ),
            "scanned_param_keys": ["-".join(key) for key in sorted(scanned_param_keys)],
        }
        fragment_reports.append(fragment_report)

        for key, lines in dihe_groups.items():
            scanned = key in scanned_param_keys
            previous = replacement_sources.get(key)
            if previous is not None:
                winner_is_new = False
                resolution = "first_fit_wins"
                if previous["scanned"] and scanned:
                    # bytype: two fragments scanned different instances of the
                    # same atom-type family. Keep the first scanned contributor
                    # unless the Fourier terms already agree.
                    same = _dihe_body_fingerprint(
                        previous.get("raw_lines") or []
                    ) == _dihe_body_fingerprint(lines)
                    resolution = (
                        "identical_scanned" if same else "first_scanned_wins"
                    )
                    warnings.warn(
                        "Two fragments directly scanned the same DIHE family "
                        f"{'-'.join(key)}: keeping {previous['fragment_dir']} "
                        f"({resolution}), dropping {fragment_dir}. This is "
                        "common with bytype fits when distinct rotors share "
                        "atom types.",
                        MergeWarning,
                        stacklevel=2,
                    )
                    winner_is_new = False
                else:
                    # A fragment that scanned this torsion owns it; otherwise the
                    # first fragment to fit it wins, deterministically.
                    winner_is_new = scanned and not previous["scanned"]
                    resolution = (
                        "scanned_over_fitted"
                        if winner_is_new
                        else "first_fit_wins"
                    )
                conflicts.append(
                    {
                        "dihedral_key": "-".join(key),
                        "resolution": resolution,
                        "kept": (
                            str(fragment_dir)
                            if winner_is_new
                            else previous["fragment_dir"]
                        ),
                        "kept_scanned": scanned if winner_is_new else previous["scanned"],
                        "dropped": (
                            previous["fragment_dir"]
                            if winner_is_new
                            else str(fragment_dir)
                        ),
                        "dropped_scanned": previous["scanned"] if winner_is_new else scanned,
                    }
                )
                if not winner_is_new:
                    continue

            replacements[key] = _annotate_dihe_lines_with_source(lines, source_label)
            replacement_sources[key] = {
                "fragment_dir": str(fragment_dir),
                "fragment_label": source_label,
                "iteration_frcmod": str(update["iteration_frcmod"]),
                "fit_torsions": fit_torsions,
                "scanned": scanned,
                "raw_lines": list(lines),
            }

    if not replacements:
        warnings.warn(
            f"No fitted torsions were merged into {parent_frcmod_path}; the "
            "output is a copy of the parent frcmod.",
            MergeWarning,
            stacklevel=2,
        )

    change_counts = parent.replace_dihe_groups(replacements)
    output_frcmod_path.write_text(parent.render())

    merged_keys = []
    for key in sorted(replacements):
        source = replacement_sources[key]
        merged_keys.append(
            {
                "dihedral_key": "-".join(key),
                "fragment_dir": source["fragment_dir"],
                "fragment_label": source["fragment_label"],
                "iteration_frcmod": source["iteration_frcmod"],
                "fit_torsions": source["fit_torsions"],
                "directly_scanned": source["scanned"],
            }
        )

    report: dict[str, Any] = {
        "parent_frcmod": str(parent_frcmod_path),
        "output_frcmod": str(output_frcmod_path),
        "fragment_dirs": [str(path) for path in fragment_dirs],
        "fragment_reports": fragment_reports,
        "merged_dihedral_groups": merged_keys,
        "replacement_counts": change_counts,
        "conflicts": conflicts,
        "skipped_fragments": skipped,
    }

    if report_path is not None:
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True))

    return report
