from __future__ import annotations

from dataclasses import dataclass

FRCMOD_SECTIONS = ("MASS", "BOND", "ANGLE", "DIHE", "IMPROPER", "NONB")
_DIHE_PK_SEPARATOR = " " * 4
_DIHE_PHASE_SEPARATOR = " " * 6
_DIHE_PERIODICITY_SEPARATOR = " " * 11
_DIHE_COMMENT_SEPARATOR = " " * 5
_DIHE_PK_WIDTH = 6
_DIHE_PHASE_WIDTH = 7
_DIHE_PERIODICITY_WIDTH = 6


def _normalize_dihe_key(atom_types: tuple[str, str, str, str]) -> tuple[str, str, str, str]:
    """Normalize a dihedral atom-type tuple so reverse order compares equal.

    Args:
        atom_types: Four atom types describing a dihedral entry.

    Returns:
        The lexicographically canonical orientation of the dihedral type tuple.
    """

    reversed_types = tuple(reversed(atom_types))
    return min(atom_types, reversed_types)


def _normalize_param_name_to_key(param_name: str) -> tuple[str, str, str, str] | None:
    """Convert an ``ffpopt`` parameter family name into a DIHE atom-type key.

    Args:
        param_name: Parameter family such as ``LIG_ca-ca-c-o``.

    Returns:
        Normalized four-atom-type tuple, or ``None`` when the name does not
        describe a four-atom dihedral family.
    """

    normalized = param_name.removeprefix("LIG_").replace("_", "-")
    atom_types = parse_dihe_atom_types(normalized)
    if atom_types is None:
        return None
    return _normalize_dihe_key(atom_types)


def parse_dihe_atom_types(line: str) -> tuple[str, str, str, str] | None:
    """Extract the atom-type key from a single ``DIHE`` line.

    Args:
        line: Raw line from the ``DIHE`` section of an Amber ``.frcmod`` file.

    Returns:
        The four stripped atom types when the line appears to define a dihedral,
        otherwise ``None``.
    """

    if not line.strip():
        return None
    prefix = line[:11]
    parts = [part.strip() for part in prefix.split("-")]
    if len(parts) != 4 or any(not part for part in parts):
        return None
    return tuple(parts)  # type: ignore[return-value]


def normalize_dihe_line(line: str) -> str:
    """Normalize one DIHE parameter line to a fixed-width frcmod layout.

    Args:
        line: Raw line from a ``DIHE`` section.

    Returns:
        The reformatted line when it contains a parseable DIHE entry, otherwise
        the original line unchanged.
    """

    atom_types = parse_dihe_atom_types(line)
    if atom_types is None:
        return line

    tokens = line[11:].split(None, 4)
    if len(tokens) < 4:
        return line

    idivf, pk, phase, periodicity = tokens[:4]
    comment = tokens[4] if len(tokens) == 5 else ""

    def format_numeric(token: str, width: int) -> str:
        try:
            return f"{float(token):.{3}f}".rjust(width)
        except ValueError:
            return token.rjust(width)

    pieces = [
        line[:11].ljust(11),
        idivf.rjust(4),
        _DIHE_PK_SEPARATOR,
        format_numeric(pk, _DIHE_PK_WIDTH),
        _DIHE_PHASE_SEPARATOR,
        format_numeric(phase, _DIHE_PHASE_WIDTH),
        _DIHE_PERIODICITY_SEPARATOR,
        format_numeric(periodicity, _DIHE_PERIODICITY_WIDTH),
    ]
    if comment:
        pieces.extend([_DIHE_COMMENT_SEPARATOR, comment])
    return "".join(pieces)


@dataclass
class FrcmodFile:
    """Parsed ``.frcmod`` content with section-preserving round-tripping.

    Attributes:
        title_lines: Lines before the first named frcmod section.
        sections: Raw lines for each named frcmod section, excluding headers.
    """

    title_lines: list[str]
    sections: dict[str, list[str]]

    @classmethod
    def parse(cls, text: str) -> "FrcmodFile":
        """Parse a raw Amber ``.frcmod`` file into named sections.

        Args:
            text: Full file contents to parse.

        Returns:
            A parsed frcmod representation that preserves section ordering and
            unmodified line content.
        """

        title_lines: list[str] = []
        sections = {section: [] for section in FRCMOD_SECTIONS}
        current_section: str | None = None
        saw_section = False

        for line in text.splitlines():
            stripped = line.strip()
            if stripped in FRCMOD_SECTIONS:
                current_section = stripped
                saw_section = True
                continue
            if saw_section and current_section is not None:
                sections[current_section].append(line)
            else:
                title_lines.append(line)

        return cls(title_lines=title_lines, sections=sections)

    @classmethod
    def read(cls, path) -> "FrcmodFile":
        """Read and parse a ``.frcmod`` file from disk.

        Args:
            path: Path to the frcmod file.

        Returns:
            Parsed frcmod contents.
        """

        return cls.parse(path.read_text())

    def dihe_groups(self) -> dict[tuple[str, str, str, str], list[str]]:
        """Group ``DIHE`` lines by normalized atom-type key.

        Returns:
            Mapping from normalized dihedral atom-type tuple to the raw lines
            that define all Fourier terms for that key.
        """

        grouped: dict[tuple[str, str, str, str], list[str]] = {}
        for line in self.sections["DIHE"]:
            atom_types = parse_dihe_atom_types(line)
            if atom_types is None:
                continue
            grouped.setdefault(_normalize_dihe_key(atom_types), []).append(
                normalize_dihe_line(line)
            )
        return grouped

    def replace_dihe_groups(
        self,
        replacements: dict[tuple[str, str, str, str], list[str]],
    ) -> dict[str, int]:
        """Replace matching ``DIHE`` groups and append brand-new ones.

        Args:
            replacements: Replacement DIHE lines keyed by normalized atom-type
                tuple.

        Returns:
            Count summary with ``replaced`` and ``added`` group totals.
        """

        if not replacements:
            return {"replaced": 0, "added": 0}

        new_lines: list[str] = []
        seen_keys: set[tuple[str, str, str, str]] = set()
        replaced = 0

        for line in self.sections["DIHE"]:
            atom_types = parse_dihe_atom_types(line)
            if atom_types is None:
                new_lines.append(line)
                continue

            key = _normalize_dihe_key(atom_types)
            if key not in replacements:
                new_lines.append(line)
                continue
            if key in seen_keys:
                continue

            new_lines.extend(replacements[key])
            seen_keys.add(key)
            replaced += 1

        added = 0
        for key, lines in replacements.items():
            if key in seen_keys:
                continue
            new_lines.extend(lines)
            added += 1

        self.sections["DIHE"] = [line for line in new_lines if line.strip()]
        return {"replaced": replaced, "added": added}

    def render(self) -> str:
        """Serialize the parsed frcmod back to text.

        Returns:
            Full frcmod text with the standard Amber section ordering.
        """

        lines: list[str] = [*self.title_lines]
        for section in FRCMOD_SECTIONS:
            if lines and lines[-1] != "":
                lines.append("")
            lines.append(section)
            lines.extend(self.sections[section])
        return "\n".join(lines) + "\n"
