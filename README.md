# scission

AMBER-aware torsion fragments from a charged ligand triplet (`.mol2` / `.lib`
/ `.frcmod`). Used for small-molecule torsion fitting: enumerate acyclic
rotatable bonds, cap and reduce fragments, write scan-ready Amber files, and
merge fitted `DIHE` terms back into the parent `frcmod`.

Docs: https://scission-da161d.gitlab.io/
Repo: https://github.com/ndlevinzon/scission

## Install

AmberTools (`tleap`, `parmchk2`) must be on `PATH` to write `parm7` / `rst7`.

From this directory:

```bash
python3 -m pip install -e .
```

## Tests

```bash
python -m unittest discover -s tests -v
```

`numpy<2` matches ParmEd. RDKit is required for SMARTS matching and
`scission pick-bond`.

## Run

```bash
scission fragment \
  --mol2 LIG.mol2 --lib LIG.lib --frcmod LIG.frcmod \
  --outdir fragments --nproc 8
```

Prints a JSON summary to stdout and writes `summary.json` plus
`fragment_index.json` under `--outdir`.

Stricter legacy torsions (drop amide-like single bonds):

```bash
scission fragment --acyclic-rotatable-only \
  --mol2 LIG.mol2 --lib LIG.lib --frcmod LIG.frcmod --outdir fragments
```

Nominate extra central bonds with SMARTS (`:1` / `:2` on the two atoms):

```bash
scission fragment \
  --include-bond-smarts "[C:1](=[O])[N:2]" \
  --mol2 LIG.mol2 --lib LIG.lib --frcmod LIG.frcmod --outdir fragments
```

Pfizer or WBO-style growth around each rotatable bond (Stern et al.,
[bioRxiv 2020.08.27.270934v2](https://www.biorxiv.org/content/10.1101/2020.08.27.270934v2)):

```bash
scission fragment --strategy pfizer \
  --mol2 LIG.mol2 --lib LIG.lib --frcmod LIG.frcmod --outdir fragments
scission fragment --strategy wbo \
  --mol2 LIG.mol2 --lib LIG.lib --frcmod LIG.frcmod --outdir fragments
```

YAML `strategy: pfizer` (or `wbo`) is equivalent. `scission` remains the
default. Register a custom scheme with `scission.register_strategy`.

Allow-list only matching bonds:

```bash
scission fragment \
  --restrict-bond-smarts "[c:1]-[c:2]" \
  --mol2 LIG.mol2 --lib LIG.lib --frcmod LIG.frcmod --outdir fragments
```

Interactive SMARTS (browser):

```bash
scission pick-bond --mol2 LIG.mol2
```

Merge fitted fragment `itXX.frcmod` files back to the parent:

```bash
scission merge \
  --parent-frcmod LIG.frcmod \
  --fragments-root fragments \
  --out LIG.merged.frcmod \
  --report LIG.merge_report.json
```

Python:

```python
from pathlib import Path
from scission import FragmentConfig, InputBundle, fragment_ligand

result = fragment_ligand(
    InputBundle(mol2_path=Path("LIG.mol2"), lib_path=Path("LIG.lib"),
                frcmod_path=Path("LIG.frcmod")),
    Path("fragments"),
    FragmentConfig(strategy="pfizer"),
)
```

Custom schemes return the same `CandidateFragment` objects (capping and Amber
writes stay unchanged):

```python
from scission import register_strategy
from scission.Fragments import candidate_from_retained_atoms

@register_strategy("quartet_only")
def quartet_only(ligand, torsion, config):
    return [candidate_from_retained_atoms(ligand, torsion, set(torsion.atom_indices))]
```


## Outputs (per fragment)

`fragment.mol2`, `.xyz`, `.lib`, `.frcmod`, `.auto.frcmod`, `.parm7`, `.rst7`,
`manifest.json`, `fit_torsions.json`. Atom indices in those JSON files are
**1-based**.


