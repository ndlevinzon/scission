# scission

AMBER-aware torsion fragments from a charged ligand triplet (`.mol2` / `.lib`
/ `.frcmod`). Used for small-molecule torsion fitting: enumerate acyclic
rotatable bonds, cap and reduce fragments, write scan-ready Amber files, and
merge fitted `DIHE` terms back into the parent `frcmod`.

Docs: https://scission-da161d.gitlab.io/

## Install

AmberTools (`tleap`, `parmchk2`) must be on `PATH` to write `parm7` / `rst7`.

From this directory:

```bash
python3 -m pip install -e .
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
    FragmentConfig(),
)
```

## Outputs (per fragment)

`fragment.mol2`, `.xyz`, `.lib`, `.frcmod`, `.auto.frcmod`, `.parm7`, `.rst7`,
`manifest.json`, `fit_torsions.json`. Atom indices in those JSON files are
**1-based**.


