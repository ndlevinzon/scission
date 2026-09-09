# Set by a ligandparam-bundled copy so ALPS can tell an in-tree scission
# from an independent checkout of the same import name.
# Independent GitLab/PyPI installs must leave this False / omitted.
__ligandparam_bundle__ = False
__version__ = "0.3.0"

from .Models import (
    Atom,
    Bond,
    ClashThresholds,
    FragmentConfig,
    FragmentationResult,
    InputBundle,
    Ligand,
    SelectedFragment,
    TorsionDefinition,
)
from .Pipeline import fragment_ligand
from .Strategies import available_strategies, register_strategy
from .Torsions import match_central_bond_smarts

__all__ = [
    "Atom",
    "Bond",
    "ClashThresholds",
    "FragmentConfig",
    "FragmentationResult",
    "InputBundle",
    "Ligand",
    "SelectedFragment",
    "TorsionDefinition",
    "available_strategies",
    "fragment_ligand",
    "match_central_bond_smarts",
    "register_strategy",
]
