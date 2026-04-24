"""relaxation/relax.py — Relax crystal structures with a uMLIP.

Usage
-----
    python scripts/relaxation/relax.py structure.cif model_name

Available model names are defined in ``scripts/utils/models.py`` (``CALCULATOR_BLOCKS``).
"""

import warnings
warnings.simplefilter("ignore", category=FutureWarning)

import sys, os
from pathlib import Path

from ase.io import read
from ase.optimize import BFGS
from ase.filters import FrechetCellFilter

sys.path.insert(0, str(Path(__file__).parents[1]))
from utils.io import ensure_dir
from utils.models import get_calculator

# ------------------------ Input handling ------------------------
if len(sys.argv) != 3:
    sys.exit("Usage: python relax.py structure.cif model_name")

cif_path, model_name = sys.argv[1], sys.argv[2]
filename = os.path.splitext(os.path.basename(cif_path))[0]

# ------------------------ Load structure ------------------------
atoms = read(cif_path)

# ------------------------ Calculator ------------------------
calculator = get_calculator(model_name)
atoms.calc = calculator

# ------------------------ Relaxation ------------------------
REPO_ROOT = Path(__file__).parents[2]
out_dir = ensure_dir(REPO_ROOT / "results" / "relaxed" / filename / model_name)
traj_path = out_dir / "relax.traj"
log_path = out_dir / "relax.log"

opt = BFGS(FrechetCellFilter(atoms), trajectory=str(traj_path), logfile=str(log_path))
opt.run(fmax=1e-3, steps=300)

out_path = out_dir / "POSCAR_relaxed"
atoms.write(str(out_path))
