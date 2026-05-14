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

# Input
if len(sys.argv) != 3:
    sys.exit("Usage: python relax.py structure.cif model_name")

cif_path, model_name = sys.argv[1], sys.argv[2]
filename = os.path.splitext(os.path.basename(cif_path))[0]

# System setup
atoms = read(cif_path)
calculator = get_calculator(model_name)
atoms.calc = calculator

REPO_ROOT = Path(__file__).parents[2]
out_dir = ensure_dir(REPO_ROOT / "results" / "relaxed" / filename / model_name)
traj_path = out_dir / f"relax_{filename}_{model_name}.traj"
log_path = out_dir / f"relax_{filename}_{model_name}.log"

# Structure opt

opt = BFGS(FrechetCellFilter(atoms), trajectory=str(traj_path), logfile=str(log_path))
opt.run(fmax=1e-3, steps=300)

out_path = out_dir / f"relaxed_{filename}_{model_name}.cif"
atoms.write(str(out_path))
