"""md/run_md.py — Run NVT molecular dynamics with a uMLIP.

Usage
-----
    python scripts/md/run_md.py structure.cif temp model_name

Available model names are defined in ``scripts/utils/models.py`` (``CALCULATOR_BLOCKS``).
"""

import warnings
warnings.simplefilter("ignore", category=FutureWarning)

import sys, os
from pathlib import Path

from ase.io import read
from ase.build import make_supercell
from ase.md.nvtberendsen import NVTBerendsen
from ase.md.nose_hoover_chain import NoseHooverChainNVT
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.units import fs

sys.path.insert(0, str(Path(__file__).parents[1]))
from utils.io import ensure_dir
from utils.models import get_calculator

# ------------------------ Input handling ------------------------
if len(sys.argv) != 4:
    sys.exit("Usage: python run_md.py structure.cif temp model_name")

cif_path, temp, model_name = sys.argv[1], float(sys.argv[2]), sys.argv[3]
filename = os.path.splitext(os.path.basename(cif_path))[0]

# ------------------------ Load structure ------------------------
cell = read(cif_path)
atoms = make_supercell(cell, ((2, 0, 0), (0, 2, 0), (0, 0, 2)), order='atom-major')
atoms.calc = get_calculator(model_name)

MaxwellBoltzmannDistribution(atoms, temperature_K=temp)

REPO_ROOT = Path(__file__).parents[2]
out_dir = ensure_dir(REPO_ROOT / "results" / "md" / filename / model_name / f"{int(temp)}K")

# ------------------------ Equilibration (Berendsen thermostat) ------------------------
init = NVTBerendsen(
    atoms=atoms,
    temperature_K=temp,
    timestep=1*fs,
    taut=100*fs,
    logfile=str(out_dir / f"relaxation_{str(temp)}.log"),
    loginterval=1000,
)
init.run(50000)  # 50 ps

# ------------------------ Production (Nose-Hoover Chain thermostat) ------------------------
md = NoseHooverChainNVT(
    atoms=atoms,
    temperature_K=temp,
    timestep=1*fs,
    tdamp=100*fs,
    trajectory=str(out_dir / "md.traj"),
    logfile=str(out_dir / f"production_{str(temp)}.log"),
    loginterval=50,
)
md.run(100_000_000)  # 100 ns
