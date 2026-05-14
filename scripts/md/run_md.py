import warnings
warnings.simplefilter("ignore", category=FutureWarning)

import sys, os
import argparse
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
from utils.config import REPO_ROOT

# Parse arguments
parser = argparse.ArgumentParser(description="Run MD simulation")
parser.add_argument("structure", help="Path to input structure (CIF format)")
parser.add_argument("temperature", type=float, help="Temperature (K)")
parser.add_argument("model", help="ML model name")
args = parser.parse_args()

cif_path, temp, model_name = args.structure, args.temperature, args.model
filename = os.path.splitext(os.path.basename(cif_path))[0]

# System setup
cell = read(cif_path)
atoms = make_supercell(cell, ((2, 0, 0), (0, 2, 0), (0, 0, 2)), order='atom-major')
atoms.calc = get_calculator(model_name)

MaxwellBoltzmannDistribution(atoms, temperature_K=temp)

out_dir = ensure_dir(REPO_ROOT / "results" / "md" / filename / model_name)
temp_int = int(temp)

# Equilibration (NVT Berendsen thermostat)
init = NVTBerendsen(
    atoms=atoms,
    temperature_K=temp,
    timestep=1*fs,
    taut=100*fs,
    logfile=str(out_dir / f"relaxation_{filename}_{model_name}_{temp_int}.log"),
    loginterval=1000,
)
init.run(50000)  # 50 ps

# Production (NVT Nose-Hoover thermostat)
md = NoseHooverChainNVT(
    atoms=atoms,
    temperature_K=temp,
    timestep=1*fs,
    tdamp=100*fs,
    trajectory=str(out_dir / f"md_{filename}_{model_name}_{temp_int}.traj"),
    logfile=str(out_dir / f"production_{filename}_{model_name}_{temp_int}.log"),
    loginterval=50,
)
md.run(100000000)  # 100 ns
