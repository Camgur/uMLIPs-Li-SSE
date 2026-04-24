"""neb/run_neb.py — Run NEB calculations with a uMLIP force engine.

Usage
-----
    python scripts/neb/run_neb.py structure.cif idx1 idx2 model_name

Available model names are defined in ``scripts/utils/models.py`` (``CALCULATOR_BLOCKS``).
"""

import warnings
warnings.simplefilter("ignore", category=FutureWarning)

import sys, os
import numpy as np
from pathlib import Path

from ase.io import read, write
from ase.optimize import BFGS
from ase.neb import NEB

sys.path.insert(0, str(Path(__file__).parents[1]))
from utils.io import ensure_dir
from utils.models import get_calculator

# ------------------------ Input handling ------------------------
if len(sys.argv) != 5:
    sys.exit("Usage: python run_neb.py structure.cif idx1 idx2 model_name")

cif_path, idx1, idx2, model_name = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
filename = os.path.splitext(os.path.basename(cif_path))[0]

atoms = read(cif_path)
if "Li" not in atoms.get_chemical_symbols():
    sys.exit("No lithium atoms found in structure.")

if idx1 >= len(atoms) or idx2 >= len(atoms):
    sys.exit("Indices out of range.")
print(f"Total atoms: {len(atoms)}, removing Li at indices {idx1}, {idx2}")

# ------------------------ Initial/final setup ------------------------
initial = atoms.copy()
final = atoms.copy()

if idx1 < idx2:
    del initial[idx2]
    del final[idx1]
else:
    del initial[idx1]
    del final[idx2]

from ase.build import sort
initial = sort(initial)
final = sort(final)

# ------------------------ Calculator ------------------------
calculator = get_calculator(model_name)

REPO_ROOT = Path(__file__).parents[2]
out_dir = ensure_dir(REPO_ROOT / "results" / "neb" / filename / model_name)

# ------------------------ Optimise endpoints ------------------------
for state, label in zip([initial, final], ["init", "fin"]):
    state.calc = calculator
    opt = BFGS(
        state,
        trajectory=str(out_dir / f"{label}_{idx1}to{idx2}.traj"),
        logfile=str(out_dir / f"{label}_{idx1}to{idx2}.log"),
    )
    opt.run(fmax=1e-3, steps=100)
    write(str(out_dir / f"{label}_{idx1}to{idx2}_opt.cif"), state)

# ------------------------ NEB setup ------------------------
# Create three interpolated images between initial/final
images = [initial, initial.copy(), initial.copy(), initial.copy(), final]
neb = NEB(images, climb=True, allow_shared_calculator=True)
neb.interpolate(method='idpp', mic=True)

for image in images:
    image.calc = calculator

# ------------------------ NEB optimisation ------------------------
traj_path = out_dir / "neb.traj"
neb_opt = BFGS(
    neb,
    trajectory=str(traj_path),
    logfile=str(out_dir / f"{idx1}to{idx2}_neb.log"),
)
neb_opt.run(fmax=0.05, steps=300)

print("NEB optimisation finished.")

# ------------------------ NEB analysis ------------------------
neb_traj = read(str(traj_path) + "@-5:")
for image in neb_traj:
    image.calc = calculator
energies = [image.get_potential_energy() for image in neb_traj]

energies_path = out_dir / f"{idx1}to{idx2}_neb_energies.txt"
with open(energies_path, "w") as f:
    f.writelines(f"{e:.6f}\n" for e in energies)

print("Energies [eV]:", energies)
print(f"Data saved to {energies_path}")
