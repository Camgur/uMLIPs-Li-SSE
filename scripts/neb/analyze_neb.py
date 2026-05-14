from pathlib import Path

from ase.io import read
from ase.neb import NEBTools

import sys
sys.path.insert(0, str(Path(__file__).parents[1]))


def extract_barrier(traj_path):
    """Extract NEB barrier height from converged trajectory."""
    images = read(str(traj_path), index=":")
    neb_tools = NEBTools(images)
    return float(neb_tools.get_barrier()[0])
