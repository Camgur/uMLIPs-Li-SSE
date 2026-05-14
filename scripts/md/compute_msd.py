import glob, re
from pathlib import Path

import numpy as np
from ase import io

import sys
sys.path.insert(0, str(Path(__file__).parents[1]))
from utils.config import ELEMENT, TIMESTEP_PS, MD_RESULTS_DIR

def compute_msd(positions, dt):
    """Compute MSD using Einstein Relation (displacement from initial frame)."""
    n_frames, n_atoms, _ = positions.shape
    msd = np.zeros(n_frames)
    for t in range(1, n_frames):
        displacement = positions[t] - positions[0]
        squared_displacement = np.sum(displacement**2, axis=1)
        msd[t] = np.mean(squared_displacement)
        if t % 1000 == 0:
            print(f"    Frame {t}/{n_frames}")
    lag_times = np.linspace(0, (n_frames - 1) * dt, n_frames)
    return lag_times, msd

def process_trajectory(traj_path, dt):
    """Load trajectory, extract mobile ions, compute MSD."""
    traj = io.Trajectory(str(traj_path))
    n_frames = len(traj)
    if n_frames == 0:
        raise ValueError(f"Empty trajectory: {traj_path}")
    
    # Extract ion indices
    atom_indices = [i for i, atom in enumerate(traj[0]) if atom.symbol == ELEMENT]
    if not atom_indices:
        raise ValueError(f"No {ELEMENT} atoms found in trajectory.")
    print(f"  Found {len(atom_indices)} {ELEMENT} atoms across {n_frames} frames")
    
    # Extract positions
    positions = np.zeros((n_frames, len(atom_indices), 3))
    for i, frame in enumerate(traj):
        positions[i] = frame.positions[atom_indices]
    return compute_msd(positions, dt)

# Process all trajectories
for traj_file in glob.glob(str(MD_RESULTS_DIR / "*" / "*" / "md_*.traj")):
    parts = traj_file.replace("\\", "/").split("/")
    material = parts[-3]
    model = parts[-2]
    filename = parts[-1]
    
    # Extract temperature from filename using regex
    match = re.search(r"md_.*_(\d+)\.traj", filename)
    if not match:
        print(f"Skipping {traj_file} (invalid filename format)")
        continue
    temp = int(match.group(1))
    
    # Check if output already exists
    out_path = Path(traj_file).parent / f"{material}_{model}_{temp}.npz"
    if out_path.exists():
        print(f"Skipping {material} {model} {temp} K (already exists)")
        continue
    
    print(f"{material} {model} {temp} K")
    
    try:
        dt = TIMESTEP_PS
        traj = io.Trajectory(str(traj_file))
        if traj[0].info and 'timestep' in traj[0].info:
            dt = traj[0].info['timestep'] / 1000.0
        
        lag_times, msd = process_trajectory(traj_file, dt)
        
        np.savez_compressed(str(out_path), lag_times_ps=lag_times, msd_A2=msd)
        print(f"  Saved: {out_path}")
        
    except Exception as exc:
        print(f"  ERROR: {exc}")
