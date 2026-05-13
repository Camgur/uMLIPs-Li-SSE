import numpy as np
from ase.io import read
import glob, os
from pathlib import Path

# MSD computation parameters
min_dt_frac = 0.1
max_dt_frac = 0.5

def compute_msd(positions, dt):
    """Compute ensemble-averaged MSD using the windowed algorithm."""
    n_frames, n_atoms, _ = positions.shape
    min_lag = max(1, int(min_dt_frac * n_frames))
    max_lag = int(max_dt_frac * n_frames)
    
    lag_times = []
    msd_values = []
    
    for lag in range(min_lag, max_lag + 1):
        displacements = positions[lag:] - positions[:-lag]
        sq_disp = np.sum(displacements**2, axis=-1)
        msd = sq_disp.mean()
        lag_times.append(lag * dt)
        msd_values.append(msd)
    
    return np.array(lag_times), np.array(msd_values)

def process_trajectory(traj_path, dt):
    """Load trajectory, extract Li positions, compute MSD."""
    frames = read(str(traj_path), index=":")
    if not frames:
        raise ValueError(f"Empty trajectory: {traj_path}")
    
    li_indices = [i for i, sym in enumerate(frames[0].get_chemical_symbols()) if sym == "Li"]
    if not li_indices:
        raise ValueError("No Li atoms found in trajectory.")
    
    print(f"  Found {len(li_indices)} Li atoms across {len(frames)} frames")
    
    positions = np.array([f.get_positions()[li_indices] for f in frames])
    return compute_msd(positions, dt)

# Process all trajectories
for traj_file in glob.glob("results/md/*/*/[0-9]*K/md.traj"):
    parts = traj_file.replace("\\", "/").split("/")
    material = parts[-4]
    model = parts[-3]
    temp_str = parts[-2]
    temp = int(temp_str.replace("K", ""))
    
    print(f"{material} {model} {temp} K")
    
    try:
        timestep_ps = 1.0
        frames_check = read(str(traj_file), index="0")
        if hasattr(frames_check, 'info') and 'timestep' in frames_check.info:
            timestep_ps = frames_check.info['timestep'] / 1000.0
        
        lag_times, msd = process_trajectory(traj_file, timestep_ps)
        
        out_path = Path(traj_file).parent / "msd.npz"
        np.savez(str(out_path), lag_times_ps=lag_times, msd_A2=msd)
        print(f"  Saved to {out_path}")
        
    except Exception as exc:
        print(f"  ERROR: {exc}")
