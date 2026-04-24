"""md/compute_msd.py — Compute mean-square displacements from MD trajectories.

For each temperature trajectory the Li-ion MSD is computed and saved to a
NumPy ``.npz`` file for use by ``arrhenius_fit.py``.

Usage
-----
    python scripts/md/compute_msd.py --material LLZO --model MACE-MP-0
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
from ase.io import read

import sys
sys.path.insert(0, str(Path(__file__).parents[1]))

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

REPO_ROOT = Path(__file__).parents[2]
MD_RESULTS_DIR = REPO_ROOT / "results" / "md"
DEFAULT_CONFIG = REPO_ROOT / "configs" / "md_params.yaml"


def compute_msd(
    positions: np.ndarray,
    dt: float,
    min_dt_frac: float = 0.1,
    max_dt_frac: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the ensemble-averaged MSD using the windowed algorithm.

    Parameters
    ----------
    positions:
        Array of shape ``(n_frames, n_atoms, 3)`` of unwrapped Li positions.
    dt:
        MD timestep between saved frames [ps].
    min_dt_frac, max_dt_frac:
        Fraction of total trajectory length defining the lag-time window.

    Returns
    -------
    lag_times_ps: np.ndarray
        Lag times in picoseconds.
    msd_A2: np.ndarray
        MSD values in Å².
    """
    n_frames, n_atoms, _ = positions.shape
    min_lag = max(1, int(min_dt_frac * n_frames))
    max_lag = int(max_dt_frac * n_frames)

    lag_times = []
    msd_values = []

    for lag in range(min_lag, max_lag + 1):
        displacements = positions[lag:] - positions[:-lag]  # (n_windows, n_atoms, 3)
        sq_disp = np.sum(displacements**2, axis=-1)         # (n_windows, n_atoms)
        msd = sq_disp.mean()
        lag_times.append(lag * dt)
        msd_values.append(msd)

    return np.array(lag_times), np.array(msd_values)


def process_trajectory(
    traj_path: Path,
    timestep_ps: float,
    min_dt_frac: float,
    max_dt_frac: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Load a trajectory, extract Li positions, and return MSD curve."""
    logger.info("Reading trajectory: %s", traj_path)
    frames = read(str(traj_path), index=":")
    if not frames:
        raise ValueError(f"Empty trajectory: {traj_path}")

    # Identify Li atom indices from first frame
    li_indices = [i for i, sym in enumerate(frames[0].get_chemical_symbols()) if sym == "Li"]
    if not li_indices:
        raise ValueError("No Li atoms found in trajectory.")
    logger.info("  Found %d Li atoms across %d frames.", len(li_indices), len(frames))

    positions = np.array([f.get_positions()[li_indices] for f in frames])  # unwrap if possible
    return compute_msd(positions, timestep_ps, min_dt_frac, max_dt_frac)


def main() -> None:
    import yaml

    parser = argparse.ArgumentParser(description="Compute Li MSD from MD trajectories.")
    parser.add_argument("--material", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)["md"]

    timestep_ps = cfg["timestep_fs"] * cfg.get("traj_interval", 100) * 1e-3
    min_dt_frac = cfg.get("msd", {}).get("min_dt_fraction", 0.1) if "msd" in cfg else 0.1
    max_dt_frac = cfg.get("msd", {}).get("max_dt_fraction", 0.5) if "msd" in cfg else 0.5

    base_dir = MD_RESULTS_DIR / args.material / args.model
    temp_dirs = sorted(base_dir.glob("*K"))

    if not temp_dirs:
        logger.error("No temperature directories found in %s", base_dir)
        return

    for temp_dir in temp_dirs:
        traj_path = temp_dir / "md.traj"
        if not traj_path.exists():
            logger.warning("Trajectory not found: %s — skipping.", traj_path)
            continue
        try:
            lag_times, msd = process_trajectory(traj_path, timestep_ps, min_dt_frac, max_dt_frac)
        except Exception as exc:
            logger.error("MSD failed for %s: %s", temp_dir.name, exc)
            continue

        out_path = temp_dir / "msd.npz"
        np.savez(str(out_path), lag_times_ps=lag_times, msd_A2=msd)
        logger.info("MSD saved to %s", out_path)


if __name__ == "__main__":
    main()
