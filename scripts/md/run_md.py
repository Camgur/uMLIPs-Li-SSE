"""md/run_md.py — Run NVT molecular dynamics with a uMLIP.

For each temperature specified in ``configs/md_params.yaml`` the script runs a
production MD simulation and saves the trajectory for subsequent MSD analysis.

Usage
-----
    python scripts/md/run_md.py --material LLZO --model MACE-MP-0
    python scripts/md/run_md.py --material LLZO --model MACE-MP-0 --temp 800
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import yaml
from ase import units
from ase.io.trajectory import Trajectory
from ase.md.langevin import Langevin

import sys
sys.path.insert(0, str(Path(__file__).parents[1]))
from utils.io import load_structure, ensure_dir
from utils.models import get_calculator

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

REPO_ROOT = Path(__file__).parents[2]
RELAXED_DIR = REPO_ROOT / "results" / "relaxed"
MD_RESULTS_DIR = REPO_ROOT / "results" / "md"
DEFAULT_CONFIG = REPO_ROOT / "configs" / "md_params.yaml"


def run_md(material: str, model_name: str, temperature_K: float, config_path: Path) -> None:
    """Run NVT MD at *temperature_K* for *material* with *model_name*.

    Parameters
    ----------
    material, model_name:
        Identify the system.
    temperature_K:
        Simulation temperature in Kelvin.
    config_path:
        Path to the MD parameters YAML file.
    """
    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)["md"]

    # Load relaxed structure
    struct_path = RELAXED_DIR / material / model_name / "POSCAR_relaxed"
    if not struct_path.exists():
        raise FileNotFoundError(
            f"Relaxed structure not found at {struct_path}. "
            "Run relax.py first."
        )
    atoms = load_structure(struct_path)
    atoms.calc = get_calculator(model_name)

    timestep = cfg["timestep_fs"] * units.fs
    friction = cfg["friction"]
    equil_steps = cfg["equilibration_steps"]
    prod_steps = cfg["production_steps"]
    traj_interval = cfg["traj_interval"]

    out_dir = ensure_dir(MD_RESULTS_DIR / material / model_name / f"{int(temperature_K)}K")
    traj_path = out_dir / "md.traj"
    log_path = out_dir / "md.log"

    dyn = Langevin(
        atoms,
        timestep=timestep,
        temperature_K=temperature_K,
        friction=friction,
        logfile=str(log_path),
        loginterval=cfg.get("log_interval", 100),
    )

    logger.info("Equilibrating for %d steps at %d K…", equil_steps, temperature_K)
    dyn.run(equil_steps)

    logger.info("Production run: %d steps — saving every %d steps", prod_steps, traj_interval)
    traj = Trajectory(str(traj_path), "w", atoms)
    dyn.attach(traj.write, interval=traj_interval)
    dyn.run(prod_steps)
    traj.close()
    logger.info("MD trajectory saved to %s", traj_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run NVT MD with a uMLIP.")
    parser.add_argument("--material", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--temp",
        type=float,
        default=None,
        help="Single temperature [K]. If omitted, all temperatures from config are used.",
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()

    config_path = Path(args.config)
    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)["md"]

    temperatures = [args.temp] if args.temp else cfg["temperatures_K"]

    for T in temperatures:
        logger.info("=== %s / %s @ %g K ===", args.material, args.model, T)
        try:
            run_md(args.material, args.model, float(T), config_path)
        except Exception as exc:
            logger.error("Failed: %s", exc)


if __name__ == "__main__":
    main()
