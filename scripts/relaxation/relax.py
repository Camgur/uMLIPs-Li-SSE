"""relaxation/relax.py — Relax crystal structures with each uMLIP.

Usage
-----
    python scripts/relaxation/relax.py --material LLZO
    python scripts/relaxation/relax.py --material LLZO --model MACE-MP-0

The script reads input structures from ``structures/<material>/`` and writes
relaxed structures to ``results/relaxed/<material>/<model>/POSCAR``.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from ase.optimize import BFGS

# Project utilities (importable when running from repo root)
import sys
sys.path.insert(0, str(Path(__file__).parents[1]))
from utils.io import load_structure, save_structure, ensure_dir
from utils.models import get_calculator, list_models

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

REPO_ROOT = Path(__file__).parents[2]
STRUCTURES_DIR = REPO_ROOT / "structures"
RESULTS_DIR = REPO_ROOT / "results" / "relaxed"


def relax(material: str, model_name: str, fmax: float = 0.01) -> None:
    """Relax the input structure for *material* using *model_name*.

    Parameters
    ----------
    material:
        Sub-directory name inside ``structures/`` (e.g. ``"LLZO"``).
    model_name:
        uMLIP name as defined in ``configs/models.yaml``.
    fmax:
        Force convergence criterion [eV/Å].
    """
    struct_dir = STRUCTURES_DIR / material
    candidates = list(struct_dir.glob("*.cif")) + list(struct_dir.glob("POSCAR"))
    if not candidates:
        raise FileNotFoundError(
            f"No structure files found in {struct_dir}. "
            "Place a CIF or POSCAR file there before relaxing."
        )
    struct_path = candidates[0]
    logger.info("Loading structure: %s", struct_path)
    atoms = load_structure(struct_path)

    logger.info("Loading calculator: %s", model_name)
    calc = get_calculator(model_name)
    atoms.calc = calc

    out_dir = ensure_dir(RESULTS_DIR / material / model_name)
    traj_path = out_dir / "relax.traj"
    log_path = out_dir / "relax.log"

    logger.info("Starting relaxation — fmax=%.4f eV/Å", fmax)
    opt = BFGS(atoms, trajectory=str(traj_path), logfile=str(log_path))
    opt.run(fmax=fmax)

    out_path = out_dir / "POSCAR_relaxed"
    save_structure(atoms, out_path, fmt="vasp")
    logger.info("Relaxed structure saved to %s", out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Relax structures with uMLIPs.")
    parser.add_argument("--material", required=True, help="Material sub-directory name.")
    parser.add_argument(
        "--model",
        default=None,
        help="uMLIP model name. If omitted, all configured models are used.",
    )
    parser.add_argument("--fmax", type=float, default=0.01, help="Force threshold [eV/Å].")
    args = parser.parse_args()

    models = [args.model] if args.model else list_models()
    for model in models:
        logger.info("=== Relaxing %s with %s ===", args.material, model)
        try:
            relax(args.material, model, fmax=args.fmax)
        except Exception as exc:
            logger.error("Failed for %s / %s: %s", args.material, model, exc)


if __name__ == "__main__":
    main()
