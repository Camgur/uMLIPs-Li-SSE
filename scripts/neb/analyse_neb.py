"""neb/analyse_neb.py — Extract and compare NEB migration barriers.

Reads the converged NEB trajectory for each (material, model) pair, extracts
the maximum energy along the path (the migration barrier), and compares it to
the DFT reference stored in ``configs/neb_params.yaml``.

Outputs a CSV summary to ``results/neb/barriers_summary.csv``.

Usage
-----
    python scripts/neb/analyse_neb.py --material LLZO
    python scripts/neb/analyse_neb.py  # analyse all materials
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

import numpy as np
import yaml
from ase.io import read

import sys
sys.path.insert(0, str(Path(__file__).parents[1]))
from utils.models import list_models

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

REPO_ROOT = Path(__file__).parents[2]
NEB_RESULTS_DIR = REPO_ROOT / "results" / "neb"
NEB_CONFIG = REPO_ROOT / "configs" / "neb_params.yaml"


def extract_barrier(traj_path: Path) -> float:
    """Return the migration barrier [eV] from a converged NEB trajectory.

    The barrier is the maximum energy relative to the first image.

    Parameters
    ----------
    traj_path:
        Path to the NEB trajectory file (ASE traj format).

    Returns
    -------
    float
        Migration barrier in eV.
    """
    images = read(str(traj_path), index=":")
    # Take the last full NEB band (one frame per image at the final step)
    n_images = len(images)
    # Each optimisation step writes n_images frames; take the last band
    last_band = images[-n_images:]
    energies = np.array([img.get_potential_energy() for img in last_band])
    return float(energies.max() - energies[0])


def analyse_material(material: str, models: list[str], ref_barriers: dict) -> list[dict]:
    """Return barrier comparison rows for *material*."""
    rows = []
    ref = ref_barriers.get(material)
    tol = 0.10  # default tolerance

    for model in models:
        traj_path = NEB_RESULTS_DIR / material / model / "neb.traj"
        if not traj_path.exists():
            logger.warning("No NEB trajectory for %s / %s — skipping.", material, model)
            continue
        try:
            barrier = extract_barrier(traj_path)
        except Exception as exc:
            logger.error("Could not extract barrier for %s / %s: %s", material, model, exc)
            continue

        delta = barrier - ref if ref is not None else None
        passed = abs(delta) <= tol if delta is not None else None

        rows.append({
            "material": material,
            "model": model,
            "barrier_eV": round(barrier, 4),
            "dft_ref_eV": ref,
            "delta_eV": round(delta, 4) if delta is not None else None,
            "passes_screening": passed,
        })
        logger.info(
            "%s / %-12s  barrier=%.3f eV  ref=%.3f eV  Δ=%+.3f eV  pass=%s",
            material, model, barrier,
            ref if ref is not None else float("nan"),
            delta if delta is not None else float("nan"),
            passed,
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyse NEB barriers.")
    parser.add_argument("--material", default=None, help="Material name. If omitted, all are analysed.")
    args = parser.parse_args()

    with open(NEB_CONFIG) as fh:
        cfg = yaml.safe_load(fh)
    ref_barriers: dict = cfg.get("reference", {}).get("barriers", {})

    models = list_models()

    if args.material:
        materials = [args.material]
    else:
        materials = [p.name for p in NEB_RESULTS_DIR.iterdir() if p.is_dir()]

    all_rows: list[dict] = []
    for mat in materials:
        all_rows.extend(analyse_material(mat, models, ref_barriers))

    if not all_rows:
        logger.warning("No results found.")
        return

    out_path = NEB_RESULTS_DIR / "barriers_summary.csv"
    fieldnames = ["material", "model", "barrier_eV", "dft_ref_eV", "delta_eV", "passes_screening"]
    with open(out_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    logger.info("Summary written to %s", out_path)


if __name__ == "__main__":
    main()
