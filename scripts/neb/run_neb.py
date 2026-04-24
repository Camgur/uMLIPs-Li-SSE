"""neb/run_neb.py — Run NEB calculations with a uMLIP force engine.

Usage
-----
    python scripts/neb/run_neb.py --material LLZO --model mace-0b3
    python scripts/neb/run_neb.py --material LLZO --model chgnet-2024 --config configs/neb_params.yaml

Available model names are defined in ``scripts/utils/models.py`` (``CALCULATOR_BLOCKS``).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import yaml
from ase.io import read
from ase.neb import NEB
from ase.optimize import BFGS

import sys
sys.path.insert(0, str(Path(__file__).parents[1]))
from utils.io import ensure_dir
from utils.models import get_calculator

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

REPO_ROOT = Path(__file__).parents[2]
NEB_RESULTS_DIR = REPO_ROOT / "results" / "neb"
DEFAULT_CONFIG = REPO_ROOT / "configs" / "neb_params.yaml"


def run_neb(material: str, model_name: str, config_path: Path) -> None:
    """Run a CI-NEB calculation for *material* using *model_name*.

    Parameters
    ----------
    material:
        Material name (must match a sub-directory in ``results/neb/``).
    model_name:
        uMLIP model name as defined in ``configs/models.yaml``.
    config_path:
        Path to the NEB parameters YAML file.
    """
    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)
    neb_cfg = cfg["neb"]
    opt_cfg = cfg["optimiser"]

    images_dir = NEB_RESULTS_DIR / material / model_name / "images"
    image_files = sorted(images_dir.glob("image_*.traj"))
    if not image_files:
        raise FileNotFoundError(
            f"No image files found in {images_dir}. "
            "Run build_neb_images.py first."
        )

    logger.info("Loading %d images from %s", len(image_files), images_dir)
    images = [read(str(p)) for p in image_files]

    # Attach calculators to all images (endpoints needed for energy evaluation)
    for img in images[1:-1]:
        img.calc = get_calculator(model_name)
    # Endpoints need a calculator for energy evaluation
    images[0].calc = get_calculator(model_name)
    images[-1].calc = get_calculator(model_name)

    neb = NEB(
        images,
        climb=neb_cfg.get("climb", True),
        k=neb_cfg.get("spring_constant", 0.5),
        method=neb_cfg.get("method", "aseneb"),
    )

    out_dir = ensure_dir(NEB_RESULTS_DIR / material / model_name)
    log_path = out_dir / "neb.log"
    traj_path = out_dir / "neb.traj"

    opt = BFGS(neb, logfile=str(log_path), trajectory=str(traj_path))
    fmax = opt_cfg.get("fmax", 0.05)
    steps = opt_cfg.get("steps", 500)

    logger.info("Running NEB optimisation — fmax=%.4f, max_steps=%d", fmax, steps)
    opt.run(fmax=fmax, steps=steps)
    logger.info("NEB complete. Trajectory saved to %s", traj_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run NEB with a uMLIP.")
    parser.add_argument("--material", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()

    run_neb(args.material, args.model, Path(args.config))


if __name__ == "__main__":
    main()
