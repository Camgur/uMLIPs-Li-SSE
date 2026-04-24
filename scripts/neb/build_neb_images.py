"""neb/build_neb_images.py — Build NEB image sets for Li migration barriers.

For each material the script:
  1. Loads the DFT-relaxed (or uMLIP-relaxed) endpoint structures.
  2. Identifies the Li migration pathway (initial → final site).
  3. Generates N linearly-interpolated intermediate images.
  4. Writes all images to ``results/neb/<material>/<model>/images/``.

Usage
-----
    python scripts/neb/build_neb_images.py --material LLZO
    python scripts/neb/build_neb_images.py --material LLZO --model MACE-MP-0 --n_images 7
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from ase.io import write

import sys
sys.path.insert(0, str(Path(__file__).parents[1]))
from utils.io import load_structure, ensure_dir

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

REPO_ROOT = Path(__file__).parents[2]
RELAXED_DIR = REPO_ROOT / "results" / "relaxed"
NEB_RESULTS_DIR = REPO_ROOT / "results" / "neb"


def build_images(
    initial: object,
    final: object,
    n_images: int,
    method: str = "linear",
) -> list:
    """Return a list of *n_images* + 2 images (endpoints included).

    Parameters
    ----------
    initial, final:
        ASE Atoms objects for the NEB endpoints.
    n_images:
        Number of intermediate images to generate.
    method:
        Interpolation method: ``"linear"`` or ``"idpp"``.
    """
    from ase.neb import NEB

    images = [initial.copy() for _ in range(n_images + 2)]
    images[-1] = final.copy()
    neb = NEB(images)
    neb.interpolate(method=method)
    return images


def main() -> None:
    parser = argparse.ArgumentParser(description="Build NEB image sets.")
    parser.add_argument("--material", required=True, help="Material name.")
    parser.add_argument("--model", default="DFT", help="Model whose relaxed structures are used.")
    parser.add_argument("--n_images", type=int, default=5, help="Number of intermediate images.")
    parser.add_argument("--method", default="linear", choices=["linear", "idpp"])
    args = parser.parse_args()

    relaxed_dir = RELAXED_DIR / args.material / args.model
    initial_path = relaxed_dir / "initial" / "POSCAR_relaxed"
    final_path = relaxed_dir / "final" / "POSCAR_relaxed"

    if not initial_path.exists() or not final_path.exists():
        raise FileNotFoundError(
            f"Endpoint structures not found in {relaxed_dir}. "
            "Ensure 'initial/' and 'final/' sub-directories exist with POSCAR_relaxed files."
        )

    logger.info("Loading endpoints from %s", relaxed_dir)
    initial = load_structure(initial_path)
    final = load_structure(final_path)

    logger.info("Generating %d intermediate images (method=%s)", args.n_images, args.method)
    images = build_images(initial, final, args.n_images, args.method)

    out_dir = ensure_dir(NEB_RESULTS_DIR / args.material / args.model / "images")
    for i, img in enumerate(images):
        write(str(out_dir / f"image_{i:02d}.traj"), img)
    logger.info("Wrote %d images to %s", len(images), out_dir)


if __name__ == "__main__":
    main()
