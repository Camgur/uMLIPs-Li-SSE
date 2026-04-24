"""dos/compare_dos.py — Compare DOS curves between uMLIP-relaxed and DFT-reference structures.

Reads VASP DOSCAR files from ``results/dos/<material>/`` and computes a
cosine-similarity score against the DFT reference DOS.  Results are written to
``results/dos/dos_similarity_summary.csv``.

Usage
-----
    python scripts/dos/compare_dos.py --material LLZO
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parents[1]))
from utils.models import list_models

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

REPO_ROOT = Path(__file__).parents[2]
DOS_RESULTS_DIR = REPO_ROOT / "results" / "dos"


def parse_doscar(doscar_path: Path, n_dos_points: int = 2000) -> np.ndarray:
    """Parse a VASP DOSCAR and return the total DOS array.

    Parameters
    ----------
    doscar_path:
        Path to the VASP DOSCAR file.
    n_dos_points:
        Number of DOS energy grid points (must match NEDOS in INCAR).

    Returns
    -------
    dos: np.ndarray
        Array of shape ``(n_dos_points, 2)`` with columns [energy, total_dos].
    """
    lines = doscar_path.read_text().splitlines()
    # Header: 6 lines; first DOS block starts at line 6
    header_lines = 6
    dos_lines = lines[header_lines: header_lines + n_dos_points]
    dos = np.array([[float(x) for x in ln.split()[:2]] for ln in dos_lines])
    return dos


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Return the cosine similarity between vectors *a* and *b*."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def compare_material(material: str, models: list[str]) -> list[dict]:
    """Return DOS cosine-similarity rows for *material*."""
    ref_doscar = DOS_RESULTS_DIR / material / "DFT" / "DOSCAR"
    if not ref_doscar.exists():
        logger.warning("DFT reference DOSCAR not found: %s — skipping %s.", ref_doscar, material)
        return []

    try:
        ref_dos = parse_doscar(ref_doscar)
    except Exception as exc:
        logger.error("Failed to parse reference DOSCAR for %s: %s", material, exc)
        return []

    ref_vec = ref_dos[:, 1]  # total DOS values
    rows = []

    for model in models:
        model_doscar = DOS_RESULTS_DIR / material / model / "DOSCAR"
        if not model_doscar.exists():
            logger.warning("DOSCAR missing for %s / %s — skipping.", material, model)
            continue
        try:
            model_dos = parse_doscar(model_doscar)
        except Exception as exc:
            logger.error("Failed to parse DOSCAR for %s / %s: %s", material, model, exc)
            continue

        similarity = cosine_similarity(ref_vec, model_dos[:, 1])
        rows.append({"material": material, "model": model, "dos_cosine_similarity": round(similarity, 6)})
        logger.info("%s / %-12s  cosine_similarity=%.4f", material, model, similarity)

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare DOS curves to DFT reference.")
    parser.add_argument("--material", default=None, help="Material name. If omitted, all are processed.")
    args = parser.parse_args()

    models = list_models()

    if args.material:
        materials = [args.material]
    else:
        materials = [p.name for p in DOS_RESULTS_DIR.iterdir() if p.is_dir()]

    all_rows: list[dict] = []
    for mat in materials:
        all_rows.extend(compare_material(mat, models))

    if not all_rows:
        logger.warning("No DOS comparison results.")
        return

    out_path = DOS_RESULTS_DIR / "dos_similarity_summary.csv"
    fieldnames = ["material", "model", "dos_cosine_similarity"]
    with open(out_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    logger.info("Summary written to %s", out_path)


if __name__ == "__main__":
    main()
