"""dos/run_dos.py — Compute DOS on uMLIP-relaxed structures via single-point DFT.

This script generates VASP input files (INCAR, KPOINTS, POSCAR, POTCAR link)
for a static DOS calculation on the structure relaxed by each uMLIP.  It is
designed to be run on an HPC cluster where VASP is available.

If you are using a different DFT code, adapt the input-file generation section.

Usage
-----
    python scripts/dos/run_dos.py --material LLZO
    python scripts/dos/run_dos.py --material LLZO --model MACE-MP-0
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from shutil import copy2

import sys
sys.path.insert(0, str(Path(__file__).parents[1]))
from utils.io import load_structure, ensure_dir
from utils.models import list_models

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

REPO_ROOT = Path(__file__).parents[2]
RELAXED_DIR = REPO_ROOT / "results" / "relaxed"
DOS_RESULTS_DIR = REPO_ROOT / "results" / "dos"

INCAR_TEMPLATE = """\
SYSTEM  = DOS calculation on uMLIP-relaxed structure
ISTART  = 0
ICHARG  = 2
ENCUT   = 520
EDIFF   = 1E-6
NSW     = 0
IBRION  = -1
ISMEAR  = -5
SIGMA   = 0.05
LORBIT  = 11
NEDOS   = 2000
LWAVE   = .FALSE.
LCHARG  = .FALSE.
"""

KPOINTS_TEMPLATE = """\
Automatic mesh
0
Gamma
 4 4 4
 0 0 0
"""


def generate_vasp_inputs(material: str, model_name: str) -> Path:
    """Write VASP static-DOS input files; return the calculation directory."""
    struct_path = RELAXED_DIR / material / model_name / "POSCAR_relaxed"
    if not struct_path.exists():
        raise FileNotFoundError(
            f"Relaxed structure not found: {struct_path}. Run relax.py first."
        )

    calc_dir = ensure_dir(DOS_RESULTS_DIR / material / model_name)

    # Copy relaxed structure as POSCAR
    copy2(str(struct_path), str(calc_dir / "POSCAR"))

    (calc_dir / "INCAR").write_text(INCAR_TEMPLATE)
    (calc_dir / "KPOINTS").write_text(KPOINTS_TEMPLATE)

    # POTCAR must be assembled externally (licensing restrictions).
    (calc_dir / "POTCAR.note").write_text(
        "Place the appropriate POTCAR file here before running VASP.\n"
    )

    submit_script = calc_dir / "submit.sh"
    submit_script.write_text(
        f"#!/bin/bash\n# Submit VASP DOS calculation for {material} / {model_name}\nvasp_std\n"
    )
    submit_script.chmod(0o755)

    logger.info("VASP inputs written to %s", calc_dir)
    return calc_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate VASP DOS inputs for uMLIP-relaxed structures.")
    parser.add_argument("--material", required=True)
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    models = [args.model] if args.model else list_models()
    for model in models:
        try:
            generate_vasp_inputs(args.material, model)
        except FileNotFoundError as exc:
            logger.warning(str(exc))


if __name__ == "__main__":
    main()
