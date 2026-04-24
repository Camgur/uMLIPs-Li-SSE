"""md/arrhenius_fit.py — Fit the Arrhenius equation to Li diffusivity data.

For each (material, model) pair the script:
  1. Reads the MSD data for each temperature.
  2. Fits D(T) = MSD / (6*t) via linear regression on the MSD-vs-time curve.
  3. Fits ln(D) vs 1/T to extract Ea and D0.
  4. Estimates room-temperature ionic conductivity via the Nernst-Einstein equation.
  5. Saves results to ``results/arrhenius/<material>/<model>/arrhenius.json``.

Usage
-----
    python scripts/md/arrhenius_fit.py --material LLZO --model MACE-MP-0
    python scripts/md/arrhenius_fit.py --material LLZO  # all passing models
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from scipy import stats

import sys
sys.path.insert(0, str(Path(__file__).parents[1]))
from utils.io import ensure_dir

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

REPO_ROOT = Path(__file__).parents[2]
MD_RESULTS_DIR = REPO_ROOT / "results" / "md"
ARRHENIUS_RESULTS_DIR = REPO_ROOT / "results" / "arrhenius"

KB_EV = 8.617333262e-5   # Boltzmann constant [eV/K]
E_CHARGE = 1.602176634e-19  # Elementary charge [C]
NA = 6.02214076e23          # Avogadro constant


def fit_diffusivity(lag_times_ps: np.ndarray, msd_A2: np.ndarray) -> float:
    """Return self-diffusion coefficient D [m²/s] from MSD vs lag-time data.

    Linear regression of MSD = 6 D t  →  slope = 6D.

    Parameters
    ----------
    lag_times_ps:
        Lag times in picoseconds.
    msd_A2:
        MSD values in Å².

    Returns
    -------
    D_m2s: float
        Diffusion coefficient in m²/s.
    """
    # Convert units: ps → s, Å² → m²
    lag_times_s = lag_times_ps * 1e-12
    msd_m2 = msd_A2 * 1e-20

    slope, _, _, _, _ = stats.linregress(lag_times_s, msd_m2)
    D = slope / 6.0
    return float(D)


def fit_arrhenius(
    temperatures_K: np.ndarray, diffusivities_m2s: np.ndarray
) -> dict[str, float]:
    """Fit the Arrhenius equation and return Ea, D0, and R².

    Parameters
    ----------
    temperatures_K:
        Array of temperatures.
    diffusivities_m2s:
        Array of diffusion coefficients at each temperature.

    Returns
    -------
    dict with keys: Ea_eV, D0_m2s, R2
    """
    inv_T = 1.0 / temperatures_K
    ln_D = np.log(diffusivities_m2s)

    slope, intercept, r_value, _, _ = stats.linregress(inv_T, ln_D)
    Ea = -slope * KB_EV  # eV
    D0 = float(np.exp(intercept))  # m²/s

    return {"Ea_eV": float(Ea), "D0_m2s": D0, "R2": float(r_value**2)}


def nernst_einstein_conductivity(
    D_m2s: float,
    n_Li_per_m3: float,
    T_K: float = 298.15,
) -> float:
    """Estimate ionic conductivity [S/m] at *T_K* via Nernst-Einstein.

    σ = n z² e² D / (k_B T)

    Parameters
    ----------
    D_m2s:
        Li self-diffusion coefficient [m²/s].
    n_Li_per_m3:
        Li carrier density [m⁻³].
    T_K:
        Temperature [K] (default: room temperature).
    """
    z = 1  # Li charge number
    kBT = 1.380649e-23 * T_K  # J
    sigma = n_Li_per_m3 * (z * E_CHARGE) ** 2 * D_m2s / kBT
    return float(sigma)


def process_model(material: str, model_name: str) -> dict | None:
    """Extract Arrhenius parameters for one (material, model) pair."""
    base_dir = MD_RESULTS_DIR / material / model_name
    temp_dirs = sorted(base_dir.glob("*K"))

    temperatures = []
    diffusivities = []

    for temp_dir in temp_dirs:
        msd_path = temp_dir / "msd.npz"
        if not msd_path.exists():
            continue
        T = float(temp_dir.name.replace("K", ""))
        data = np.load(str(msd_path))
        try:
            D = fit_diffusivity(data["lag_times_ps"], data["msd_A2"])
        except Exception as exc:
            logger.warning("D fitting failed at %g K: %s", T, exc)
            continue
        temperatures.append(T)
        diffusivities.append(D)
        logger.info("  T=%g K  D=%.3e m²/s", T, D)

    if len(temperatures) < 2:
        logger.warning("Insufficient data points for Arrhenius fit (%d).", len(temperatures))
        return None

    result = fit_arrhenius(np.array(temperatures), np.array(diffusivities))
    result["temperatures_K"] = temperatures
    result["diffusivities_m2s"] = diffusivities
    logger.info(
        "%s / %s  Ea=%.3f eV  D0=%.3e m²/s  R²=%.4f",
        material, model_name, result["Ea_eV"], result["D0_m2s"], result["R2"],
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit Arrhenius equation to Li diffusivity.")
    parser.add_argument("--material", required=True)
    parser.add_argument("--model", default=None, help="Model name. If omitted, all are processed.")
    args = parser.parse_args()

    if args.model:
        models = [args.model]
    else:
        base = MD_RESULTS_DIR / args.material
        models = [p.name for p in base.iterdir() if p.is_dir()] if base.exists() else []

    for model in models:
        result = process_model(args.material, model)
        if result is None:
            continue
        out_dir = ensure_dir(ARRHENIUS_RESULTS_DIR / args.material / model)
        out_path = out_dir / "arrhenius.json"
        with open(out_path, "w") as fh:
            json.dump(result, fh, indent=2)
        logger.info("Saved to %s", out_path)


if __name__ == "__main__":
    main()
