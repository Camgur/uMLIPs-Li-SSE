# uMLIPs Pipeline

A computational pipeline for benchmarking universal Machine Learning Interatomic Potentials (uMLIPs) for predicting activation energies of lithium-ion diffusion in solid-state electrolytes. Accompanies the following paper:

Cameron A. Gurwell, Taiana L. E. Pereira, Mengyang Cui, et al. Experimental Validation of Universal Machine Learning Interatomic Potentials for Lithium-Ion Dynamics in Solid Electrolytes via 7Li NMR.  *ChemRxiv* . 28 April 2026.
DOI: [https://doi.org/10.26434/chemrxiv.15002480/v1](https://doi.org/10.26434/chemrxiv.15002480/v1)

---

## Table of Contents

1. [Background](#1-background)
2. [Selected uMLIPs and Materials](#2-selected-umlips-and-materials)
   - 2.1 [Universal Machine Learning Interatomic Potentials](#21-universal-machine-learning-interatomic-potentials)
   - 2.2 [Example Materials](#22-example-materials)
3. [Methodology and Pipeline](#3-methodology-and-pipeline)
   - 3.1 [Pre-Screening: DOS and NEB Calculations](#31-pre-screening-dos-and-neb-calculations)
     - 3.1.1 [Density of States (DOS)](#311-density-of-states-dos)
     - 3.1.2 [Nudged Elastic Band (NEB)](#312-nudged-elastic-band-neb)
   - 3.2 [Lithium Diffusion Simulations](#32-lithium-diffusion-simulations)
     - 3.2.1 [Arrhenius Fitting](#321-arrhenius-fitting)
     - 3.2.2 [Comparison with Experimental ssNMR Data](#322-comparison-with-experimental-ssnmr-data)
4. [Repository Structure](#4-repository-structure)
5. [Installation and Dependencies](#5-installation-and-dependencies)

---

## 1. Background

Solid-state electrolytes (SSEs) are a key enabling technology for next-generation solid-state batteries, offering improved safety and energy density compared to conventional liquid electrolytes. A critical performance metric for SSEs is the activation energy (E<sub>a</sub>) for lithium-ion diffusion.

Traditionally, E<sub>a</sub> is extracted by fitting the Arrhenius equation to lithium diffusivity data obtained from molecular dynamics (MD) simulations at multiple temperatures. When performed with *ab initio* MD (AIMD), this approach is computationally prohibitive: simulations long enough to achieve statistical convergence of diffusion coefficients demand significant high-performance computing resources.

Despite decades of progress in pseudopotentials and density functional theory (DFT), atomistic simulations remain computationally demanding, with costs that scale poorly and limit their applicability to large systems. Universal machine-learning interatomic potentials (uMLIPs) offer a promising alternative, reducing these timescales to hours while retaining near–DFT accuracy.

Although uMLIPs offer significant computational savings over DFT-based approaches, their application to lithium diffusion remains nontrivial in cost. As a computationally efficient pre-screening step, DOS and nudged elastic band (NEB) calculations are used to identify high-performing uMLIPs, thereby restricting expensive long lithium diffusion MD simulations required for Arrhenius fitting to only the most promising candidates.

This work introduces a two-stage pipeline:

1. **Pre-screening** (low-cost): Density of States (DOS) and Nudged Elastic Band (NEB) calculations identify uMLIPs that accurately describe the electronic and energetic landscape of each material.
2. **Full evaluation** (high-cost): Only the most promising uMLIPs identified in 1 are used for long lithium-diffusion MD simulations and subsequent Arrhenius fitting, with final validation against experimental solid-state NMR (ssNMR) diffusion data.

---

## 2. Selected uMLIPs and Materials

### 2.1 Universal Machine Learning Interatomic Potentials

The following uMLIPs are evaluated in this study. All models are used without additional fine-tuning to assess their out-of-the-box transferability to lithium SSE chemistries.

| Family                                                                  | Model                                   | Alias           | Training Dataset   |
| ----------------------------------------------------------------------- | --------------------------------------- | --------------- | ------------------ |
| [***CHGNet-Torch***](https://github.com/CederGroupHub/chgnet)           | 0.3.0                                   | CHGNet-torch    | MPTrj              |
| [***CHGNet-MatGL***](https://github.com/materialsvirtuallab/matgl)      | CHGNet-MPtrj-2023.12.1-2.7M-PES         | CHGNet-2023     | MPTrj              |
|                                                                         | CHGNet-MPtrj-2024.2.13-11M-PES          | CHGNet-2024     | MPTrj              |
|                                                                         | CHGNet-MatPES-PBE-2025.2.10-2.7M-PES    | CHGNet-PBE-2025 | MatPES PBE         |
|                                                                         | CHGNet-MatPES-r2SCAN-2025.2.10-2.7M-PES | CHGNet-r2-2025  | MatPES r2SCAN      |
| [***M3GNet***](https://github.com/materialsvirtuallab/matgl)            | M3GNet-MP-2021.2.8-PES                  | M3GNet-2021     | MPF                |
|                                                                         | M3GNet-MatPES-PBE-v2025.1-PES           | M3GNet-PBE-2025 | MatPES PBE         |
|                                                                         | M3GNet-MatPES-r2SCAN-v2025.1-PES        | M3GNet-r2-2025  | MatPES r2SCAN      |
| [***MACE***](https://github.com/ACEsuit/mace)                           | MACE-MP-0b3-medium                      | MACE-0b3        | MPTrj              |
|                                                                         | 2024-01-07-mace-128-L2_epoch-199        | MACE-l2         | MPTrj              |
|                                                                         | MACE-MPA-0                              | MACE-MPA        | MPTrj + sAlex      |
|                                                                         | MACE-OMAT-0                             | MACE-OMAT       | OMAT24             |
|                                                                         | MACE-MATPES-r2SCAN-0                    | MACE-r2         | MatPES r2SCAN      |
| [***ORB***](https://github.com/orbital-materials/orb-models)            | orb-v2                                  | ORB-v2          | MPTrj + Alexandria |
|                                                                         | orb_v3_conservative_20_mpa              | ORB-v3-cMPA     | MPTrj + sAlex      |
|                                                                         | orb_v3_conservative_20_omat             | ORB-v3-cOMAT    | OMAT24             |
|                                                                         | orb_v3_conservative_inf_mpa             | ORB-v3-iMPA     | MPTrj + sAlex      |
|                                                                         | orb_v3_conservative_inf_omat            | ORB-v3-iOMAT    | OMAT24             |

### 2.2 Example Materials

This example benchmark focuses on the following lithium-based materials:

| Material                                      | Alias | Structure Type   | Li Conductivity Regime |
| --------------------------------------------- | ----- | ---------------- | ---------------------- |
| Li<sub>10</sub>SnP<sub>2</sub>S<sub>12</sub>  | LSnPS | Electrolyte      | High                   |
| LiFeV<sub>2</sub>O<sub>7</sub>                | LFVO  | Cathode          | Low–Moderate           |

---

## 3. Methodology and Pipeline

### 3.1 Pre-Screening: DOS and NEB Calculations

Pre-screening is performed at low computational cost to identify uMLIPs whose predicted energy surfaces are consistent with DFT reference calculations. Two complementary descriptors are used:

#### 3.1.1 Density of States (DOS)

The electronic DOS provides a fingerprint of a material's electronic structure. Although uMLIPs do not directly produce electronic wavefunctions, the forces and energies they predict reflect the underlying potential energy surface. Here, DOS (computed via single-point DFT on the respective uMLIP-relaxed structures) is used to assess whether structural relaxations by each uMLIP preserve key features of the DFT-reference geometry.

- Structures are relaxed with each uMLIP using the ASE `BFGS` optimiser.
- Single-point DFT (CASTEP/PBE) calculations are then performed on the relaxed geometries.
- Resulting DOS curves are compared to fully DFT-relaxed reference structures

#### 3.1.2 Nudged Elastic Band (NEB)

NEB calculations provide direct estimates of migration barriers for Li<sup>+</sup> hopping between adjacent sites. This makes NEB a powerful and cost-efficient pre-screening criterion.

- Initial and final images are taken from uMLIP-relaxed structures.
- Intermediate images are generated by linear interpolation.
- Each uMLIP is used as the force engine within ASE's NEB implementation.
- Barriers (E<sub>a</sub>) are compared against NEB reference barriers from DFT (CASTEP/PBE).

The best-performing models were included in the lithium diffusion simulations.

### 3.2 Lithium Diffusion Simulations

The best uMLIPs that pass the pre-screening stage proceed to long MD simulations designed to extract diffusion coefficients at multiple temperatures.

#### 3.2.1 Arrhenius Fitting

The mean-square displacement (MSD) of Li<sup>+</sup> is computed from MD trajectories at multiple temperatures (300-1200K for this work). The diffusion coefficient *D* is recovered from the Einstein relation:

$$
D = \frac{1}{6 N_{Li} \Delta t} \sum_{i=1}^{n} \langle |r_i(t + \Delta t) - r_i(t)|^2 \rangle
$$

The temperature dependence of *D* is then fitted to the Arrhenius equation:

$$
D = D_0 \exp \left(-\frac{E_a}{k_B T}\right)
$$

to extract E<sub>a</sub>.

#### 3.2.2 Comparison with Experimental ssNMR Data

Solid-state NMR (ssNMR) provides experimental activation energies and diffusion coefficients that are independent of any simulation model. Calculated E<sub>a</sub> values from the best uMLIPs are compared against ssNMR data for each material to provide a direct, quantitative validation.

---

## 4. Repository Structure

```
uMLIPs Pipeline/
│
├── README.md
│
├── structures/                  # Input crystal structures
│   ├── LSnPS/
│   └── LiFeV2O7/
│
├── scripts/
│   ├── relaxation/              # Structure relaxation with each uMLIP
│   │   └── run_relax.py
│   ├── dos/                     # DOS pre-screening
│   │   └── compare_dos.py
│   ├── neb/                     # NEB pre-screening
│   │   ├── run_neb.py
│   │   └── plot_neb.py          # NEB barrier analysis and visualization
│   ├── md/                      # MD simulations and MSD analysis
│   │   ├── run_md.py
│   │   ├── compute_msd.py
│   │   └── arrhenius_fit.py
│   └── utils/                   # Shared utilities
│       ├── config.py            # Centralized configuration (paths, parameters, models, colors, line styles)
│       ├── io.py                # I/O helpers for structure loading/saving
│       └── models.py            # uMLIP calculator loader via registry
│
├── results/
│   ├── dos/                     # DOS comparison plots and Excel exports
│   ├── neb/                     # NEB barrier results and analysis
│   └── md/                      # MD trajectories and MSD data
│       └── <material>/<model>/   # Temperature subdirectories with MD trajectories
│
├── notebooks/                   # Analysis and plotting notebooks
│   ├── 01_dos_screening.ipynb
│   ├── 02_neb_screening.ipynb
│   ├── 03_arrhenius_analysis.ipynb
│   └── 04_ssnmr_comparison.ipynb
│
└── Requirements.txt             # Environment specification
```

---

## 5. Installation and Dependencies

### Prerequisites

- Python ≥ 3.10
- [ASE](https://wiki.fysik.dtu.dk/ase/) ≥ 3.23
- [PyMatGen](https://pymatgen.org/) ≥ 2024.1
- [SUMO](https://github.com/smtg-ucl/sumo) ≥ 2.3 (for DOS analysis with VBM/CBM detection)
- [NumPy](https://numpy.org/), [SciPy](https://scipy.org/), [Matplotlib](https://matplotlib.org/)
- [Openpyxl](https://openpyxl.readthedocs.io/) (for Excel export in DOS comparison)
- Individual uMLIP packages (see `requirements.txt`)

### Supported uMLIP Packages

- **CHGNet**: Via `matgl` or `chgnet-torch`
- **M3GNet**: Via `matgl`
- **MACE**: Via `mace`
- **ORB**: Via `orb-models`

### Setup

```bash
# Clone the repository
git clone https://github.com/Camgur/uMLIPs-Li-SSE.git
cd uMLIPs-Li-SSE

# Create and activate the venv
python -m venv "venv"
source path-to-venv/bin/activate
pip install -r requirements.txt
```

### Configuration

All scripts use centralized configuration from `scripts/utils/config.py`. Key configuration variables include:

- **Paths**: `REPO_ROOT`, `RESULTS_DIR`, `NEB_RESULTS_DIR`, `MD_RESULTS_DIR`, `CONFIGS_DIR`, `DOS_OUTPUT_DIR`
- **Parameters**: `ELEMENT` (mobile ion species), `TIMESTEP_PS` (MD timestep), `DEFAULT_TEMP` (default temperature)
- **Models**: List of all supported uMLIP model aliases
- **Visualization**: Color and line style mappings for consistent figure generation

Configuration can be overridden via CLI arguments in individual scripts (e.g., `--material`, `--model`, `--temperature`).

---

## 6. Usage

### Stage 1 — Pre-Screening

```bash
# 1a. Relax structures with a uMLIP
python scripts/relaxation/relax.py structures/LSnPS/LSnPS.cif mace-0b3

# 1b. Run DOS pre-screening (compares DOS across all models)
python scripts/dos/compare_dos.py --material LSnPS

# 1c. Run NEB pre-screening
python scripts/neb/run_neb.py structures/LSnPS/LSnPS.cif idx1 idx2 mace-0b3

# 1d. Analyze NEB barriers
python scripts/neb/plot_neb.py --material LSnPS --models mace-0b3 mace-mpa chgnet-2024
```

### Stage 2 — MD Simulations (top candidates only)

```bash
# Run MD at a given temperature
python scripts/md/run_md.py structures/LSnPS/LSnPS.cif 800 mace-0b3

# Compute MSD from trajectories (automatically discovers all md_*.traj files)
python scripts/md/compute_msd.py --material LSnPS

# Fit Arrhenius equation (automatically discovers all MSD .npz files)
python scripts/md/arrhenius_fit.py --material LSnPS
```

### Analysis Notebooks

Open the notebooks in `notebooks/` for interactive analysis and figure generation.
