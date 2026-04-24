# uMLIPs-pipeline

A computational pipeline for benchmarking universal Machine Learning Interatomic Potentials (uMLIPs) for predicting activation energies of lithium-ion diffusion in solid-state electrolytes.

---

## Table of Contents

1. [Background](#1-background)
2. [Selected uMLIPs and Materials](#2-selected-umlips-and-materials)
   - 2.1 [Universal Machine Learning Interatomic Potentials](#21-universal-machine-learning-interatomic-potentials)
   - 2.2 [Solid-State Electrolyte Materials](#22-solid-state-electrolyte-materials)
3. [Methodology and Pipeline](#3-methodology-and-pipeline)
   - 3.1 [Pre-Screening: DOS and NEB Calculations](#31-pre-screening-dos-and-neb-calculations)
     - 3.1.1 [Density of States (DOS)](#311-density-of-states-dos)
     - 3.1.2 [Nudged Elastic Band (NEB)](#312-nudged-elastic-band-neb)
   - 3.2 [Lithium Diffusion MD Simulations](#32-lithium-diffusion-md-simulations)
     - 3.2.1 [Arrhenius Fitting](#321-arrhenius-fitting)
     - 3.2.2 [Comparison with Experimental ssNMR Data](#322-comparison-with-experimental-ssnmr-data)
4. [Repository Structure](#4-repository-structure)
5. [Installation and Dependencies](#5-installation-and-dependencies)
6. [Usage](#6-usage)
7. [Results Summary](#7-results-summary)
8. [References](#8-references)

---

## 1. Background

Solid-state electrolytes (SSEs) are a key enabling technology for next-generation solid-state batteries, offering improved safety and energy density compared to conventional liquid electrolytes. A critical performance metric for SSEs is the activation energy (E<sub>a</sub>) for lithium-ion diffusion, which governs ionic conductivity and, ultimately, battery performance.

Traditionally, E<sub>a</sub> is extracted by fitting the Arrhenius equation to lithium diffusivity data obtained from molecular dynamics (MD) simulations at multiple temperatures. When performed with *ab initio* MD (AIMD), this approach is computationally prohibitive: simulations long enough to achieve statistical convergence of diffusion coefficients demand significant high-performance computing resources.

Universal Machine Learning Interatomic Potentials (uMLIPs) — pre-trained on large, chemically diverse datasets — offer orders-of-magnitude computational savings over DFT-based methods, making it feasible to run the long MD trajectories required for Arrhenius analysis. However, the accuracy of existing uMLIPs for lithium diffusion in SSEs is not yet well-characterised, and the cost of exhaustive MD benchmarking across many models and materials remains non-trivial.

This work addresses that gap by introducing a two-stage pipeline:

1. **Pre-screening** (low-cost): Density of States (DOS) and Nudged Elastic Band (NEB) calculations quickly identify uMLIPs that accurately describe the electronic and energetic landscape of each material.
2. **Full evaluation** (high-cost): Only the most promising uMLIPs, identified in Stage 1, are used for long lithium-diffusion MD simulations and subsequent Arrhenius fitting, with final validation against experimental solid-state NMR (ssNMR) diffusion data.

---

## 2. Selected uMLIPs and Materials

### 2.1 Universal Machine Learning Interatomic Potentials

The following uMLIPs are evaluated in this study. All models are used in inference mode (no additional fine-tuning) to assess their out-of-the-box transferability to lithium SSE chemistries.

| Model | Architecture | Training Dataset | Reference |
|-------|-------------|-----------------|-----------|
| MACE-MP-0 | MACE (equivariant GNN) | MPtrj | [Batatia et al., 2023] |
| CHGNet | Graph neural network | MPtrj | [Deng et al., 2023] |
| M3GNet | Graph neural network | Materials Project | [Chen & Ong, 2022] |
| SevenNet | Equivariant GNN | MPtrj | [Park et al., 2024] |
| ORB | Transformer-based | MPtrj + others | [Neumann et al., 2024] |

> **Note:** This table will be updated as additional models are incorporated into the benchmark.

### 2.2 Solid-State Electrolyte Materials

The benchmark focuses on lithium-based SSE materials spanning several structural and chemical families:

| Material | Structure Type | Li Conductivity Regime |
|----------|---------------|----------------------|
| Li<sub>7</sub>La<sub>3</sub>Zr<sub>2</sub>O<sub>12</sub> (LLZO) | Garnet | High |
| Li<sub>6</sub>PS<sub>5</sub>Cl (argyrodite) | Argyrodite | High |
| Li<sub>3</sub>PS<sub>4</sub> (β-LPS) | Thio-LISICON | Moderate |
| Li<sub>10</sub>GeP<sub>2</sub>S<sub>12</sub> (LGPS) | LGPS-type | Very high |
| LiPON | Amorphous | Low–Moderate |

> **Note:** Materials list subject to revision as the study progresses.

---

## 3. Methodology and Pipeline

### 3.1 Pre-Screening: DOS and NEB Calculations

Pre-screening is performed at low computational cost to identify uMLIPs whose predicted energy surfaces are consistent with DFT reference calculations. Two complementary descriptors are used:

#### 3.1.1 Density of States (DOS)

The electronic DOS provides a fingerprint of a material's electronic structure. Although uMLIPs do not directly produce electronic wavefunctions, the forces and energies they predict reflect the underlying potential energy surface. Here, DOS (computed via single-point DFT on uMLIP-relaxed structures) is used to assess whether structural relaxations by each uMLIP preserve key features of the DFT-reference geometry.

- Structures are relaxed with each uMLIP using the ASE `BFGS` optimiser.
- Single-point DFT (VASP/PBE) calculations are then performed on the relaxed geometries.
- Resulting DOS curves are compared to fully DFT-relaxed reference structures using a cosine-similarity metric.

#### 3.1.2 Nudged Elastic Band (NEB)

NEB calculations provide direct estimates of migration barriers for Li<sup>+</sup> hopping between adjacent sites — the microscopic quantity most directly linked to E<sub>a</sub>. This makes NEB a powerful and cost-efficient pre-screening criterion.

- Initial and final images are taken from DFT-relaxed structures.
- Intermediate images are generated by linear interpolation.
- Each uMLIP is used as the force engine within ASE's NEB implementation (`ase.neb.NEB`).
- Barriers (E<sub>NEB</sub>) are compared against DFT-NEB reference barriers from the literature or computed in-house.

Models whose NEB barriers deviate by more than a defined threshold (e.g., ±0.1 eV) from DFT references are excluded from the full MD evaluation.

### 3.2 Lithium Diffusion MD Simulations

uMLIPs that pass the pre-screening stage proceed to long MD simulations designed to extract diffusion coefficients at multiple temperatures.

#### 3.2.1 Arrhenius Fitting

The mean-square displacement (MSD) of Li<sup>+</sup> ions is computed from MD trajectories at several temperatures (typically 600–1200 K for accelerated sampling). The self-diffusion coefficient *D* is obtained from the Einstein relation:

$$D = \lim_{t \to \infty} \frac{\langle |r(t) - r(0)|^2 \rangle}{6t}$$

The temperature dependence of *D* is then fitted to the Arrhenius equation:

$$D(T) = D_0 \exp\!\left(-\frac{E_a}{k_B T}\right)$$

to extract E<sub>a</sub> and the pre-exponential factor *D*<sub>0</sub>. Room-temperature ionic conductivity is subsequently estimated via the Nernst–Einstein equation.

#### 3.2.2 Comparison with Experimental ssNMR Data

Solid-state NMR (ssNMR) provides experimental activation energies and diffusion coefficients that are independent of any simulation model. Calculated E<sub>a</sub> values from the best uMLIPs are compared against available ssNMR literature data for each material to provide a direct, quantitative validation.

---

## 4. Repository Structure

```
uMLIPs-pipeline/
│
├── README.md                    # This file
│
├── structures/                  # Input crystal structures (CIF / POSCAR)
│   ├── LLZO/
│   ├── argyrodite/
│   ├── beta-LPS/
│   ├── LGPS/
│   └── LiPON/
│
├── scripts/
│   ├── relaxation/              # Structure relaxation with each uMLIP
│   │   └── relax.py
│   ├── dos/                     # DOS pre-screening
│   │   ├── run_dos.py
│   │   └── compare_dos.py
│   ├── neb/                     # NEB pre-screening
│   │   ├── build_neb_images.py
│   │   ├── run_neb.py
│   │   └── analyse_neb.py
│   ├── md/                      # MD simulations and MSD analysis
│   │   ├── run_md.py
│   │   ├── compute_msd.py
│   │   └── arrhenius_fit.py
│   └── utils/                   # Shared utilities
│       ├── io.py
│       └── models.py            # uMLIP loader/wrapper
│
├── configs/                     # Calculation parameters (YAML)
│   ├── models.yaml              # List of uMLIPs and loading instructions
│   ├── neb_params.yaml
│   └── md_params.yaml
│
├── results/
│   ├── dos/                     # DOS comparison metrics
│   ├── neb/                     # NEB barrier results
│   ├── md/                      # MD trajectories and MSD data
│   └── arrhenius/               # Fitted Ea values and conductivity estimates
│
├── notebooks/                   # Analysis and plotting notebooks
│   ├── 01_dos_screening.ipynb
│   ├── 02_neb_screening.ipynb
│   ├── 03_arrhenius_analysis.ipynb
│   └── 04_ssnmr_comparison.ipynb
│
└── environment.yml              # Conda environment specification
```

---

## 5. Installation and Dependencies

### Prerequisites

- Python ≥ 3.10
- [ASE](https://wiki.fysik.dtu.dk/ase/) ≥ 3.23
- [PyMatGen](https://pymatgen.org/) ≥ 2024.1
- [NumPy](https://numpy.org/), [SciPy](https://scipy.org/), [Matplotlib](https://matplotlib.org/)
- Individual uMLIP packages (see `environment.yml`)

### Environment Setup

```bash
# Clone the repository
git clone https://github.com/Camgur/uMLIPs-pipeline.git
cd uMLIPs-pipeline

# Create and activate the conda environment
conda env create -f environment.yml
conda activate umlips-pipeline
```

---

## 6. Usage

### Stage 1 — Pre-Screening

```bash
# 1a. Relax structures with all uMLIPs
python scripts/relaxation/relax.py --config configs/models.yaml --material LLZO

# 1b. Run DOS pre-screening
python scripts/dos/run_dos.py --material LLZO
python scripts/dos/compare_dos.py --material LLZO

# 1c. Run NEB pre-screening
python scripts/neb/build_neb_images.py --material LLZO
python scripts/neb/run_neb.py --material LLZO --config configs/neb_params.yaml
python scripts/neb/analyse_neb.py --material LLZO
```

### Stage 2 — MD Simulations (top candidates only)

```bash
# Run MD at multiple temperatures
python scripts/md/run_md.py --material LLZO --model MACE-MP-0 --config configs/md_params.yaml

# Compute MSD and fit Arrhenius equation
python scripts/md/compute_msd.py --material LLZO --model MACE-MP-0
python scripts/md/arrhenius_fit.py --material LLZO --model MACE-MP-0
```

### Analysis Notebooks

Open the notebooks in `notebooks/` for interactive analysis and figure generation.

---

## 7. Results Summary

> Results will be populated as the benchmark progresses.

| Material | Best uMLIP | E<sub>a</sub> (uMLIP) [eV] | E<sub>a</sub> (ssNMR) [eV] | Δ [eV] |
|----------|-----------|--------------------------|--------------------------|--------|
| LLZO     | TBD       | TBD                      | ~0.30–0.40               | TBD    |
| Argyrodite | TBD     | TBD                      | ~0.20–0.30               | TBD    |
| β-LPS    | TBD       | TBD                      | ~0.35–0.45               | TBD    |
| LGPS     | TBD       | TBD                      | ~0.20–0.25               | TBD    |
| LiPON    | TBD       | TBD                      | ~0.50–0.60               | TBD    |

---

## 8. References

- Batatia, I. et al. (2023). MACE: Higher order equivariant message passing neural networks for fast and accurate force fields. *NeurIPS*.
- Chen, C. & Ong, S. P. (2022). A universal graph deep learning interatomic potential for the periodic table. *Nature Computational Science*, 2, 718–728.
- Deng, B. et al. (2023). CHGNet as a pretrained universal neural network potential for charge-informed atomistic modelling. *Nature Machine Intelligence*, 5, 1031–1041.
- Neumann, M. et al. (2024). Orb: A fast, scalable neural network potential. *arXiv:2410.22570*.
- Park, Y.-J. et al. (2024). SevenNet: A interatomic potential benchmark and pretrained model for materials. *npj Computational Materials*.
- Experimental ssNMR references to be added per material.
