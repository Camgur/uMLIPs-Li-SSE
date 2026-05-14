"""Global configuration for uMLIPs-Li-SSE scripts.

All variables can be overridden via CLI using argparse or environment variables.
"""

from pathlib import Path

# Paths
REPO_ROOT = Path(__file__).parents[1]
CONFIGS_DIR = REPO_ROOT / "configs"
RESULTS_DIR = REPO_ROOT / "results"
DOS_OUTPUT_DIR = RESULTS_DIR / "dos"
NEB_RESULTS_DIR = RESULTS_DIR / "neb"
MD_RESULTS_DIR = RESULTS_DIR / "md"

# Default Parameters
GAUSSIAN_SIGMA = 0.05  # Smoothing parameter for DOS
DEFAULT_STRUCTURE = 'LFVO'  # Default structure for DOS analysis
ELEMENT = "Li"  # Mobile ion species
TIMESTEP_PS = 0.05  # MD timestep in picoseconds
DEFAULT_TEMP = 800  # Default temperature in K

# Models
MODELS = [
    'chgnet-2023', 
    'chgnet-2024',
    'chgnet-PBE-2025',
    'chgnet-r2-2025',
    'chgnet-torch',
    'm3gnet-PBE-2025', 
    'm3gnet-r2-2025', 
    'm3gnet-2021', 
    'mace-0b3', 
    'mace-mpa', 
    'mace-l2', 
    'mace-omat',
    'mace-r2',
    'orb-v2',
    'orb-v3-cMPA',
    'orb-v3-cOMAT',
    'orb-v3-iMPA',
    'orb-v3-iOMAT',
    'dft',
]

# Visualization defaults
COLORS = {
    'chgnet-2023': "#404a66",
    'chgnet-2024': "#022273",
    'chgnet-PBE-2025': "#9dcbfc",
    'chgnet-r2-2025': '#648fff',
    'chgnet-torch': "#0248f9",
    'm3gnet-PBE-2025': "#420A73",
    'm3gnet-r2-2025': "#B858F4",
    'm3gnet-2021': "#8C12F7",
    'mace-0b3': "#890646",
    'mace-l2': "#f280b7",
    'mace-mpa': "#EA56CA",
    'mace-omat': "#8E0180",
    'mace-r2': '#dc267f',
    'orb-v2': '#fe6100',
    'orb-v3-cMPA': "#ff0000",
    'orb-v3-cOMAT': "#ff4800",
    'orb-v3-iMPA': "#c06329",
    'orb-v3-iOMAT': "#f49b00",
    'dft': '#000000',
}

LINE_STYLES = {
    'chgnet-2023': 'dotted',
    'chgnet-2024': 'dotted',
    'chgnet-PBE-2025': 'dotted',
    'chgnet-r2-2025': 'dotted',
    'chgnet-torch': 'dotted',
    'm3gnet-2021': 'dotted',
    'm3gnet-PBE-2025': 'dotted',
    'm3gnet-r2-2025': 'dotted',
    'mace-0b3': 'dashed',
    'mace-mpa': 'dashed',
    'mace-l2': 'dashed',
    'mace-omat': 'dashed',
    'mace-r2': 'dashed',
    'orb-v2': 'dashdot',
    'orb-v3-cMPA': 'dashdot',
    'orb-v3-cOMAT': 'dashdot',
    'orb-v3-iMPA': 'dashdot',
    'orb-v3-iOMAT': 'dashdot',
    'dft': 'solid',
}


def get_config(override_dict=None):
    """Return configuration dictionary, optionally overridden by CLI args or env vars"""
    config = {
        'REPO_ROOT': REPO_ROOT,
        'RESULTS_DIR': RESULTS_DIR,
        'NEB_RESULTS_DIR': NEB_RESULTS_DIR,
        'MD_RESULTS_DIR': MD_RESULTS_DIR,
        'CONFIGS_DIR': CONFIGS_DIR,
        'DOS_OUTPUT_DIR': DOS_OUTPUT_DIR,
        'ELEMENT': ELEMENT,
        'TIMESTEP_PS': TIMESTEP_PS,
        'DEFAULT_TEMP': DEFAULT_TEMP,
        'MODELS': MODELS,
        'COLORS': COLORS,
        'LINE_STYLES': LINE_STYLES,
        'GAUSSIAN_SIGMA': GAUSSIAN_SIGMA,
        'DEFAULT_STRUCTURE': DEFAULT_STRUCTURE,
    }
    
    if override_dict:
        config.update(override_dict)
    
    return config
