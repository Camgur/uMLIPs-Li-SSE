import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from openpyxl import Workbook
from scipy.ndimage import gaussian_filter1d
from sumo.io.castep import read_dos
from pymatgen.electronic_structure.core import Spin

# Config
OUTPUT_DIR = "results/figures/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MATERIALS = [
    'LFVO',
]

MODELS = [
    'chgnet-r2-2025',
    'm3gnet-r2-2025',
    'mace-mpa',
    'mace-r2',
]

COLORS = {
    'chgnet-r2-2025': '#648fff',
    'm3gnet-r2-2025': "#B858F4",
    'mace-mpa': "#EA56CA",
    'mace-r2': '#dc267f',
    'dft': '#000000',
}

GAUSSIAN_SIGMA = 0.05
Y_LIMITS = (-5, 10)

# Helpers
def find_bands_file(directory):
    """Find .bands file in directory."""
    files = glob.glob(os.path.join(directory, "*.bands"))
    return files[0] if files else None

def load_and_total(bands_file, gaussian=0.05, emin=-20, emax=20):
    """Load CASTEP DOS with SUMO read_dos.
    
    Returns: energies, total_dos, efermi, dos_obj
    """
    dos_obj, _ = read_dos(bands_file, gaussian=gaussian, emin=emin, emax=emax, total_only=False)
    energies = np.array(dos_obj.energies)
    
    # Sum spins if dict
    if isinstance(dos_obj.densities, dict):
        total = sum(np.array(arr) for arr in dos_obj.densities.values())
    else:
        total = np.array(dos_obj.densities)
        if total.ndim > 1:
            total = total.sum(axis=0)
    
    return energies, np.array(total), float(dos_obj.efermi), dos_obj

def cosine_similarity(a, b):
    """Cosine similarity between vectors a and b."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))

# Load DOS data and compare
similarity_data = []
wb = Workbook()
ws = wb.active
ws.append(["Material", "Model", "Cosine Similarity"])

for material in MATERIALS:
    # Load DFT reference
    dft_dir = f"results/dos/dft/{material}/"
    dft_file = find_bands_file(dft_dir)
    
    if not dft_file:
        print(f"WARNING: No DFT .bands file for {material}")
        continue
    
    try:
        e_dft, dos_dft, efermi_dft, dos_dft_obj = load_and_total(dft_file, gaussian=GAUSSIAN_SIGMA)
        dos_dft_smooth = np.array(dos_dft)  # Already smoothed by read_dos
    except Exception as exc:
        print(f"ERROR: Failed to load DFT for {material}: {exc}")
        continue
    
    # Normalize DFT DOS
    ref_vec = dos_dft_smooth / np.max(dos_dft_smooth) if np.max(dos_dft_smooth) > 0 else dos_dft_smooth
    
    # Create plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Shift energies relative to Fermi
    e_dft_shift = e_dft - efermi_dft
    mask_dft = (e_dft_shift >= Y_LIMITS[0]) & (e_dft_shift <= Y_LIMITS[1])
    
    # Plot ML models first
    plotted_models = []
    for model in MODELS:
        model_dir = f"results/dos/{material}/{model}/"
        model_file = find_bands_file(model_dir)
        
        if not model_file:
            continue
        
        try:
            e_ml, dos_ml, efermi_ml, dos_ml_obj = load_and_total(model_file, gaussian=GAUSSIAN_SIGMA)
            dos_ml_smooth = np.array(dos_ml)
        except Exception as exc:
            print(f"ERROR: Failed to load {model} for {material}: {exc}")
            continue
        
        # Compute similarity
        similarity = cosine_similarity(ref_vec, dos_ml_smooth / np.max(dos_ml_smooth) if np.max(dos_ml_smooth) > 0 else dos_ml_smooth)
        similarity_data.append({'material': material, 'model': model, 'similarity': similarity})
        
        print(f"{material} {model:20} cosine_similarity={similarity:.4f}")
        
        # Plot
        e_ml_shift = e_ml - efermi_ml
        mask_ml = (e_ml_shift >= Y_LIMITS[0]) & (e_ml_shift <= Y_LIMITS[1])
        dos_ml_norm = dos_ml_smooth[mask_ml] / np.max(dos_ml_smooth) if np.max(dos_ml_smooth) > 0 else dos_ml_smooth[mask_ml]
        
        ax.plot(e_ml_shift[mask_ml], dos_ml_norm, color=COLORS.get(model, '#777777'), 
                label=f"{model} ({similarity:.3f})", lw=1.5)
        
        plotted_models.append(model)
        ws.append([material, model, round(similarity, 6)])
    
    # Plot DFT last (on top)
    dos_dft_norm = dos_dft_smooth[mask_dft] / np.max(dos_dft_smooth) if np.max(dos_dft_smooth) > 0 else dos_dft_smooth[mask_dft]
    ax.plot(e_dft_shift[mask_dft], dos_dft_norm, color=COLORS['dft'], label="DFT", lw=2.5)
    
    ws.append([material, 'dft', ""])
    
    # Format plot
    ax.set_xlabel("Energy (eV)", fontsize=14)
    ax.set_ylabel("DOS (normalized)", fontsize=14)
    ax.set_title(f"{material} DOS Comparison", fontsize=16)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axvline(0.0, color='gray', linestyle='--', lw=1.0, alpha=0.5)
    ax.tick_params(labelsize=12)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    
    fig.tight_layout()
    png_path = os.path.join(OUTPUT_DIR, f"dos_{material}.png")
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {png_path}")

# Save Excel summary
xlsx_path = os.path.join(OUTPUT_DIR, "dos_comparison.xlsx")
wb.save(xlsx_path)
print(f"Saved: {xlsx_path}")
