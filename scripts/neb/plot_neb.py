import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from ase.io import read
from ase.neb import NEBTools

import sys
sys.path.insert(0, str(Path(__file__).parents[1]))
from utils.config import MODELS, COLORS, LINE_STYLES, NEB_RESULTS_DIR


def extract_barrier(traj_path):
    """Extract NEB barrier height from converged trajectory."""
    images = read(str(traj_path), index=":")
    neb_tools = NEBTools(images)
    return float(neb_tools.get_barrier()[0])


def main():
    parser = argparse.ArgumentParser(description="Plot NEB barriers")
    parser.add_argument("--material", default="LSnPS", help="Material name (default: LSnPS)")
    parser.add_argument("--models", nargs='+', default=None, help="Specific models to plot (default: all)")
    args = parser.parse_args()
    
    material = args.material
    models_to_plot = args.models if args.models else MODELS
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    for model in models_to_plot:
        traj_path = NEB_RESULTS_DIR / material / model / "neb.traj"
        if not traj_path.exists():
            print(f"[WARNING] No NEB trajectory found for {material} / {model}")
            continue
        
        try:
            images = read(str(traj_path), index=":")
            neb_tools = NEBTools(images)
            
            # Get energies and reaction coordinates
            energies = neb_tools.get_energies()
            s = neb_tools.get_distances()
            
            # Normalize to first image
            energies = energies - energies[0]
            
            barrier = extract_barrier(traj_path)
            label = f"{model}: {barrier:.3f} eV"
            
            ax.plot(s, energies, 
                    color=COLORS.get(model, '#000000'),
                    linestyle=LINE_STYLES.get(model, 'solid'),
                    linewidth=2,
                    marker='o',
                    markersize=4,
                    label=label)
            
            print(f"Plotted {material} / {model}: barrier = {barrier:.3f} eV")
            
        except Exception as exc:
            print(f"[ERROR] Could not process {material} / {model}: {exc}")

    ax.set_xlabel("Reaction Coordinate (Å)", fontsize=14)
    ax.set_ylabel("Energy (eV)", fontsize=14)
    ax.set_title(f"NEB Barriers: {material}", fontsize=16)
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    # Save
    out_path = NEB_RESULTS_DIR / f"neb_barriers_{material}.png"
    fig.savefig(out_path, dpi=300)
    print(f"Saved figure to {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
