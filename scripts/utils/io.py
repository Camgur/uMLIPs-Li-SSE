"""utils/io.py — I/O helpers for the uMLIPs-pipeline."""

from pathlib import Path

from ase.io import read, write


def load_structure(path):
    """Load a crystal structure from path (CIF, POSCAR, extxyz, …).
    
    Returns ase.Atoms object.
    """
    return read(str(path))


def save_structure(atoms, path, fmt=None):
    """Write atoms to path.
    
    Parameters:
        atoms: structure to save
        path: destination file path
        fmt: ASE format string (inferred from extension if None)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write(str(path), atoms, format=fmt)


def ensure_dir(path):
    """Create path (and parents) if it does not exist; return as Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
