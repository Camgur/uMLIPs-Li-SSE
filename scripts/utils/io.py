"""utils/io.py — I/O helpers for the uMLIPs-pipeline."""

from __future__ import annotations

from pathlib import Path

from ase import Atoms
from ase.io import read, write


def load_structure(path: str | Path) -> Atoms:
    """Load a crystal structure from *path* (CIF, POSCAR, extxyz, …).

    Parameters
    ----------
    path:
        Path to the structure file.  Any format supported by ASE is accepted.

    Returns
    -------
    ase.Atoms
    """
    return read(str(path))


def save_structure(atoms: Atoms, path: str | Path, fmt: str | None = None) -> None:
    """Write *atoms* to *path*.

    Parameters
    ----------
    atoms:
        The structure to save.
    path:
        Destination file path.
    fmt:
        ASE format string (inferred from extension if *None*).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write(str(path), atoms, format=fmt)


def ensure_dir(path: str | Path) -> Path:
    """Create *path* (and parents) if it does not exist; return as Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
