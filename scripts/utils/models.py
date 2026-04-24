"""utils/models.py — uMLIP loader utilities.

Loads and returns an ASE Calculator for a given model name using the
parameters defined in configs/models.yaml.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import yaml


_CONFIG_PATH = Path(__file__).parents[2] / "configs" / "models.yaml"


def load_config(config_path: Path = _CONFIG_PATH) -> dict[str, Any]:
    """Load the models YAML configuration."""
    with open(config_path) as fh:
        return yaml.safe_load(fh)


def get_calculator(model_name: str, config_path: Path = _CONFIG_PATH):
    """Return an ASE Calculator instance for *model_name*.

    Parameters
    ----------
    model_name:
        Key matching an entry under ``models:`` in *config_path*.
    config_path:
        Path to the models YAML configuration file.

    Returns
    -------
    ase.calculators.calculator.Calculator
        Configured calculator ready for use.

    Raises
    ------
    KeyError
        If *model_name* is not found in the configuration.
    ImportError
        If the required uMLIP package is not installed.
    """
    config = load_config(config_path)
    models = config.get("models", {})

    if model_name not in models:
        available = ", ".join(models.keys())
        raise KeyError(
            f"Model '{model_name}' not found in config. "
            f"Available models: {available}"
        )

    entry = models[model_name]
    package = entry["package"]
    calc_class_name = entry["calculator"]
    kwargs = entry.get("kwargs", {})

    try:
        module = importlib.import_module(package)
    except ImportError as exc:
        raise ImportError(
            f"Package '{package}' required for model '{model_name}' is not installed. "
            f"Please install it and re-run."
        ) from exc

    calc_class = getattr(module, calc_class_name)
    return calc_class(**kwargs)


def list_models(config_path: Path = _CONFIG_PATH) -> list[str]:
    """Return the names of all models defined in the configuration."""
    config = load_config(config_path)
    return list(config.get("models", {}).keys())
