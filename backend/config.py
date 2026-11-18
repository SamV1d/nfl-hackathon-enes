"""Configuración: resolución de la carpeta de datos."""
from __future__ import annotations

import os
from pathlib import Path

DATA_DIR_ENV = "DATA_DIR"
DEFAULT_DATA_DIR = "Datos/data"
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def resolve_data_dir(env_value: str | None, project_root: Path) -> Path:
    """Valor no vacío -> project_root / valor; si no, project_root / Datos/data."""
    if env_value is not None and env_value.strip():
        return project_root / env_value.strip()
    return project_root / DEFAULT_DATA_DIR


def data_dir_from_env() -> Path:
    return resolve_data_dir(os.environ.get(DATA_DIR_ENV), PROJECT_ROOT)
