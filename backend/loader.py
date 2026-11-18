"""Carga de los CSV fuente. Solo lectura; nunca abre tracking."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED_FILES = ("plays.csv", "pffScoutingData.csv", "players.csv", "games.csv")

# Columnas mínimas para considerar que el encabezado existe (detecta CSV sin header).
_KEY_COLUMNS = {
    "plays.csv": ("gameId", "playId"),
    "pffScoutingData.csv": ("gameId", "playId", "nflId", "pff_role"),
    "players.csv": ("nflId",),
    "games.csv": ("gameId",),
}

UNREADABLE, EMPTY, MALFORMED = "unreadable", "empty", "malformed"


class StartupError(Exception):
    """Fallo de arranque. `missing` lista archivos ausentes; `file`/`category` un archivo ilegible."""

    def __init__(
        self,
        missing: list[str] | None = None,
        file: str | None = None,
        category: str | None = None,
    ):
        self.missing = list(missing or [])
        self.file = file
        self.category = category
        if self.missing:
            msg = "Archivos CSV faltantes: " + ", ".join(self.missing)
        else:
            msg = f"No se pudo cargar {file}: {category}"
        super().__init__(msg)


def check_files(data_dir: Path) -> list[str]:
    """Nombres de los archivos requeridos que no existen en data_dir."""
    data_dir = Path(data_dir)
    return [name for name in REQUIRED_FILES if not (data_dir / name).is_file()]


def _read_one(path: Path) -> pd.DataFrame:
    name = path.name
    try:
        df = pd.read_csv(path)  # abre en modo lectura
    except pd.errors.EmptyDataError:
        raise StartupError(file=name, category=EMPTY) from None
    except (pd.errors.ParserError, ValueError) as exc:
        # UnicodeDecodeError es subclase de ValueError: es un archivo ilegible, no mal formado.
        if isinstance(exc, UnicodeDecodeError):
            raise StartupError(file=name, category=UNREADABLE) from None
        raise StartupError(file=name, category=MALFORMED) from None
    except OSError:
        raise StartupError(file=name, category=UNREADABLE) from None

    if df.empty:
        raise StartupError(file=name, category=EMPTY)
    if not set(_KEY_COLUMNS[name]).issubset(df.columns):
        raise StartupError(file=name, category=MALFORMED)
    return df


def read_all(data_dir: Path) -> dict[str, pd.DataFrame]:
    """Lee completos los cuatro CSV requeridos. Clave = nombre del archivo."""
    data_dir = Path(data_dir)
    missing = check_files(data_dir)
    if missing:
        raise StartupError(missing=missing)
    return {name: _read_one(data_dir / name) for name in REQUIRED_FILES}
