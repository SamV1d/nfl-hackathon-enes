"""Propiedades 1 y 2. Generadores aleatorios con semilla (hypothesis no está en el stack)."""
from pathlib import Path

import numpy as np
import pytest

from backend.config import DEFAULT_DATA_DIR, resolve_data_dir
from backend.loader import REQUIRED_FILES, StartupError, check_files, read_all

ITERATIONS = 100


def test_property_1_data_dir_resolution():
    """Feature: coach-pass-rush-assistant, Property 1: Data directory resolution"""
    rng = np.random.default_rng(1)
    alphabet = list("abcXYZ019_-.")
    for _ in range(ITERATIONS):
        root = Path("/proj") / "".join(rng.choice(alphabet, size=rng.integers(1, 8)))
        kind = rng.integers(0, 3)
        if kind == 0:
            env = None
        elif kind == 1:
            env = ""
        else:
            parts = ["".join(rng.choice(alphabet, size=rng.integers(1, 6))) for _ in range(rng.integers(1, 4))]
            env = "/".join(parts)
        expected = root / env if env else root / DEFAULT_DATA_DIR
        assert resolve_data_dir(env, root) == expected


def test_property_2_missing_files_all_reported(tmp_path):
    """Feature: coach-pass-rush-assistant, Property 2: Missing files are all reported"""
    rng = np.random.default_rng(2)
    for i in range(ITERATIONS):
        d = tmp_path / f"case{i}"
        d.mkdir()
        present = [f for f in REQUIRED_FILES if rng.random() < 0.5]
        for f in present:
            (d / f).write_text("x\n")
        absent = [f for f in REQUIRED_FILES if f not in present]
        assert check_files(d) == absent
        if absent:
            with pytest.raises(StartupError) as ei:
                read_all(d)
            assert ei.value.missing == absent
            assert all(name in str(ei.value) for name in absent)


@pytest.mark.parametrize("content,category", [("", "empty"), ("a,b\n1,2\n", "malformed")])
def test_unreadable_categories(tmp_path, content, category):
    for f in REQUIRED_FILES:
        (tmp_path / f).write_text("gameId,playId,nflId,pff_role\n1,1,1,x\n")
    (tmp_path / "plays.csv").write_text(content)
    with pytest.raises(StartupError) as ei:
        read_all(tmp_path)
    assert (ei.value.file, ei.value.category) == ("plays.csv", category)
