"""Ejecuta la auditoría: `python -m backend.ml.audit` imprime un resumen y el JSON completo."""
from __future__ import annotations

import json
import sys
import time

from ..config import data_dir_from_env
from ..dataset import build_play_dataset
from ..loader import read_all
from .validation import run_audit


def load_and_audit() -> dict:
    files = read_all(data_dir_from_env())
    dataset = build_play_dataset(files["plays.csv"], files["pffScoutingData.csv"])
    return run_audit(dataset, files["plays.csv"], files["games.csv"])


def main() -> None:
    t0 = time.monotonic()
    result = load_and_audit()
    if "--json" in sys.argv:
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return
    for key in ("pressure_with_defensive_choice", "pressure_pre_snap_only", "sack"):
        r = result[key]
        print(f"\n== {key}: campeón {r['champion']} | estado {r['status']}")
        print("  test:", {k: round(v, 4) for k, v in r["test"].items() if isinstance(v, float)})
        print("  base:", {k: round(v, 4) for k, v in r["test_baseline"].items() if isinstance(v, float)})
        print("  skill Brier:", round(r["brier_skill_vs_baseline"], 4), r["confidence_intervals"])
        print("  motivos:", r["reasons"])
    for key in ("yards_net", "yards_on_attempts"):
        r = result[key]
        print(f"\n== {key}: campeón {r['champion']} | estado {r['status']}")
        print("  test:", {k: round(v, 4) for k, v in r["test"].items() if isinstance(v, float)})
        print("  base:", {k: round(v, 4) for k, v in r["test_baseline"].items() if isinstance(v, float)})
        print("  skill MSE:", round(r["mse_skill_vs_baseline"], 4), r["mse_skill_ci95"])
        print("  motivos:", r["reasons"])
    print(f"\n{time.monotonic() - t0:.1f} s")


if __name__ == "__main__":
    main()
