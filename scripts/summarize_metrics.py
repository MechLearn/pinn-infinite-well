# scripts/summarize_metrics.py
import os
import json
import glob
from collections import defaultdict

import numpy as np
import pandas as pd


def is_collapse(m: dict, integral_thr=0.2, l2_thr=0.5) -> bool:
    """Criterio operativo de colapso (ajustable)."""
    integral = float(m.get("integral", np.nan))
    l2 = float(m.get("L2", np.nan))
    return (integral < integral_thr) and (l2 > l2_thr)


def collect_formulation(formulation: str, base_pattern: str):
    """Busca metrics.json bajo outputs/InfiniteWell/{base_pattern}/{formulation}/..."""
    pattern = os.path.join(base_pattern, formulation, "*_seed*", "mode_*", "metrics.json")
    files = sorted(glob.glob(pattern))
    rows = []
    for fp in files:
        with open(fp, "r") as f:
            m = json.load(f)

        exp = m.get("experiment_name", "exp")
        seed = int(m.get("seed", -1))
        n = int(m.get("n", -1))

        rows.append({
            "formulation": formulation,
            "experiment": exp,
            "seed": seed,
            "n": n,
            "E_rel": float(m.get("E_rel", np.nan)),
            "L2": float(m.get("L2", np.nan)),
            "integral": float(m.get("integral", np.nan)),
            "noise_sigma": float(m.get("noise_sigma", 0.0)),
            "collapse": is_collapse(m),
            "path": fp,
        })
    return rows


def main():
    # Ajusta esta ruta base según qué estés resumiendo:
    # - baseline (sin ruido, OE1/Rayleigh puro):   "outputs/InfiniteWell/runs"
    # - barrido de ruido (OE2):                     "outputs/InfiniteWell/noise/runs"
    base_pattern = "outputs/InfiniteWell/noise/runs"

    rows = []
    for formulation in ["sin_rayleigh", "con_rayleigh"]:
        found = collect_formulation(formulation, base_pattern)
        print(f"{formulation}: {len(found)} archivos metrics.json encontrados")
        rows.extend(found)

    if not rows:
        print(f"No encontré ningún metrics.json bajo {base_pattern}/{{formulation}}/")
        return

    df = pd.DataFrame(rows)
    df = df.sort_values(["formulation", "experiment", "n", "seed"]).reset_index(drop=True)

    summary = (
        df.groupby(["formulation", "experiment", "n"])
          .agg(
              runs=("seed", "count"),
              collapses=("collapse", lambda x: int(np.sum(x))),
              collapse_rate=("collapse", "mean"),
              E_rel_mean=("E_rel", "mean"),
              E_rel_std=("E_rel", "std"),
              L2_mean=("L2", "mean"),
              L2_std=("L2", "std"),
              integral_mean=("integral", "mean"),
              integral_std=("integral", "std"),
          )
          .reset_index()
          .sort_values(["formulation", "experiment", "n"])
    )

    os.makedirs("outputs/summary", exist_ok=True)
    df.to_csv("outputs/summary/all_runs_noise.csv", index=False)
    summary.to_csv("outputs/summary/summary_by_n_noise.csv", index=False)

    print("\n=== SUMMARY (por formulation, experiment, n) ===")
    print(summary.to_string(index=False))

    print("\nGuardado:")
    print(" - outputs/summary/all_runs_noise.csv")
    print(" - outputs/summary/summary_by_n_noise.csv")


if __name__ == "__main__":
    main()