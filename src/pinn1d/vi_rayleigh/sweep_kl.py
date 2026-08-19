#!/usr/bin/env python
"""Cola de corridas VI-PINN — umbral de kl_weight (pozo infinito, formulación Rayleigh).

Colócalo en la raíz del repo (donde `import pinn1d` funcione) y corre:

  python sweep_kl.py bisect              # bisección log del umbral, n=9  (empieza en ~0.0022)
  python sweep_kl.py probe               # klw=0.005 con prior_sigma=1.0  (diagnóstico mecanismo)
  python sweep_kl.py seeds 0.002 0.003   # seeds 1 y 2 en los valores del borde del bracket
  python sweep_kl.py grid                # barrido n∈{3,5,7} × klw∈{0.001,0.002,0.005,0.01}

Cada corrida se registra (append) en results_sweep.csv.

SUPUESTO ÚNICO A VERIFICAR: que run_one_mode_vi(n, cfg) devuelva un dict con el
L2 relativo bajo alguna clave que contenga "l2". Si no devuelve nada útil, el
script te pide el L2 por teclado al final de cada corrida y sigue solo.
"""
import csv
import math
import sys
import time
from pathlib import Path

from pinn1d.vi_rayleigh.train import run_one_mode_vi

CSV_PATH = Path("results_sweep.csv")
COLLAPSE_L2 = 0.5        # L2 relativo > 0.5  =>  colapso (tus colapsos dan ~0.97)
BRACKET_TOL = 1.3        # detener bisección cuando hi/lo <= 1.3
N_COL_FIXED = 2048 * 9   # FIJO para todo n: elimina el confound de n_col = 2048*n


def make_cfg(n, klw, seed=0, prior_sigma=0.3):
    tag = f"n{n}_kl{klw:.5f}".replace("0.", "0p")
    if prior_sigma != 0.3:
        tag += f"_prior{prior_sigma}".replace(".", "p")
    # el seed no va en el tag: train.py ya agrega "_seed{seed}" a la carpeta
    return {
        "model": {"hidden": 64, "use_sine": True},
        "n_col": N_COL_FIXED,
        "epochs": 18000,
        "lr": 3e-4,
        "lam": 300.0,
        "kl_weight": klw,
        "prior_sigma": prior_sigma,
        "seed": seed,
        "experiment_name": tag,
    }


def extract_l2(result):
    if isinstance(result, dict):
        for key in ("rel_l2", "l2_rel", "l2", "L2", "l2_error", "error_l2"):
            if key in result:
                return float(result[key])
        for key, val in result.items():
            if "l2" in str(key).lower() and isinstance(val, (int, float)):
                return float(val)
    if isinstance(result, (int, float)):
        return float(result)
    return None


def run_and_log(n, klw, seed=0, prior_sigma=0.3):
    cfg = make_cfg(n, klw, seed, prior_sigma)
    print(f"\n=== n={n}  klw={klw:g}  seed={seed}  prior_sigma={prior_sigma}"
          f"  ({cfg['experiment_name']}) ===")
    t0 = time.time()
    result = run_one_mode_vi(n, cfg)
    minutes = (time.time() - t0) / 60
    l2 = extract_l2(result)
    if l2 is None:
        l2 = float(input("No pude leer el L2 del retorno de run_one_mode_vi. "
                         "Escribe el L2 relativo final: "))
    collapsed = l2 > COLLAPSE_L2
    is_new = not CSV_PATH.exists()
    with CSV_PATH.open("a", newline="") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(["n", "kl_weight", "prior_sigma", "seed", "l2",
                             "collapsed", "minutes", "experiment_name"])
        writer.writerow([n, klw, prior_sigma, seed, f"{l2:.5f}",
                         int(collapsed), f"{minutes:.1f}", cfg["experiment_name"]])
    print(f"--> L2={l2:.4f}  {'COLAPSO' if collapsed else 'converge'}  ({minutes:.1f} min)")
    return l2, collapsed


def bisect(n=9, lo=0.001, hi=0.005):
    """lo ya verificado como convergente, hi como colapso. Bisección geométrica."""
    while hi / lo > BRACKET_TOL:
        mid = round(math.sqrt(lo * hi), 5)
        _, collapsed = run_and_log(n, mid)
        if collapsed:
            hi = mid
        else:
            lo = mid
        print(f"[bracket] umbral en ({lo:.5f}, {hi:.5f})  ratio={hi / lo:.2f}")
    print(f"\nUMBRAL FINAL n={n}: kl_weight* en ({lo:.5f}, {hi:.5f})")
    return lo, hi


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "bisect"
    if mode == "bisect":
        bisect()
    elif mode == "probe":
        run_and_log(9, 0.005, prior_sigma=1.0)
    elif mode == "seeds":
        klws = [float(a) for a in sys.argv[2:]]
        if not klws:
            sys.exit("Uso: python sweep_kl.py seeds 0.002 0.003")
        for klw in klws:
            for seed in (1, 2):
                run_and_log(9, klw, seed=seed)
    elif mode == "grid":
        for n in (3, 5, 7):
            for klw in (0.001, 0.002, 0.005, 0.01):
                run_and_log(n, klw)
    elif mode == "repro9":
        for klw in (0.001, 0.00224, 0.005, 0.01):
            run_and_log(9, klw)
        for klw in (0.00335, 0.00409):
            for seed in (0, 1, 2):
                run_and_log(9, klw, seed=seed)
    else:
        sys.exit(f"Modo desconocido: {mode} (usa bisect | probe | seeds | grid | repro9)")


if __name__ == "__main__":
    main()