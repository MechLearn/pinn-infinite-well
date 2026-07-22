"""
OE2 — Barrido de ruido en puntos de colocación.

Corre, para cada arquitectura (A1/A2), cada nivel de ruido sigma y cada
semilla, el barrido de modos n_min..n_max, reutilizando run_one_mode_learnE
con noise_sigma > 0. Es resumible: si metrics.json ya existe para una
combinación, la salta.

Ejemplos:
    python scripts/run_sweep_noise.py --arch A1 --sigmas 0 0.01 0.05 0.1 \
        --n_min 1 --n_max 20 --seeds 0 1 2 3 4

    python scripts/run_sweep_noise.py --arch A1 A2 --sigmas 0.05 \
        --n_min 14 --n_max 16 --seeds 0
"""

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pinn1d.config import load_config
from pinn1d.train import run_one_mode_learnE

ARCH_HIDDEN = {"A1": 128, "A2": 256}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=str, default="configs/noise.yaml")
    p.add_argument("--arch", type=str, nargs="+", default=["A1"], choices=["A1", "A2"])
    p.add_argument("--sigmas", type=float, nargs="+", default=[0.0, 0.01, 0.05, 0.1])
    p.add_argument("--n_min", type=int, default=1)
    p.add_argument("--n_max", type=int, default=20)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--resample_every_epoch", action="store_true",
                    help="Si se pasa, resamplea el ruido cada época en vez de fijarlo una vez por corrida.")
    p.add_argument("--force", action="store_true", help="Reentrena aunque metrics.json ya exista.")
    return p.parse_args()


def sigma_tag(sigma: float) -> str:
    return f"s{sigma:g}".replace(".", "p")


def metrics_path_for(cfg: dict, exp_name: str, seed: int, n: int) -> str:
    return os.path.join(cfg["save_dir"], f"{exp_name}_seed{seed}", f"mode_{n}", "metrics.json")


def main():
    args = parse_args()
    base_cfg = load_config(args.config)

    total = len(args.arch) * len(args.sigmas) * len(args.seeds) * (args.n_max - args.n_min + 1)
    done = skipped = 0
    t0 = time.time()

    print(f"=== OE2: barrido de ruido en colocación ===")
    print(f"arch={args.arch} sigmas={args.sigmas} n=[{args.n_min},{args.n_max}] seeds={args.seeds}")
    print(f"Total de corridas planeadas: {total}\n")

    for arch in args.arch:
        for sigma in args.sigmas:
            exp_name = f"{arch}_{sigma_tag(sigma)}"
            for seed in args.seeds:
                for n in range(args.n_min, args.n_max + 1):
                    cfg = dict(base_cfg)
                    cfg["model"] = {"hidden": ARCH_HIDDEN[arch], "use_sine": True}
                    cfg["experiment_name"] = exp_name
                    cfg["seed"] = seed
                    cfg["noise_sigma"] = float(sigma)
                    cfg["noise_resample_every_epoch"] = bool(args.resample_every_epoch)

                    mpath = metrics_path_for(cfg, exp_name, seed, n)
                    if os.path.exists(mpath) and not args.force:
                        print(f"⏭️  {exp_name} seed={seed} n={n} ya existe — salto")
                        skipped += 1
                        continue

                    seed_n = int(seed) * 100 + n
                    torch.manual_seed(seed_n)

                    print(f"\n===== {exp_name} | seed={seed} | n={n} | sigma={sigma} =====")
                    run_one_mode_learnE(n, cfg)
                    done += 1

    elapsed = time.time() - t0
    print(f"\n✅ Completado. Corridas nuevas: {done} | saltadas (ya existían): {skipped} | tiempo: {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
