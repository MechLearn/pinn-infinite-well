#!/usr/bin/env python
"""Chequeo de calibración de la posterior VI (p. ej. run n=9, kl_weight=0.001).

Pregunta que responde: la banda ±2σ "casi invisible" ¿es DESHONESTA
(sobreconfianza: el error real se sale de la banda) o solo PEQUEÑA
(cobertura ~95%: la incertidumbre es correcta y el problema es visual)?

No requiere reentrenar: solo muestrear la posterior del checkpoint existente.

ÚNICO PUNTO A ADAPTAR: sample_posterior() — carga tu modelo BBB y haz S
forward passes estocásticos. El resto corre tal cual.

Uso:  python check_calibration.py            # n=9 por defecto
      python check_calibration.py 5          # otro modo
"""
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

L = 1.0  # ancho del pozo — AJUSTA si tu dominio no es [0, 1]
NOMINAL = {1: 68.3, 2: 95.4, 3: 99.7}


def psi_exact(x, n):
    return np.sqrt(2.0 / L) * np.sin(n * np.pi * x / L)


def sample_posterior(x, n, S=300):
    """Devuelve array (S, len(x)) de psi muestreadas de la posterior.

    ADAPTA AQUÍ — típico para Bayes by Backprop:

        import torch
        from pinn1d.vi_rayleigh.model import <TuModeloBBB>
        model = <cargar checkpoint del experimento vi_n9 con klw=0.001>
        model.eval()  # ojo: en BBB el forward debe seguir MUESTREANDO pesos
        xt = torch.tensor(x, dtype=torch.float32).reshape(-1, 1)
        with torch.no_grad():
            return np.stack([model(xt).squeeze().cpu().numpy() for _ in range(S)])
    """
    raise NotImplementedError("Adapta sample_posterior() a tu checkpoint/modelo")


def main(n=9, S=300, n_x=1000):
    x = np.linspace(0.0, L, n_x)
    samples = sample_posterior(x, n, S)              # (S, n_x)
    mu = samples.mean(axis=0)
    sigma = samples.std(axis=0)
    ex = psi_exact(x, n)
    if np.trapz(mu * ex, x) < 0:                     # psi definida salvo signo
        ex = -ex

    err = np.abs(mu - ex)
    rel_l2 = np.sqrt(np.trapz((mu - ex) ** 2, x) / np.trapz(ex ** 2, x))

    print(f"n={n}  S={S}   L2 relativo de la media: {rel_l2:.4f}")
    print(f"sigma: media={sigma.mean():.2e}  max={sigma.max():.2e}")
    print(f"error: medio={err.mean():.2e}  max={err.max():.2e}")
    print(f"ratio mediano error/sigma: {np.median(err / np.maximum(sigma, 1e-30)):.2f}"
          "   (~1 => calibrada; >>1 => sobreconfiada; <<1 => subconfiada)")
    for k in (1, 2, 3):
        cov = 100 * np.mean(err <= k * sigma)
        print(f"cobertura +/-{k} sigma: {cov:5.1f}%   (nominal {NOMINAL[k]}%)")

    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
    axes[0].plot(x, ex, "k--", lw=1, label=r"$\psi$ exacta")
    axes[0].plot(x, mu, "C0", lw=1, label=r"$\mu$ posterior")
    axes[0].fill_between(x, mu - 2 * sigma, mu + 2 * sigma,
                         color="C0", alpha=0.3, label=r"$\pm 2\sigma$")
    axes[0].legend(); axes[0].set_title(f"n={n}: media posterior y banda")
    axes[1].plot(x, sigma, "C1")
    axes[1].set_ylabel(r"$\sigma(x)$")
    axes[1].set_title("Incertidumbre en su propia escala (aunque sea 'invisible' arriba)")
    axes[2].plot(x, err, "C3", lw=1, label=r"$|\mu-\psi|$")
    axes[2].plot(x, 2 * sigma, "C0", lw=1, label=r"$2\sigma$")
    axes[2].set_yscale("log"); axes[2].legend(); axes[2].set_xlabel("x")
    axes[2].set_title("Calibración: el error debería quedar debajo de 2 sigma ~95% del dominio")
    fig.tight_layout()
    out = f"calibration_n{n}.png"
    fig.savefig(out, dpi=150)
    print(f"figura guardada en {out}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 9)