"""
Gráfica mejorada: relación entre la integral de normalización (qué tan
colapsada está psi) y el error relativo de energía, para todos los casos
que cumplen el criterio de colapso (integral < 0.2 y L2 > 0.5).

Un punto abajo-a-la-izquierda = colapso severo con energía "engañosamente
correcta" (falso positivo). Un punto arriba-a-la-izquierda = colapso severo
con energía que también refleja el error (honesto).

Uso:
    python scripts/plot_collapse_honesty_scatter.py
"""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    df = pd.read_csv("outputs/summary/all_runs_noise.csv")
    collapsed = df[df["collapse"] == True].copy()

    eps = 1e-13
    collapsed["integral_plot"] = collapsed["integral"].clip(lower=eps)
    collapsed["E_rel_plot"] = collapsed["E_rel"].clip(lower=eps)

    colors = {"sin_rayleigh": "#E24B4A", "con_rayleigh": "#378ADD"}
    labels = {"sin_rayleigh": "Sin Rayleigh (parámetro libre)", "con_rayleigh": "Con Rayleigh"}

    fig, ax = plt.subplots(figsize=(9, 7))

    for form in ["sin_rayleigh", "con_rayleigh"]:
        sub = collapsed[collapsed["formulation"] == form]
        ax.scatter(
            sub["integral_plot"], sub["E_rel_plot"],
            s=18, alpha=0.35, color=colors[form],
            label=f"{labels[form]}  (n={len(sub)})",
            edgecolors="none",
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Integral de normalización $\int\psi^2\,dx$  (log)  →  más colapsada hacia la izquierda")
    ax.set_ylabel(r"Error relativo de energía $E_{rel}$  (log)  →  peor arriba")

    # Línea de referencia horizontal: umbral de "energía parece correcta"
    ax.axhline(1e-3, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax.text(ax.get_xlim()[1] if False else eps*3, 1.4e-3,
            "umbral: energía 'parece correcta' por debajo de esta línea",
            fontsize=9, color="gray", va="bottom")

    ax.set_title(
        "¿La energía reportada refleja el colapso de ψ?\n"
        "Casos colapsados (integral < 0.2 y L2 > 0.5), arquitectura A1, 30 seeds",
        fontsize=12,
    )
    ax.legend(loc="upper left", fontsize=10, framealpha=0.9)
    ax.grid(alpha=0.2, which="both")

    plt.tight_layout()

    out_dir = "outputs/summary/figs"
    os.makedirs(out_dir, exist_ok=True)
    outpath = os.path.join(out_dir, "collapse_honesty_scatter.png")
    plt.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close()

    print("Listo. Figura guardada en:", outpath)


if __name__ == "__main__":
    main()