"""
Gráfica final para OE2: tasa de colapso vs n, comparando sin_rayleigh vs
con_rayleigh, un panel por nivel de ruido (sigma).

Uso:
    python scripts/plot_collapse_rate.py

Requiere que ya exista outputs/summary/summary_by_n_noise.csv
(generado por scripts/summarize_metrics.py).
"""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def sigma_from_experiment(exp_name: str) -> float:
    """Extrae el valor de sigma del nombre de experimento, ej. 'A1_s0p05' -> 0.05"""
    tag = exp_name.split("_")[1]  # ej. "s0p05"
    tag = tag[1:]  # quita la 's' -> "0p05"
    tag = tag.replace("p", ".")
    return float(tag)


def main():
    in_csv = "outputs/summary/summary_by_n_noise.csv"
    if not os.path.exists(in_csv):
        raise FileNotFoundError(f"No existe: {in_csv}. Corre primero summarize_metrics.py")

    df = pd.read_csv(in_csv)

    # Solo arquitectura A1, que es la que tiene 30 seeds completas en ambas formulaciones
    df = df[df["experiment"].str.startswith("A1")].copy()
    df["sigma"] = df["experiment"].apply(sigma_from_experiment)

    sigmas = sorted(df["sigma"].unique())
    formulations = ["sin_rayleigh", "con_rayleigh"]
    colors = {"sin_rayleigh": "tab:red", "con_rayleigh": "tab:blue"}
    markers = {"sin_rayleigh": "o", "con_rayleigh": "s"}

    fig, axes = plt.subplots(1, len(sigmas), figsize=(5.5 * len(sigmas), 4.5), sharey=True)
    if len(sigmas) == 1:
        axes = [axes]

    for ax, sigma in zip(axes, sigmas):
        for formulation in formulations:
            sub = df[(df["formulation"] == formulation) & (df["sigma"] == sigma)].sort_values("n")
            ax.plot(
                sub["n"], sub["collapse_rate"],
                marker=markers[formulation], color=colors[formulation],
                label=formulation, linewidth=1.8, markersize=4, alpha=0.85,
            )
        ax.set_title(f"σ = {sigma}")
        ax.set_xlabel("n (modo)")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("Tasa de colapso (30 seeds)")
    axes[0].legend(loc="upper left", fontsize=9)
    plt.suptitle("Tasa de colapso vs n — sin_rayleigh vs con_rayleigh, por nivel de ruido (arch. A1, 30 seeds)")
    plt.tight_layout()

    out_dir = "outputs/summary/figs"
    os.makedirs(out_dir, exist_ok=True)
    outpath = os.path.join(out_dir, "collapse_rate_vs_n_comparison.png")
    plt.savefig(outpath, dpi=160, bbox_inches="tight")
    plt.close()

    print(f"Listo. Figura guardada en: {outpath}")


if __name__ == "__main__":
    main()
