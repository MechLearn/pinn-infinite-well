# scripts/plot_energy_scatter.py
import os
import pandas as pd
import matplotlib.pyplot as plt


def main():
    in_csv = "outputs/summary/all_runs_noise.csv"
    if not os.path.exists(in_csv):
        raise FileNotFoundError(
            f"No existe: {in_csv}. Ejecuta primero scripts/summarize_metrics.py"
        )

    df = pd.read_csv(in_csv)

    out_dir = "outputs/summary/figs"
    os.makedirs(out_dir, exist_ok=True)

    # Un panel por sigma (extraído del nombre del experimento, ej. "A1_s0p05")
    sigmas_presentes = sorted(df["noise_sigma"].unique())
    formulations = ["sin_rayleigh", "con_rayleigh"]
    colors = {"sin_rayleigh": "tab:red", "con_rayleigh": "tab:blue"}

    fig, axes = plt.subplots(1, len(sigmas_presentes), figsize=(5*len(sigmas_presentes), 4), sharey=True)
    if len(sigmas_presentes) == 1:
        axes = [axes]

    for ax, sigma in zip(axes, sigmas_presentes):
        sub = df[(df["noise_sigma"] == sigma) & (df["experiment"].str.startswith("A1"))]
        for formulation in formulations:
            sub_f = sub[sub["formulation"] == formulation]
            if len(sub_f) == 0:
                continue
            ok = sub_f[sub_f["collapse"] == False]
            bad = sub_f[sub_f["collapse"] == True]
            ax.scatter(ok["n"], ok["E_rel"], label=f"{formulation} (no colapsa)",
                       marker="o", color=colors[formulation], alpha=0.6)
            ax.scatter(bad["n"], bad["E_rel"], label=f"{formulation} (colapsa)",
                       marker="x", color=colors[formulation])
        ax.set_yscale("log")
        ax.set_xlabel("n")
        ax.set_title(f"sigma={sigma}")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("E_rel (log scale)")
    axes[0].legend(fontsize=7)
    plt.suptitle("A1: sin_rayleigh vs con_rayleigh — E_rel vs n, por nivel de ruido")
    plt.tight_layout()

    outpath = os.path.join(out_dir, "A1_Erel_scatter_comparison.png")
    plt.savefig(outpath, dpi=160, bbox_inches="tight")
    plt.close()

    print("Listo. Figura guardada en:")
    print(f" - {outpath}")


if __name__ == "__main__":
    main()