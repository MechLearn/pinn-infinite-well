"""
Compara, lado a lado, la evolución de la integral (norma de psi) y de la
energía aprendida durante el entrenamiento, para un caso de sin_rayleigh
(colapsado) y un caso de con_rayleigh (idealmente también colapsado, para
contraste directo).

Muestra visualmente que en sin_rayleigh la energía se "congela" justo cuando
psi colapsa (gradiente anulado), mientras que en con_rayleigh la energía
sigue moviéndose de forma acoplada a la integral (por la invariancia de
escala del cociente de Rayleigh).

Uso:
    python scripts/plot_energy_freeze_comparison.py \
        <ruta_caso_sin_rayleigh> <ruta_caso_con_rayleigh>

Ejemplo:
    python scripts/plot_energy_freeze_comparison.py \
        outputs/InfiniteWell/noise/runs/sin_rayleigh/A1_s0p1_seed28/mode_24 \
        outputs/InfiniteWell/noise/runs/con_rayleigh/A1_s0p1_seed12/mode_29
"""
import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_case(case_dir):
    hist_path = os.path.join(case_dir, "history.npz")
    if not os.path.exists(hist_path):
        raise FileNotFoundError(f"No existe: {hist_path}")
    data = np.load(hist_path)
    return data["epochs_logged"], data["E_hist"], data["int_hist"]


def plot_column(ax_int, ax_E, epochs, int_hist, E_hist, title, E_color):
    ax_int.plot(epochs, np.clip(int_hist, 1e-13, None), color="#639922", linewidth=1.5)
    ax_int.set_yscale("log")
    ax_int.set_title(title, fontsize=12)
    ax_int.grid(alpha=0.3)
    ax_int.axhline(1e-3, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)

    ax_E.plot(epochs, E_hist, color=E_color, linewidth=1.5)
    ax_E.grid(alpha=0.3)

    below = np.where(int_hist < 1e-3)[0]
    if len(below) > 0:
        freeze_epoch = epochs[below[0]]
        ax_int.axvline(freeze_epoch, color="red", linestyle=":", linewidth=1.2)
        ax_E.axvline(freeze_epoch, color="red", linestyle=":", linewidth=1.2,
                      label=f"colapso ~ época {freeze_epoch}")
        ax_E.legend(fontsize=9)


def main():
    if len(sys.argv) < 3:
        print("Uso: python scripts/plot_energy_freeze_comparison.py <caso_sin_rayleigh> <caso_con_rayleigh>")
        return

    case_sin = sys.argv[1]
    case_con = sys.argv[2]

    epochs_sin, E_sin, int_sin = load_case(case_sin)
    epochs_con, E_con, int_con = load_case(case_con)

    fig, axes = plt.subplots(2, 2, figsize=(13, 7), sharex=False)
    (ax_int_sin, ax_int_con), (ax_E_sin, ax_E_con) = axes

    plot_column(ax_int_sin, ax_E_sin, epochs_sin, int_sin, E_sin,
                "Sin Rayleigh (parámetro libre)", "#E24B4A")
    plot_column(ax_int_con, ax_E_con, epochs_con, int_con, E_con,
                "Con Rayleigh", "#378ADD")

    ax_int_sin.set_ylabel(r"Integral $\int\psi^2\,dx$  (log)")
    ax_E_sin.set_ylabel("Energía E (aprendida)")
    ax_E_sin.set_xlabel("Época")
    ax_E_con.set_xlabel("Época")

    plt.suptitle(
        "¿Por qué la energía puede parecer correcta durante el colapso?\n"
        "Congelamiento del gradiente (sin Rayleigh) vs acoplamiento estructural (con Rayleigh)",
        fontsize=13,
    )
    plt.tight_layout()

    out_dir = "outputs/summary/figs"
    os.makedirs(out_dir, exist_ok=True)
    outpath = os.path.join(out_dir, "energy_freeze_comparison.png")
    plt.savefig(outpath, dpi=170, bbox_inches="tight")
    plt.close()

    print("Listo. Figura guardada en:", outpath)


if __name__ == "__main__":
    main()