# src/pinn1d/vi_rayleigh/train.py
import os
import math
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .model import VIRayleighNet
from .losses import compute_losses

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_one_mode_vi(n: int, cfg: dict):
    base_dir = cfg.get("save_dir", "outputs/InfiniteWell/vi_rayleigh/runs")
    exp_name = cfg.get("experiment_name", "vi_test")
    seed     = cfg.get("seed", 0)

    save_dir = os.path.join(base_dir, f"{exp_name}_seed{seed}", f"mode_{n}")
    os.makedirs(save_dir, exist_ok=True)

    print(f"Usando device: {DEVICE}")

    HIDDEN    = cfg["model"]["hidden"]
    USE_SINE  = cfg["model"]["use_sine"]
    N_col     = cfg.get("n_col", max(1024, 2048 * n))
    EPOCHS    = cfg.get("epochs", 8000)
    LR0       = cfg.get("lr", 5e-4)
    lam       = cfg.get("lam", 40.0)
    kl_weight = cfg.get("kl_weight", 1.0 / N_col)
    n_samples_eval = cfg.get("n_samples_eval", 50)  # forward passes para estimar bandas

    torch.manual_seed(seed)

    prior_sigma = cfg.get("prior_sigma", 1.0)
    net = VIRayleighNet(n=n, hidden=HIDDEN, use_sine=USE_SINE, prior_sigma=prior_sigma).to(DEVICE)
    x_col = torch.linspace(0, 1, N_col, dtype=torch.float32).reshape(-1, 1).to(DEVICE)

    optimizer = torch.optim.Adam(net.parameters(), lr=LR0)

    loss_hist, lfisica_hist, kl_hist, E_hist, epochs_logged = [], [], [], [], []

    for ep in range(1, EPOCHS + 1):
        optimizer.zero_grad()
        L, Lf, Lpde, Lnorm, KL, integral, E = compute_losses(net, x_col, lam, kl_weight)
        L.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optimizer.step()

        if ep % 200 == 0 or ep == 1 or ep == EPOCHS:
            loss_hist.append(float(L))
            lfisica_hist.append(float(Lf))
            kl_hist.append(float(KL))
            E_hist.append(float(E))
            epochs_logged.append(ep)

        if ep % max(1000, EPOCHS // 5) == 0 or ep == 1:
            print(
                f"n={n} ep={ep} L={float(L):.3e} L_fisica={float(Lf):.3e} "
                f"KL={float(KL):.3e} E={float(E):.6f} integral={float(integral):.4f}"
            )

    # --- Evaluación: múltiples forward pass para estimar media y banda de incertidumbre ---
    net.eval()
    xs = torch.linspace(0, 1, 500, dtype=torch.float32).reshape(-1, 1).to(DEVICE)

    psi_samples = []
    with torch.no_grad():
        for _ in range(n_samples_eval):
            psi_s = net(xs).cpu().numpy().squeeze()
            psi_samples.append(psi_s)
    psi_samples = np.array(psi_samples)  # shape: (n_samples_eval, 500)

    xs_np = np.linspace(0, 1, 500, dtype=np.float32)
    psi_exact = np.sqrt(2.0) * np.sin(n * math.pi * xs_np)

    # Alinear signo de cada muestra respecto a la referencia
    for i in range(psi_samples.shape[0]):
        if np.dot(psi_samples[i], psi_exact) < 0:
            psi_samples[i] *= -1

    psi_mean = psi_samples.mean(axis=0)
    psi_std  = psi_samples.std(axis=0)

    E_final = E_hist[-1]
    E_exact = float((n * math.pi) ** 2)
    E_rel   = abs(E_final - E_exact) / (abs(E_exact) + 1e-12)
    l2_err  = float(np.sqrt(np.mean((psi_mean - psi_exact) ** 2)))

    # --- Figura: media +/- 2 sigma vs solución exacta ---
    plt.figure(figsize=(7, 4))
    plt.plot(xs_np, psi_exact, "--", color="black", label="Exacta")
    plt.plot(xs_np, psi_mean, color="tab:blue", label="Media VI-PINN")
    plt.fill_between(xs_np, psi_mean - 2*psi_std, psi_mean + 2*psi_std,
                      color="tab:blue", alpha=0.3, label="±2σ")
    plt.title(f"VI-PINN | n={n} | E={E_final:.4f} | rel={E_rel:.2e} | L2={l2_err:.2e}")
    plt.xlabel("x"); plt.ylabel("psi"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "vi_mode.png"), dpi=150)
    plt.close()

    # --- Figura: evolución de la pérdida ---
    plt.figure(figsize=(7, 4))
    plt.semilogy(epochs_logged, loss_hist, label="ELBO (total)")
    plt.semilogy(epochs_logged, lfisica_hist, label="Física")
    plt.semilogy(epochs_logged, kl_hist, label="KL")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "vi_loss.png"), dpi=150)
    plt.close()

    print(f"\n✅ Listo. Resultados guardados en {save_dir}")
    print(f"E final: {E_final:.6f} | E exacta: {E_exact:.6f} | E_rel: {E_rel:.4e} | L2: {l2_err:.4e}")

    return {
        "n": n, "save_dir": save_dir,
        "psi_mean": psi_mean, "psi_std": psi_std, "psi_exact": psi_exact,
        "E_hist": E_hist, "loss_hist": loss_hist,
        "E_final": E_final, "E_exact": E_exact, "E_rel": E_rel, "L2": l2_err,
    }