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
from .kl_schedule import kl_weight_at

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NOMINAL = {1: 68.3, 2: 95.4, 3: 99.7}  # cobertura nominal gaussiana


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
    klw_lo    = cfg.get("kl_weight", 1.0 / N_col)
    kl_sched  = cfg.get("kl_schedule", {})  # {} => sin annealing, comportamiento fijo
    n_samples_eval = cfg.get("n_samples_eval", 50)

    torch.manual_seed(seed)

    prior_sigma = cfg.get("prior_sigma", 1.0)
    net = VIRayleighNet(n=n, hidden=HIDDEN, use_sine=USE_SINE, prior_sigma=prior_sigma).to(DEVICE)
    x_col = torch.linspace(0, 1, N_col, dtype=torch.float32).reshape(-1, 1).to(DEVICE)

    optimizer = torch.optim.Adam(net.parameters(), lr=LR0)

    loss_hist, lfisica_hist, kl_hist, E_hist, klw_hist, epochs_logged = [], [], [], [], [], []

    for ep in range(1, EPOCHS + 1):
        klw = kl_weight_at(ep, EPOCHS, klw_lo, **kl_sched)

        optimizer.zero_grad()
        L, Lf, Lpde, Lnorm, KL, integral, E = compute_losses(net, x_col, lam, klw)
        L.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optimizer.step()

        if ep % 200 == 0 or ep == 1 or ep == EPOCHS:
            loss_hist.append(float(L.detach()))
            lfisica_hist.append(float(Lf.detach()))
            kl_hist.append(float(KL.detach()))
            E_hist.append(float(E.detach()))
            klw_hist.append(klw)
            epochs_logged.append(ep)

        if ep % max(1000, EPOCHS // 5) == 0 or ep == 1:
            print(
                f"n={n} ep={ep} L={float(L.detach()):.3e} L_fisica={float(Lf.detach()):.3e} "
                f"KL={float(KL.detach()):.3e} E={float(E.detach()):.6f} "
                f"klw={klw:.5f} integral={float(integral.detach()):.4f}"
            )

    # --- Evaluación: múltiples forward pass para estimar media y banda ---
    net.eval()
    xs = torch.linspace(0, 1, 500, dtype=torch.float32).reshape(-1, 1).to(DEVICE)

    psi_samples = []
    with torch.no_grad():
        for _ in range(n_samples_eval):
            psi_s = net(xs).cpu().numpy().squeeze()
            psi_samples.append(psi_s)
    psi_samples = np.array(psi_samples)

    xs_np = np.linspace(0, 1, 500, dtype=np.float32)
    psi_exact = np.sqrt(2.0) * np.sin(n * math.pi * xs_np)

    for i in range(psi_samples.shape[0]):
        if np.dot(psi_samples[i], psi_exact) < 0:
            psi_samples[i] *= -1

    psi_mean = psi_samples.mean(axis=0)
    psi_std  = psi_samples.std(axis=0)
    np.save(os.path.join(save_dir, "psi_samples.npy"), psi_samples)  

    E_final = E_hist[-1]
    E_exact = float((n * math.pi) ** 2)
    E_rel   = abs(E_final - E_exact) / (abs(E_exact) + 1e-12)
    l2_err  = float(np.sqrt(np.mean((psi_mean - psi_exact) ** 2)))
    rel_l2  = float(np.sqrt(np.trapezoid((psi_mean - psi_exact)**2, xs_np) / np.trapezoid(psi_exact**2, xs_np)))

    # --- NUEVO: chequeo de calibración, integrado aquí mismo ---
    err = np.abs(psi_mean - psi_exact)
    coverage = {}
    for k in (1, 2, 3):
        cov = 100 * np.mean(err <= k * psi_std)
        coverage[k] = cov
    ratio_err_sigma = float(np.median(err / np.maximum(psi_std, 1e-30)))

    print(f"\n--- Calibración ---")
    for k in (1, 2, 3):
        print(f"cobertura +/-{k} sigma: {coverage[k]:5.1f}%   (nominal {NOMINAL[k]}%)")
    print(f"ratio mediano error/sigma: {ratio_err_sigma:.2f}  (~1 => calibrada; >>1 => sobreconfiada)")

    # --- Figura: media +/- 2 sigma vs solución exacta ---
    plt.figure(figsize=(7, 4))
    plt.plot(xs_np, psi_exact, "--", color="black", label="Exacta")
    plt.plot(xs_np, psi_mean, color="tab:blue", label="Media VI-PINN")
    plt.fill_between(xs_np, psi_mean - 2*psi_std, psi_mean + 2*psi_std,
                      color="tab:blue", alpha=0.3, label="±2σ")
    plt.title(f"VI-PINN | n={n} | E={E_final:.4f} | rel={E_rel:.2e} | L2={rel_l2:.2e}")
    plt.xlabel("x"); plt.ylabel("psi"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "vi_mode.png"), dpi=150)
    plt.close()

    # --- Figura: evolución de la pérdida (ahora incluye klw) ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6), sharex=True)
    ax1.semilogy(epochs_logged, loss_hist, label="ELBO (total)")
    ax1.semilogy(epochs_logged, lfisica_hist, label="Física")
    ax1.semilogy(epochs_logged, kl_hist, label="KL")
    ax1.set_ylabel("loss"); ax1.legend()
    ax2.plot(epochs_logged, klw_hist, color="tab:green")
    ax2.set_ylabel("kl_weight"); ax2.set_xlabel("epoch")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "vi_loss.png"), dpi=150)
    plt.close()

    print(f"\n✅ Listo. Resultados guardados en {save_dir}")
    print(f"E final: {E_final:.6f} | E exacta: {E_exact:.6f} | E_rel: {E_rel:.4e} | L2_rel: {rel_l2:.4e}")

    return {
        "n": n, "save_dir": save_dir,
        "psi_mean": psi_mean, "psi_std": psi_std, "psi_exact": psi_exact,
        "E_hist": E_hist, "loss_hist": loss_hist,
        "E_final": E_final, "E_exact": E_exact, "E_rel": E_rel,
        "L2": rel_l2, "rel_l2": rel_l2,
        "coverage": coverage, "ratio_err_sigma": ratio_err_sigma,
    }