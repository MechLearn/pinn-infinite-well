# src/pinn1d/vi_rayleigh/losses.py
import torch
from ..derivatives import second_derivative


def compute_losses(model, x_batch, lam, kl_weight):
    """
    ELBO para el pozo infinito con VI-PINN + cociente de Rayleigh.

    ELBO = pérdida_física (verosimilitud esperada, aprox. vía muestra única)
           + kl_weight * KL(q(theta) || p(theta))

    La pérdida física es idéntica en espíritu a con_rayleigh/losses.py:
    residuo de la PDE + normalización, con E calculado vía Rayleigh sobre
    la psi muestreada en este forward pass (pesos re-muestreados por
    reparametrización dentro de model.forward()).

    kl_weight pondera cuánto pesa el prior frente al ajuste a los datos
    (colocación). Valores típicos: kl_weight = 1/N_col (normaliza el KL
    para que no domine sobre la pérdida física cuando N_col es grande).
    """
    psi, psi_xx = second_derivative(model, x_batch)

    psi_sq    = psi.squeeze() ** 2
    psi_val   = psi.squeeze()
    psixx_val = psi_xx.squeeze()
    xb        = x_batch.squeeze()
    dx        = xb[1:] - xb[:-1]

    # Denominador: ∫ψ² dx  (trapecio)
    denominator = torch.sum(0.5 * (psi_sq[1:] + psi_sq[:-1]) * dx)

    # Numerador: -∫ψ·ψ'' dx  (trapecio)
    psi_psixx = psi_val * psixx_val
    numerator = -torch.sum(0.5 * (psi_psixx[1:] + psi_psixx[:-1]) * dx)

    # Cociente de Rayleigh (energía derivada de la psi muestreada)
    E_rayleigh = numerator / (denominator + 1e-8)

    # Residuo PDE
    res  = psi_xx + E_rayleigh * psi
    LPDE = torch.mean(res ** 2)

    # Normalización
    integral = denominator
    Lnorm    = (integral - 1.0) ** 2

    # Pérdida física (igual que en con_rayleigh)
    L_fisica = LPDE + lam * Lnorm

    # Divergencia KL de la red (suma sobre todas las capas bayesianas)
    KL = model.kl_divergence()

    # ELBO negativo (esto es lo que se minimiza)
    L = L_fisica + kl_weight * KL

    return L, L_fisica, LPDE, Lnorm, KL, integral, E_rayleigh