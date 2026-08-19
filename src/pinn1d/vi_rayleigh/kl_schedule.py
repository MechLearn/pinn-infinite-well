#!/usr/bin/env python
"""KL annealing en dos fases para VI-PINN (Bayes by Backprop).
 
Idea: el colapso a kl_weight alto es una barrera de OPTIMIZACIÓN (la cuenca
trivial gana desde la inicialización), no necesariamente la preferencia del
ELBO. Entonces:
 
  Fase 1 (warm_frac del entrenamiento): kl_weight = klw_lo (bajo) — la media
          converge al modo n correcto.
  Fase 2 (resto): rampa GEOMÉTRICA klw_lo -> klw_hi — partiendo de la cuenca
          correcta, el KL alto ya no colapsa la media, solo infla sigma.
 
Si funciona: L2 bajo Y banda visible. Y si con la rampa 0.01 ya no colapsa,
queda demostrado que el umbral es de optimización, no del paisaje ELBO.
 
INTEGRACIÓN en pinn1d/vi_rayleigh/train.py — en el loop, donde hoy usas
cfg['kl_weight'] fijo:
 
    from kl_schedule import kl_weight_at
    klw = kl_weight_at(epoch, cfg['epochs'],
                       klw_lo=cfg['kl_weight'],
                       **cfg.get('kl_schedule', {}))
    loss = physics_loss + klw * kl_term
 
Sin 'kl_schedule' en el cfg el comportamiento es idéntico al actual
(kl_weight constante), así que no rompe ningún experimento previo.
 
Cfg de prueba sugerido (n=9):
 
    cfg = {
        'model': {'hidden': 64, 'use_sine': True},
        'n_col': 2048 * 9,
        'epochs': 24000,            # +6k para que la fase 2 tenga aire
        'lr': 3e-4, 'lam': 300.0,
        'kl_weight': 0.001,         # klw_lo: fase 1 idéntica al run que converge
        'kl_schedule': {'klw_hi': 0.005, 'warm_frac': 0.7},
        'prior_sigma': 0.3, 'seed': 0,
        'experiment_name': 'vi_n9_anneal_1e-3_to_5e-3',
    }
 
Si 0.005 funciona (L2 bajo, banda visible), sube klw_hi a 0.01.
"""
 
 
def kl_weight_at(epoch, epochs, klw_lo, klw_hi=None, warm_frac=0.7):
    """kl_weight en el epoch dado. Constante si no hay klw_hi (retrocompatible)."""
    if klw_hi is None or klw_hi <= klw_lo:
        return klw_lo
    t0 = warm_frac * epochs
    if epoch <= t0:
        return klw_lo
    frac = min(1.0, (epoch - t0) / max(1.0, epochs - t0))   # 0 -> 1 en fase 2
    return klw_lo * (klw_hi / klw_lo) ** frac               # rampa geométrica
 
 
if __name__ == "__main__":
    # mini-verificación visual del schedule
    epochs = 24000
    for e in (0, 10000, 16800, 18000, 20000, 22000, 24000):
        print(f"epoch {e:>6}: klw = {kl_weight_at(e, epochs, 1e-3, 5e-3):.5f}")