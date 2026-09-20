# scripts/decompose_variance.py
import numpy as np
import sys

def load_samples(path):
    return np.load(path)  # shape (S, n_x)

def decompose(psi_samples):
    psi_mean = psi_samples.mean(axis=0)
    S = psi_samples.shape[0]

    # Guardarrail: si la media colapso, c_k se vuelve inestable (denominador ~0)
    # y los porcentajes salen con cara de validos pero no significan nada.
    norm = np.mean(psi_mean ** 2)          # aprox de integral(psi^2) en malla uniforme [0,1]
    if norm < 0.5:
        print(f"*** COLAPSO: integral(psi_mean^2) = {norm:.3e} (deberia ser ~1) ***")
        print("    La media es ruido; c_k y los porcentajes NO son interpretables.")
        return None
    print(f"integral(psi_mean^2) = {norm:.4f}  [OK]")

    # Para cada muestra, el mejor factor de escala c_k que minimiza
    # ||c_k * psi_mean - psi_k||^2  ->  c_k = (psi_k . psi_mean) / (psi_mean . psi_mean)
    denom = np.dot(psi_mean, psi_mean)
    c_k = np.array([np.dot(psi_samples[k], psi_mean) / denom for k in range(S)])

    # Parte de escala: reconstruir cada muestra como c_k * psi_mean
    scale_part = c_k[:, None] * psi_mean[None, :]
    # Parte de forma: lo que sobra
    shape_part = psi_samples - scale_part

    var_total = np.var(psi_samples, axis=0).sum()
    var_scale = np.var(scale_part, axis=0).sum()
    var_shape = np.var(shape_part, axis=0).sum()

    pct_scale = 100 * var_scale / var_total
    pct_shape = 100 * var_shape / var_total

    print(f"Varianza total: {var_total:.6e}")
    print(f"  Parte de escala: {pct_scale:.1f}%")
    print(f"  Parte de forma:  {pct_shape:.1f}%")
    print(f"  (c_k: media={c_k.mean():.4f}, std={c_k.std():.4f})")

    return pct_scale, pct_shape

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "psi_samples.npy"
    samples = load_samples(path)
    decompose(samples)