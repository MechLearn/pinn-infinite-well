# Ruta de trabajo — PINNs para Schrödinger 1D (v2)

Reestructuración de la ruta de investigación en tres frentes separados, para distinguir claramente lo que exige el anteproyecto de la maestría de lo que es ambición personal/proyección doctoral. Reemplaza la lógica de "ruta rigurosa vs ruta práctica" de la versión anterior (`main-2.pdf`, 21 jul 2026).

## Por qué se reestructura así

El anteproyecto exige el baseline determinista (OE1) en los **tres** casos canónicos, pero el filtrado (OE2) y VI-PINN (OE3) están redactados de forma genérica, sin especificar número de casos, y la metodología (12.1) deja la extensión a otros potenciales condicionada a "el avance del proyecto". Eso permite concentrar la profundidad en un solo caso (pozo infinito) sin incumplir el anteproyecto, siempre que el baseline sí cubra los tres.

---

## Frente 1 — Baseline determinista en los tres casos canónicos (OE1)

Obligatorio para la maestría. Cubre literalmente lo que pide el objetivo específico 1.

| Caso | Estado | Notas |
|---|---|---|
| Pozo infinito | ✅ Hecho | Parámetro libre, Paper 1 (WEA 2026, aceptado). Colapso binario A1/A2. |
| Oscilador armónico | ✅ Hecho (parcial) | Parámetro libre, Paper 2 (borrador). Hallazgo: convergencia a modo incorrecto (L²≈√2), no resuelto — requeriría deflación (ver Frente 3). |
| Pozo finito | ⏳ Pendiente | Reemplaza a "barrera de potencial" (error de redacción del anteproyecto original, a corregir en el documento final). Falta implementar: energías ligadas vía ecuación trascendental, PINN determinista, validación contra referencia numérica. |

---

## Frente 2 — Profundización en el caso principal: Pozo Infinito (OE2 + OE3)

Lo mínimo que cumple el anteproyecto más allá del baseline. Es el frente crítico de cara a la sustentación.

### OE2 — Esquema de filtrado / robustez

| Mecanismo | Estado | Notas |
|---|---|---|
| Cociente de Rayleigh | ✅ Hecho | Experimentos completos (A1/A2, n=1-30, 5 seeds) en `pinn-schrodinger-rayleigh`. Mejora medible de estabilidad (n≈13 → n≈27-30). Se reencuadra como el "esquema de filtrado adaptativo" real de OE2, en vez de Kalman. Falta: migrar a `pinn-infinite-well/experiments/rayleigh/`, redactar Paper 3. |
| Ruido en puntos de colación | 🔄 En curso | Código implementado hoy (`train.py`, `configs/noise.yaml`, `scripts/run_sweep_noise.py`), bug de integral desordenada corregido, smoke test aprobado. Falta correr el barrido completo (A1, σ∈{0,0.01,0.05,0.1}, n=1-20, 5 seeds) y analizar resultados. |
| Filtro de Kalman | ❌ Descartado | No aparece en la hipótesis ni en el objetivo específico 2 (solo en resumen/palabras clave/introducción, secciones narrativas). Se ajustará el documento final para no nombrarlo, describiendo el mecanismo real (Rayleigh + ruido). |

### OE3 — VI-PINN

| Estado | Notas |
|---|---|
| ❌ No iniciado | **Prioridad crítica.** Aparece en la hipótesis, objetivo general, OE3 y el título del anteproyecto — no es sustituible por ensembles sin reescribir el anteproyecto. Pendiente: teoría (ELBO, pesos como distribución q(θ), priors), implementación sobre el pozo infinito, pilotos en modos representativos, comparación de bandas de incertidumbre contra parámetro libre y Rayleigh. |

---

## Frente 3 — Ruta expandida (ambición personal / línea doctoral, no bloqueante)

Esto es lo que antes vivía en `main-2.pdf` como "Ruta Rigurosa". No es necesario para aprobar la maestría — es la proyección hacia una línea doctoral en métodos variacionales neuronales. Replica el pipeline completo en cada caso canónico por separado.

### Pozo infinito (extensión sobre el Frente 2)
- Deflación / ortogonalidad (opcional aquí — el ansatz espectral ya evita convergencia a modo incorrecto en este caso).
- Ensembles como comparación adicional frente a VI-PINN.

### Oscilador armónico (ruta propia completa)
- Deflación espectral — **aquí sí es necesaria**, es lo que falta para resolver el hallazgo de "modo incorrecto" del Paper 2.
- Rayleigh + deflación.
- Ruido en colación.
- VI-PINN.

### Pozo finito (ruta propia completa)
- Rayleigh.
- Ruido en colación.
- VI-PINN.

---

## Qué es obligatorio vs qué es opcional

**Obligatorio para la maestría (Frentes 1 y 2):**
1. Pozo finito — baseline determinista.
2. Barrido completo de ruido en pozo infinito.
3. Migrar y redactar Rayleigh (Paper 3) como el mecanismo real de OE2.
4. VI-PINN en pozo infinito (OE3) — el más crítico y el único en cero.
5. Corregir el documento final: "barrera" → "pozo finito", quitar menciones específicas a Kalman.

**Opcional / trabajo futuro (Frente 3):**
- Deflación en oscilador armónico.
- Réplica completa de Rayleigh + ruido + VI-PINN en oscilador armónico y pozo finito.
- Ensembles comparativos.

---

## Estado actual (resumen ejecutivo)

```
OE1  Pozo infinito     ████████████ 100%
OE1  Oscilador armónico ██████████░  ~85% (falta deflación, no bloqueante)
OE1  Pozo finito        ░░░░░░░░░░░░   0%
OE2  Rayleigh           ███████████░  ~90% (falta migrar + redactar)
OE2  Ruido colación     ██████░░░░░░  ~50% (código listo, falta barrido)
OE3  VI-PINN            ░░░░░░░░░░░░   0%  ← crítico
```
