# Contexto de tesis: PINNs para Schrödinger 1D (actualizado 21 jul 2026)

Documento para subir a una conversación nueva o al conocimiento del proyecto "Tesis" en claude.ai, de forma que cualquier chat nuevo (en cualquier dispositivo) arranque con este contexto sin tener que reexplicarlo.

## Quién y qué

David Steven Maldonado Aponte, físico de formación, profesor de física en secundaria en Bogotá. Cursa la Maestría en Ciencias de la Información y las Comunicaciones (énfasis IA) en la Universidad Distrital Francisco José de Caldas, bajo la dirección de Andrés Leonardo Jutinico (PhD), grupo LASER. GitHub: MechLearn. Defensa prevista noviembre-diciembre de 2026.

Tesis: **Physics-Informed Neural Networks (PINNs) para la ecuación de Schrödinger 1D, con cuantificación de incertidumbre.** Repo principal de trabajo: `pinn-infinite-well` (GitHub, org MechLearn), conectado como carpeta local en el Mac (`~/VirtualNeural/pinn-infinite-well`) y también clonable en el PC con RTX 5070.

## El anteproyecto (documento oficial, nov. 2025) — lo que realmente exige

- **Objetivo general:** desarrollar un algoritmo PINN para Schrödinger 1D, incorporando un esquema de filtrado y VI-PINN, para analizar cuantificación de incertidumbre en tres casos canónicos: pozo infinito, "barrera de potencial" (error de redacción, debía decir **pozo finito** — la barrera es un problema de scattering sin eigenvalores, no encaja) y oscilador armónico.
- **OE1:** PINN determinista base, en los **tres** casos canónicos (exigencia explícita).
- **OE2:** "Integrar un esquema de filtrado" — redactado de forma genérica, **no nombra Kalman**. Pero el resumen, palabras clave, introducción y marco teórico sí nombran explícitamente "Kalman-PINN" como el mecanismo previsto. La hipótesis (11.1) y el objetivo específico 2, en cambio, no lo mencionan — dan margen para sustituirlo, siempre que se implemente *algún* filtro real (no basta con solo probar ruido sobre el baseline sin filtro).
- **OE3:** VI-PINN — aparece en la hipótesis, objetivo general, OE3 y el título del anteproyecto. **No es sustituible por ensembles** sin reescribir el anteproyecto formalmente.
- La metodología (12.1) permite concentrar el filtrado y VI-PINN en un solo caso principal ("según el avance del proyecto"), siempre que el baseline (OE1) cubra los tres casos.

**Nota clave:** el anteproyecto es una hoja de ruta inicial, no un contrato irrompible. El documento final de tesis puede divergir de él (ya pasó con lo de la barrera), siempre que se justifique y el director esté al tanto — no requiere aprobación formal para redactar el documento final distinto, solo coherencia y transparencia con el director antes de la sustentación.

## Ruta de trabajo actual — tres frentes

**Frente 1 — Baseline determinista, tres casos (OE1), obligatorio:**
- Pozo infinito: ✅ hecho (parámetro libre/softplus, Paper 1 aceptado en WEA 2026 Springer CCIS, colapso binario A1 n\*=14 / A2 n\*=9).
- Oscilador armónico: ✅ hecho parcial (Paper 2 en borrador, hallazgo de "convergencia a modo incorrecto", firma L²≈√2, sin resolver — falta deflación).
- Pozo finito: ⏳ pendiente, no iniciado.

**Frente 2 — Profundización en pozo infinito (OE2+OE3), obligatorio, crítico para la sustentación:**
- Rayleigh (cociente de Rayleigh como reformulación variacional de la energía, **no es un filtro** — es una reformulación de la pérdida, técnicamente pertenece a OE1 pero cumple la función de estabilización que busca OE2): ✅ experimentos completos en repo aparte (`pinn-schrodinger-rayleigh`), mejora estabilidad de n≈13 a n≈27-30. Pendiente: migrar a `pinn-infinite-well/experiments/rayleigh/` y redactar Paper 3.
- Ruido en puntos de colación (`x → x+η, η~N(0,σ²)`, sobre el baseline): 🔄 en curso. Código implementado en `pinn-infinite-well` (`src/pinn1d/train.py`, `configs/noise.yaml`, `scripts/run_sweep_noise.py`). Bug corregido: el ruido desordenaba los puntos, rompiendo la integral trapezoidal de normalización — se corrigió reordenando (`torch.sort`) tras perturbar. Smoke test aprobado. Falta correr el barrido completo (A1, σ∈{0,0.01,0.05,0.1}, n=1-20, 5 seeds).
- **Filtro real pendiente de implementar:** se descartó Kalman (evita comprometerse a UKF sobre cientos de pesos) a favor de un **EMA/Polyak averaging de los pesos** (`θ_filtrado = β·θ_filtrado + (1-β)·θ`), que sí cuenta legítimamente como "esquema de filtrado adaptativo" y permite comparar PINN clásica vs PINN filtrada como pide la metodología (12.3). **Aún no implementado — próximo paso técnico.**
- VI-PINN (OE3): ❌ no iniciado, prioridad crítica, único objetivo específico en cero.

**Frente 3 — Ruta expandida (opcional, no bloqueante, proyección doctoral):**
Replica deflación + Rayleigh + ruido + VI-PINN en oscilador armónico y pozo finito por separado. Documentado en `docs/ruta_trabajo_v2.pdf` (PDF con formato tipo anteproyecto) dentro del repo. No es necesario para la maestría.

## Ajustes de redacción pendientes para el documento final de tesis (no el anteproyecto, que queda como histórico)
1. "Barrera de potencial" → "pozo finito" en todo el documento.
2. Quitar menciones específicas a "Kalman-PINN" en resumen/palabras clave/introducción/marco teórico, describir en su lugar el mecanismo real (Rayleigh + ruido + EMA).
3. Mantener VI-PINN intacto en todas las secciones — es innegociable.

## Estado técnico del repositorio (`pinn-infinite-well`)

- Rama `main`, remoto `https://github.com/MechLearn/pinn-infinite-well.git`.
- Últimos commits incluyen: migración de imports (`pinn1d.InfiniteWell.train` → `pinn1d.train`), implementación de ruido en colocación, y este documento de ruta.
- Entorno: conda `pinn311` (Python 3.11), PyTorch con backend MPS (Mac M5) / CUDA (PC RTX 5070), sin TensorFlow.
- Pendiente de limpieza menor: `.gitignore` para artefactos de LaTeX (`*.aux *.log *.out *.toc`), algunos ya commiteados por error.
- `scripts/run_sweep.py` (el viejo, no el nuevo `run_sweep_noise.py`) todavía tiene un `import tensorflow as tf` roto, remanente de antes de migrar a PyTorch — no usar, usar `run_sweep_noise.py` o `run_one.py`.

## Cronograma orientativo (propuesto, sin validar formalmente con el director)

- 21 jul – 1 ago: cerrar barrido de ruido (OE2).
- 4 – 22 ago: migrar Rayleigh, redactar Paper 3, implementar EMA.
- 25 ago – 17 oct: VI-PINN (bloque más grande y crítico).
- 21-23 oct: WEA 2026, presentación oral en Pereira (fecha fija, ya aceptado).
- 24 oct – 21 nov: redacción del documento completo de tesis.
- 24 nov – 5 dic: revisión con director.
- 8 – 19 dic: ensayo y sustentación.

## Estilo de trabajo de David
Prefiere avanzar paso a paso, pedir aclaraciones teóricas antes de implementar, archivos completos y listos para usar (no fragmentos), y feedback visual iterativo. Corrige activamente interpretaciones incorrectas de sus resultados — hay que tomar sus correcciones en serio y ajustar, no defender la posición inicial por inercia. Es exigente con el rigor académico y con que el documento final refleje honestamente lo que se hizo, no lo que se planeó originalmente.
