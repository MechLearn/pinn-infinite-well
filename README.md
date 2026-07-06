# PINN Infinite Well

Physics-Informed Neural Networks for the 1D infinite square well eigenvalue problem.

## Structure

- `src/pinn1d/` — core PINN modules (model, losses, derivatives, train)
- `configs/` — YAML configuration files
- `scripts/` — run and analysis scripts
- `experiments/` — results organized by objective
  - `baseline/` — OE1: binary collapse characterization
  - `noise/` — OE2: collocation noise experiments
  - `rayleigh/` — OE3: Rayleigh quotient mitigation
  - `vipinn/` — OE4: variational inference PINNs

## Hardware
- Mac M5 (MPS) — prototyping
- PC RTX 5070 (CUDA) — full sweeps

## Author
David Steven Maldonado Aponte
Universidad Distrital Francisco José de Caldas
