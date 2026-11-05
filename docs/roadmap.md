# Roadmap: evidence-gated extensions

Each stage proceeds only if its gate is met on held-out data. Nothing below stage 0 exists. `biophysical/future/interfaces.py` holds interface definitions only.

## Stage 0: single-context expression response (this MVP)
See `benchmarking.md` for what was demonstrated.

## Stage 1: more perturbation data, same scope
- **Work:** apply the pipeline to the Replogle K562 essential screen, whose larger perturbation count gives far more training examples, and evaluate across CRISPRi datasets.
- **Gate:** repeat the unseen-gene comparison with the mean baseline and a noise ceiling; report transfer between screens without pooling them blindly.

## Stage 2: prospective experiment selection
- **Work:** run recommended experiments in a partner lab, or retrospectively on a screen not used in development.
- **Gate:** a pre-specified objective is reached with fewer experiments than random selection, with a confidence interval that excludes zero.

## Stage 3: new cell contexts
- **Data:** the same perturbations measured in different contexts (e.g. Replogle K562 and RPE1).
- **Baseline:** the response measured in the other context, unchanged.
- **Gate:** better than that baseline on perturbations and contexts held out jointly.

## Later modules

Each must state the governing equations, parameters with units, boundary conditions and assumptions before composition with other modules. Any simulated trajectory must be labelled simulated. Modules that run independently are not a whole-cell simulator by virtue of being connected.

| Module | Biological problem | Data needed | Conventional baseline | Validation |
|---|---|---|---|---|
| Mechanistic pathway ODEs | Signalling dynamics of a defined pathway | Time-resolved measurements, kinetic parameters | Published pathway model | Held-out perturbation and time points; parameter identifiability |
| Stochastic kinetics | Expression noise and bursting | Time-resolved or allele-resolved single-cell data | Two-state bursting model | Distributional fit on held-out conditions |
| Reaction-diffusion | Spatial signalling | Spatial, time-resolved imaging | Analytic or finite-volume solution | Numerical convergence, then held-out conditions |
| Regulatory or interaction networks | Propagation of effects | Perturbation data plus a prior network | Network-free model (this benchmark) | Improved held-out prediction; causal claims need interventional tests |
| Learned transition operators, temporal state-space models | Trajectories under perturbation | Time-resolved perturbation data | Persistence and linear dynamics | Multi-step rollout on held-out perturbations |
| Physics-informed neural operators | Fast surrogates for a validated solver | A validated simulator to emulate | The solver itself | Error against the solver, and conservation-law checks |
| Thermodynamic constraints | Physically plausible kinetics | Measured free energies or fluxes | Unconstrained model | Constraint violation and held-out prediction |
