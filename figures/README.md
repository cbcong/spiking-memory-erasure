# Rebuild the paper figures

After installing the repository's Python requirements, run from the repository root.

```sh
python -m figures.build_figures --output-root reproduced_figures
```

The command writes seven vector PDFs to `reproduced_figures/Manuscript/figs/` and four to
`reproduced_figures/Supplementary/figs/`. It uses archived synthetic experiment
outputs and deterministic simulations. A CPU is sufficient and no dataset downloads
are required.

| Output | Source |
| --- | --- |
| `module.pdf`, `protocol.pdf` | Fixed-point model and certificate calculation |
| `replication.pdf`, `paired_completion.pdf` | Frozen 72-graph confirmation report in `results/` |
| `deadline_sensitivity.pdf`, `graph_drive_shift.pdf` | Confirmation and deadline sweep reports in `results/` |
| `ring_scaling.pdf` | Archived ring scaling and word census in `figures/data/` |
| `ring_raster.pdf` | Replay of the archived single-weak-edge ring with 32 modules |
| `multiweak_layout.pdf`, `multiweak_size_control.pdf` | Archived matched ring layouts and size controls in `figures/data/` |
| `three_module_phases.pdf` | Deterministic replay of the three-module obstruction |
