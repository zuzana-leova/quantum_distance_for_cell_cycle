# End-to-End PQC Pipeline (Canonical Guide)

This is the single go-to guide for the custom pipeline in this repository.

## Scope

The pipeline covers:

1. Data loading/creation
2. Encoding (PCA + scaling)
3. PQC setup (ChebyshevPQC)
4. PQC training (torch_adam + SPSA)
5. Stress sweeps (one-factor sensitivity)
6. Finalist selection + manifest export
7. Notebook scoring/plots

## Primary Scripts

- `run_full_pqc_pipeline.sh`: orchestrates the full 5-stage run
- `scripts/run_inter_circles_stress_test.py`: training + sensitivity sweep
- `scripts/export_stress_quantum_configs.py`: ranks and exports top configs
- `scripts/Computing_Outlier_Scores_Refactored.ipynb`: scoring and visual analysis

## Compact Workflow

1. Prepare data (`toy_circle`, `toy_sphere`, `inter_circles`, `eyeglasses`).
2. Apply PCA to `pcs` components and scale to `[-0.9, 0.9]`.
3. Train PQC with `torch_adam` and SPSA gradient estimates.
4. Sweep around baseline config:
   - `pcs`: `[8, 12]`
   - `alpha`: `[0.8, 1.0, 1.2]`
   - `gamma`: `[4, 6, 8]`
   - capacity: baseline and baseline+1
5. Rank experiments by final loss + complexity + geometry signal.
6. Export top-K finalists to notebook-ready manifest.
7. Load manifest in notebook section 2.7 and compute final scores/plots.

## Baseline Circuit Defaults

- `toy_circle`: 5q3l
- `toy_sphere`: 5q3l
- `inter_circles`: 8q2l
- `eyeglasses`: 6q3l

## Quick Start

```bash
cd /home/leova3397/projects/squlearn

# Optional sanity checks
python master_pipeline_orchestrator.py --dataset toy_sphere --mode explain
python master_pipeline_orchestrator.py --dataset eyeglasses --mode validate

# Full pipeline run
bash run_full_pqc_pipeline.sh \
  --dataset toy_sphere \
  --torch-device cuda \
  --seed 42 \
  --opt-iterations 100 \
  --adam-multiplier 1 \
  --spsa-samples 1 \
  --top-k 4
```

## Output Contract

Typical output directory:

```text
{outroot}/{dataset}_{timestamp}/
  stage1_sweep/
  stage2_candidates/
  stage3_finalists/
  stage4_final_manifest/
  stage5_notebook_payload/
```

Notebook consumes:

- `stage5_notebook_payload/quantum_configs_for_notebook.json`

## Notes

- Use `--save-matrix` in stress testing so optimized params are persisted as `.npz` artifacts.
- The notebook expects valid paths for exported `opt_params` in the manifest.
