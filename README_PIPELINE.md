# End-to-End PQC Pipeline

This document is the GitHub-friendly entry point for the full workflow:

1. Artificial data creation/loading
2. Data encoding (PCA + scaling)
3. PQC parameterization (ChebyshevPQC)
4. PQC training (torch_adam + SPSA)
5. Sensitivity sweep stress testing
6. Final candidate selection and manifest export
7. Notebook scoring and figure generation

## Quick Start

```bash
cd /home/leova3397/projects/squlearn

# Learn/validate pipeline shape without long compute
python master_pipeline_orchestrator.py --dataset toy_sphere --mode explain
python master_pipeline_orchestrator.py --dataset eyeglasses --mode validate

# Run full 5-stage pipeline
bash run_full_pqc_pipeline.sh \
  --dataset toy_sphere \
  --torch-device cuda \
  --seed 42 \
  --opt-iterations 100 \
  --adam-multiplier 1 \
  --spsa-samples 1 \
  --top-k 4
```

## Pipeline Stages

### Stage 1: Data and Encoding

- Input datasets: `toy_circle`, `toy_sphere`, `inter_circles`, `eyeglasses`
- Preprocessing:
  - PCA to `pcs` dimensions (typically 8 or 12)
  - MinMax scaling to `[-0.9, 0.9]`
- Purpose: map classical data into quantum-friendly feature space.

### Stage 2: PQC Configuration

- Circuit: `ChebyshevPQC`
- Hyperparameters:
  - `num_qubits`, `num_layers`
  - `pcs`, `alpha`, `nonlinearity`
  - `gamma` (outer RBF kernel)
- Dataset baselines:
  - `toy_circle`: 5q3l
  - `toy_sphere`: 5q3l
  - `inter_circles`: 8q2l
  - `eyeglasses`: 6q3l

### Stage 3: PQC Training

- Optimizer: `torch_adam`
- Gradient estimate: SPSA (`spsa_samples=1` default)
- Typical iterations: 100
- Loss objective: Frobenius loss against identity kernel behavior.

### Stage 4: Sensitivity Sweep Stress Test

Script: `scripts/run_inter_circles_stress_test.py`

- One-factor sweep around baseline:
  - `pcs`: `[8, 12]`
  - `alpha`: `[0.8, 1.0, 1.2]`
  - `gamma`: `[4, 6, 8]`
  - capacity delta: baseline and baseline+1
- Saves per-config metrics and trained artifacts (`.npz`) when `--save-matrix` is enabled.

### Stage 5: Candidate Selection and Final Manifest

Script: `scripts/export_stress_quantum_configs.py`

- Ranks experiments using final loss, complexity, and geometry signal.
- Selects top-K configs and exports notebook-ready manifest.

### Stage 6: Notebook Scoring and Plots

Notebook: `scripts/Computing_Outlier_Scores_Refactored.ipynb`

- Section 2.7 loads `quantum_configs_for_notebook.json`
- Computes kernel-based scores and produces analysis plots.

## Core Scripts

- `run_full_pqc_pipeline.sh`: full 5-stage orchestration
- `scripts/run_inter_circles_stress_test.py`: stress sweeps + training runs
- `scripts/export_stress_quantum_configs.py`: export/rank finalists
- `master_pipeline_orchestrator.py`: explain/validate helper

## Output Layout

Typical structure:

```text
{outroot}/{dataset}_{timestamp}/
  stage1_sweep/
  stage2_candidates/
  stage3_finalists/
  stage4_final_manifest/
  stage5_notebook_payload/
```

Final notebook payload:

- `stage5_notebook_payload/quantum_configs_for_notebook.json`

## Extended Documentation

- `PIPELINE_QUICK_REFERENCE.md`
- `MASTER_PIPELINE_GUIDE.md`
- `START_HERE_PIPELINE.txt` (legacy terminal launcher)
