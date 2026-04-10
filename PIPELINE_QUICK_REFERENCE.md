# PQC Pipeline - Quick Reference & Architecture

**Latest Update**: April 10, 2026  
**Status**: All 5-stage implementation complete and validated

---

## 🎯 Pipeline at a Glance

```
DATA              ENCODING        PQC TRAINING        SWEEP            MANIFEST        NOTEBOOK
(1000 obs)    PCA→Scaling    ChebyshevPQC+      One-factor      Rank & Select    Generate
     ↓             ↓          Optimization        Sensitivity       Finalists        Scores
   .npz        [-0.9,0.9]    (100 iter)          (36+ configs)    (top-K)          (Plots)
               n_components

Stage 1         Stage 1          Stage 3            Stage 4           Stage 2,4,5      Stage 5
Local           Local         GPU-intensive     GPU-intensive       CPU/GPU         Notebook
(fast)          (fast)        (5 min/config)    (3+ hours total)     (1 hour)        (1 hour)
```

---

## 📋 Dataset Summary

| Name | Shape | Type | Path | Baseline |
|------|-------|------|------|----------|
| `toy_circle` | 1000×2 | 2D circles | `/data/gpfs/.../toy_data_x.npz` | 5q3l, pcs=8 |
| `toy_sphere` | 1000×3 | Swiss roll | `/data/gpfs/.../toy_sphere_raw.npz` | 5q3l, pcs=8 |
| `inter_circles` | 1000×2 | Interlocking | `/data/gpfs/.../inter_circles_raw.npz` | 8q2l, pcs=8 |
| `eyeglasses` | 1000×64 → pcs | Biological | `/data/gpfs/.../eyeglasses_raw.npz` | 6q3l, pcs=12 |

---

## 🔬 PQC Architecture (ChebyshevPQC)

```
Classical Input: x ∈ ℝ^d (e.g., d=12 after PCA)
            ↓
┌───────────────────────────────────┐
│ ENCODING LAYER (data-dependent)   │
│ • Chebyshev basis: T_k(x_i * α)   │
│ • Two-qubit gates (CNOT)          │
│ • Trainable parameter: α          │
└──────────┬──────────────────────────┘
           ↓
      [n_layers × VARIATIONAL LAYER]
      (Trainable rotations & entanglement)
           ↓
┌───────────────────────────────────┐
│ OUTPUT: Circuit Unitary           │
│ U ∈ ℂ^(2^num_qubits × 2^qubits)   │
│ E.g., 6q → 64×64 unitary          │
└───────────────────────────────────┘
```

**Hyperparameters**:
- `num_qubits`: [5-8] qubit registers
- `num_layers`: [2-3] repetitions
- `pcs`: [8,12] encoded features
- `alpha`: [0.8-1.2] encoding weight
- `nonlinearity`: ["arccos", "tanh"]

---

## 📊 Optimization Pipeline

### Loss Function
$$L(\theta) = \frac{1}{n} \|K(\theta) - I_n\|_F^2$$

where $K(\theta)$ = Quantum kernel matrix, $\theta$ = circuit parameters

### Gradient Estimation (SPSA)
- **Single-sample gradient**: Requires only 2 loss evaluations
- **Perturbation**: δ ∼ Bernoulli(±1)
- **Efficiency**: 1 sample/step vs 2p for finite differences

### Optimizer (torch_adam)
```python
{
  "method": "torch_adam",
  "iterations": 100,
  "spsa_samples": 1,
  "epsilon": [1e-4 → 1e-2],  # Adaptive decay
  "device": "cuda"
}
```

**Result**: Optimized parameters saved as `.npz` file

---

## 🧪 Sensitivity Sweep (One-Factor)

**Methodology**: Vary ONE parameter, fix others at baseline

**Sweep Grid** (eyeglasses example):

| Parameter | Baseline | Values | Experiments |
|-----------|----------|--------|------------|
| pcs | 12 | 12, 16 | 2 |
| alpha | 1.0 | 0.8, 1.0, 1.2 | 3 |
| gamma | 6 | 4, 6, 8 | 3 |
| capacity | 6q3l | +0, +1 layer | 2 |
| **Total** | - | - | **2×3×3×2 = 36** |

**Time**: ~3 hours on single GPU

**Output**: `inter_circles_sensitivity_results.json` with metrics for each config

---

## 📁 File Structure & Outputs

### Run Directory Layout

```
/data/gpfs/.../pqc_full_pipeline/
└── eyeglasses_20260410_143000/          # Timestamp-based
    ├── stage1_sweep/
    │   ├── inter_circles_sensitivity_results.json  ← Master manifest
    │   ├── inter_circles_sensitivity_results.csv
    │   ├── matrices/
    │   │   ├── opt_params_pcs8_alpha0.8_gamma4.npz
    │   │   ├── opt_params_pcs12_alpha1.0_gamma6.npz
    │   │   └── ...
    │   └── inter_circles_sensitivity_summary.md
    │
    ├── stage2_candidates/
    │   ├── eyeglasses_final_quantum_configs.json    ← Top-K selected
    │   └── exported_opt_params/
    │       ├── opt_params_config_1.npz
    │       ├── opt_params_config_2.npz
    │       ├── opt_params_config_3.npz
    │       └── opt_params_config_4.npz
    │
    ├── stage3_finalists/
    │   ├── finalist_retraining_results.json
    │   └── finalist_opt_params/
    │       └── ...
    │
    ├── stage4_final_manifest/
    │   ├── final_quantum_configs.json
    │   └── stage5_configs.json
    │
    └── stage5_notebook_payload/
        ├── quantum_configs_for_notebook.json   ← Notebook loads THIS
        ├── README.md
        └── environment_snapshot.json
```

### Key Files

**Stage 1**: `inter_circles_sensitivity_results.json`
```json
{
  "baseline_config": {...},
  "experiments": [
    {
      "name": "config_name",
      "qubits": 6,
      "pcs": 12,
      "alpha": 1.0,
      "gamma": 6,
      "optimization_loss_initial": 0.8",
      "optimization_loss_final": 0.12",
      "artifact_npz": "/path/to/matrices/opt_params_config_name.npz"
    },
    ...
  ]
}
```

**Stage 5**: `quantum_configs_for_notebook.json`
```json
{
  "dataset": "eyeglasses",
  "timestamp": "2026-04-10T14:30:00",
  "quantum_configs": [
    {
      "name": "finalist_1",
      "path_opt_params": "/path/to/opt_params_finalist_1.npz",
      "num_qubits": 6,
      "num_layers": 3,
      "pcs": 12,
      "alpha": 1.0,
      "gamma": 6,
      "nonlinearity": "arccos",
      "loss_final": 0.0950,
      "metrics": {
        "model_complexity": 45.2,
        "geometry_diff": 1.234,
        "effective_rank": 124
      }
    },
    ...
  ]
}
```

---

## 🚀 Quick Start

### Option 1: Full Pipeline (Recommended)

```bash
cd /home/leova3397/projects/squlearn

# Create tmux session
tmux new-session -s pqc -c $PWD

# Launch full 5-stage pipeline
bash run_full_pqc_pipeline.sh \
  --dataset toy_sphere \
  --torch-device cuda \
  --seed 42 \
  --opt-iterations 100 \
  --top-k 4

# Expected output:
# ✓ Stage 1: 3+ hours (GPU)
# ✓ Stage 2: 15 min (CPU)
# ✓ Stage 3: 20 min (GPU)
# ✓ Stage 4: 5 min (CPU)
# ✓ Stage 5: 5 min (disk)
# Total: ~4 hours for toy_sphere
```

### Option 2: Individual Stages

```bash
# Stage 1 only: Sensitivity sweep
python scripts/run_inter_circles_stress_test.py \
  --dataset toy_sphere \
  --data /data/gpfs/.../toy_sphere_raw.npz \
  --sample-size 999999 \
  --pcs-values "8,12" \
  --alpha-values "0.8,1.0,1.2" \
  --gamma-values "4,6,8" \
  --save-matrix \
  --torch-device cuda \
  --outdir results/toy_sphere_stage1

# Stage 2: Export finalists
python scripts/export_stress_quantum_configs.py \
  --results-json results/toy_sphere_stage1/inter_circles_sensitivity_results.json \
  --output-dir results/toy_sphere_stage2 \
  --top-k 4
```

### Option 3: Education Mode

```bash
# Understand the pipeline without executing
python master_pipeline_orchestrator.py \
  --dataset toy_sphere \
  --mode explain

# Validate configuration
python master_pipeline_orchestrator.py \
  --dataset eyeglasses \
  --mode validate
```

---

## 🔧 Integration with Notebook

### In Computing_Outlier_Scores_Refactored.ipynb

**Section 2.7** (quantum config loader):

```python
# Method 1: Explicit path
manifest_path = "/data/gpfs/.../stage5_notebook_payload/quantum_configs_for_notebook.json"
with open(manifest_path) as f:
    manifest = json.load(f)
quantum_configs = manifest["quantum_configs"]

# Method 2: Environment variable (preferred)
export QUANTUM_CONFIG_MANIFEST=/data/gpfs/.../quantum_configs_for_notebook.json

# Method 3: Auto-discovery (if PQC_PIPELINE_ROOT set)
export PQC_PIPELINE_ROOT=/data/gpfs/.../pqc_full_pipeline
# Notebook auto-globs for latest manifest
```

**Sections 2.8+**: For each quantum config in list:

```python
for config in quantum_configs:
    # Load opt_params from config["path_opt_params"]
    # Compute quantum kernel K
    # Extract PHATE scores
    # Plot side-by-side with classical baselines
```

---

## 📈 Metrics & Ranking

**Computed during training**:
- `loss_initial`: Initial kernel loss (before optimization)
- `loss_final`: Final loss (after 100 iterations)
- `model_complexity_s`: Spectral complexity (sum of log eigenvalues)
- `effective_rank_90/95`: Degrees of freedom
- `kernel_mean/std`: Kernel statistics

**Combined ranking score**:
```
rank_score = 0.30 × norm(loss_final) + 0.30 × norm(complexity) - 0.40 × norm(geometry_diff)
```

Lower rank_score → Better config (selected for top-K)

---

## 🎓 Conceptual Flowchart

```
┌──────────────────────────┐
│   Synthetic Data (1000)  │
│   toy_circle, sphere,    │
│   inter_circles, glasses │
└────────────┬─────────────┘
             │
             ↓ (PCA + Scale)
┌──────────────────────────┐
│   Encoded Data (pcs×1000)│
│   [-0.9, 0.9] range      │
└────────────┬─────────────┘
             │
             ↓ (ChebyshevPQC)
┌──────────────────────────┐
│   Quantum Circuit Matrix │
│   U ∈ ℂ^(2^q × 2^q)      │
└────────────┬─────────────┘
             │
             ↓ (ProjectedKernel)
┌──────────────────────────┐
│   Kernel Matrix K (n×n)  │
│   via RBF outer kernel   │
└────────────┬─────────────┘
             │
             ↓ (Optimize)
┌──────────────────────────┐
│  Min ||K - I||_F / n     │
│  torch_adam + SPSA       │
│  100 iterations          │
└────────────┬─────────────┘
             │
             ↓ (Save params)
┌──────────────────────────┐
│   opt_params.npz         │
│   (.artifact_npz path)   │
└────────────┬─────────────┘
             │
      ┌──────┴──────┐
      │             │
      ↓             ↓
[Sweep variant configs with different hyperparams]
      │             │
      └──────┬──────┘
             │
             ↓ (Rank 36 results)
┌──────────────────────────┐
│  Select top-K (e.g., 4)  │
│  by loss/complexity      │
└────────────┬─────────────┘
             │
             ↓ (Package)
┌──────────────────────────┐
│  quantum_configs_for_    │
│  notebook.json           │
└────────────┬─────────────┘
             │
             ↓ (Notebook loads)
┌──────────────────────────┐
│  Compute Scores & Plot   │
│  PHATE + persistence     │
│  Publication figures     │
└──────────────────────────┘
```

---

## ⚙️ Configuration Parameters

### All Adjustable Hyperparameters

```bash
# Baseline circuit architecture
--dataset toy_sphere           # toy_circle|toy_sphere|inter_circles|eyeglasses

# Data preprocessing
--sample-size 999999           # Use full dataset
--pcs 8                         # Principal components (from PCA)

# PQC encoding
--pcs-values "8,12"            # Sweep these values
--alpha-values "0.8,1.0,1.2"  # Chebyshev basis weight
--nonlinearity arccos          # Polynomial basis nonlinearity

# Kernel & projection
--gamma-values "4,6,8"         # RBF kernel bandwidth
--approx-rank 200              # Nystrom low-rank approximation

# Training & optimization
--opt-iterations 100           # torch_adam iterations  
--adam-multiplier 1            # Iteration scaling factor
--spsa-samples 1               # Gradient samples per step
--epsilon 1e-4                 # Initial step size

# Hardware
--torch-device cuda            # cuda|cpu
--seed 42                      # Reproducibility

# Output & artifact management
--save-matrix                  # Save .npz files
--top-k 4                      # Select top-K finalists
--outdir results/...           # Output base directory
```

---

## 🐛 Troubleshooting

| Issue | Cause | Fix |
|-------|-------|-----|
| CUDA out of memory | Large dataset or high dim | Reduce `approx_rank` or use smaller PCA |
| Missing `artifact_npz` | `--save-matrix` not set | Re-run Stage 1 with `--save-matrix` flag |
| Notebook can't find manifest | Path incorrect or outdated | Set `export QUANTUM_CONFIG_MANIFEST=/full/path` |
| Loss not decreasing | Bad learning rate | Increase `--adam-multiplier` or `--opt-iterations` |
| Slow sweep convergence | Too many parameters | Reduce sweep grid (fewer pcs/alpha/gamma values) |

---

## 📚 Key References

- **ChebyshevPQC**: Chebyshev polynomial encoding with learned α parameter
- **ProjectedQuantumKernel**: Nystrom low-rank approximation (approx_rank=200)
- **torch_adam**: Adam optimizer with SPSA gradient estimation (1 sample/step)
- **SPSA**: Simultaneous Perturbation Stochastic Approximation
- **RBF Kernel**: Radial basis function outer kernel with bandwidth γ
- **Frobenius Loss**: ||K - I||_F^2 / n (penalizes deviation from identity)
- **One-Factor Sweep**: Vary single parameter, keep baseline fixed

---

## 📞 Next Steps

1. ✅ Review this guide and `MASTER_PIPELINE_GUIDE.md`
2. ✅ Run in explain mode: `python master_pipeline_orchestrator.py --dataset toy_sphere --mode explain`
3. ✅ Validate: `python master_pipeline_orchestrator.py --dataset eyeglasses --mode validate`
4. 🚀 **Execute**: `bash run_full_pqc_pipeline.sh --dataset toy_sphere --torch-device cuda ...`
5. 📊 Monitor Stage 1 (3+ hours on GPU)
6. 📋 Verify Stage 5: `ls stage5_notebook_payload/quantum_configs_for_notebook.json`
7. 📓 Load in notebook: `export QUANTUM_CONFIG_MANIFEST=...` then run Sections 2.7+
8. 🎨 Export publication figures

---

**Last Updated**: April 10, 2026  
**Status**: All components implemented and validated  
**Ready for execution**: ✅
