# Stress Test Invocation Guide

Three ways to run the inter_circles stress test:

## Option 1: Local Execution (CPU/GPU)

```bash
bash ./run_stress_test.sh [OPTIONS]

# Quick test (CPU)
bash ./run_stress_test.sh --dataset inter_circles --seed 42 --mode one-factor

# With GPU (if available)
bash ./run_stress_test.sh --dataset inter_circles --seed 42 --torch-device cuda

# Full diagnostic suite
bash ./run_stress_test.sh \
  --dataset inter_circles \
  --seed 42 \
  --mode one-factor \
  --sample-size 250 \
  --opt-iterations 100 \
  --torch-device cuda \
  --diag-rbf-gamma median \
  --save-matrix
```

## Option 2: Spartan HPC via SLURM Wrapper (Recommended)

```bash
sbatch submit_stress_test.sh [OPTIONS]

# Standard GPU run (A100 partition, 4 hrs, 8 CPUs, 32 GB)
sbatch submit_stress_test.sh \
  --dataset inter_circles \
  --seed 42 \
  --mode one-factor \
  --torch-device cuda

# Monitor job
squeue -u leova3397
squeue -j <JOBID> --format="%i %.9P %.8j %.8u %.2t %.10M %.10l %.6D %R"

# View live output
tail -f logs/stress_test_<JOBID>.log

# View after completion
cat logs/stress_test_<JOBID>.log
cat logs/stress_test_<JOBID>.err
```

## Option 3: Spartan HPC Standalone (Direct Submit)

```bash
sbatch ./submit_stress_test_spartan_gpu.sh
```

This runs with hardcoded defaults (30 experiments, one-factor mode).

---

## Supported Arguments

All scripts accept:

| Argument | Values | Default | Description |
|----------|--------|---------|-------------|
| `--data` | path | (auto-resolve by dataset) | Path to NPZ data file |
| `--dataset` | {inter_circles, custom} | inter_circles | Dataset identifier |
| `--seed` | integer | 42 | Random seed for reproducibility |
| `--mode` | {one-factor, capacity, all, grid} | one-factor | Experiment mode |
| `--sample-size` | integer | 250 | Samples per experiment |
| `--opt-iterations` | integer | 50 | Optimization iterations per config |
| `--optimizer-method` | {torch_adam, cobyla} | torch_adam | Optimizer type |
| `--output` | path | ./inter_circles_results/ | Output directory |
| `--torch-device` | {cpu, cuda} | cuda | Compute device |
| `--no-timestamp` | (flag) | (off) | Skip timestamp in output dir |
| `--save-matrix` | (flag) | (off) | Save per-config kernel matrices |
| `--diag-rbf-gamma` | {config, median} | config | RBF baseline gamma strategy |
| `--diag-approx-rank` | integer | 0 | Low-rank PSD approximation (0=disabled) |
| `--help` | (flag) | - | Print usage information |

---

## Typical Output

```
Results saved to: /home/leova3397/projects/squlearn/inter_circles_results_20260408_153642/
├── inter_circles_sensitivity_results.csv          (One row per experiment + metrics)
├── inter_circles_sensitivity_results.json         (Full manifest with metadata)
├── inter_circles_sensitivity_summary.md           (Markdown report with top deviations)
└── matrices/                                      (Optional: NPZ files per config)
    ├── config_0_kernel_pqk.npz
    ├── config_0_optimized_params.npy
    └── ...
```

---

## Spartan-Specific Notes

**Resource Request**:
- GPU: 1x A100 (gpu-a100 partition)
- CPU: 8 cores (for data loading + parallel gradients)
- Memory: 32 GB
- Wall time: 4 hours (adequate for one-factor with 50 opt iters)

**Module AutoLoad**:
- CUDA 11.4.1 + cuDNN 8.2.4
- Python 3.10.4
- foss/2021b toolchain

**Virtual Environment**:
- Auto-activated from `/home/leova3397/projects/squlearn/.venv/`
- Must be pre-created before job submission

**Output Locations**:
- Job logs: `./logs/stress_test_<JOBID>.log`
- Results: `/data/gpfs/projects/punim0613/zuzana/qphate/results/stress_test_out/`

---

## Performance Guidance

| Mode | Time (1 GPU) | Configs | Opt Iters | CPU/GPU |
|------|---------|---------|-----------|---------|
| one-factor | ~20 min | 6 | 50 | 100% GPU, some CPU I/O |
| capacity | ~15 min | 4 | 50 | 100% GPU |
| grid (4x4x4) | ~3 hrs | 64 | 50 | 100% GPU |
| all (24 combos) | ~1 hr | 24 | 50 | 100% GPU |

**Recommendations**:
- Start with `--mode one-factor` to understand parameter sensitivity
- Use `--torch-device cuda` for A100 (100x faster than CPU)
- Increase `--opt-iterations` if optimization hasn't converged (check loss_final)
- Use `--diag-approx-rank 50` only if memory is constrained (default: full eigh)

---

## Continuation Workflow

1. **Local validation** (5 min): 
   ```bash
   bash ./run_stress_test.sh --dataset inter_circles --mode one-factor --torch-device cpu
   ```

2. **GPU pilot run** (30 min):
   ```bash
   sbatch submit_stress_test.sh --dataset inter_circles --mode one-factor --torch-device cuda
   ```

3. **Analyze outputs**:
   - CSV: Sort by `geometric_difference_g` (quantum signature strength)
   - CSV: Sort by `model_complexity_quantum_s` (kernel expressivity)
   - Find best by `optimization_loss_final` or s-score

4. **Extract best params** (from NPZ):
   ```python
   import numpy as np
   best_params = np.load('matrices/config_5_optimized_params.npy')  # Example
   print(best_params)  # Array shape (n_params,)
   ```

5. **Feed to notebook**:
   - Initialize quantum kernel with `best_params`
   - Run full 18K sample inference with frozen parameters
   - Downstream: PHATE, clustering, noise robustness

---

## Troubleshooting

**`sbatch` command not found**:
- Only available on Spartan nodes, not login nodes
- Use job submission: `sbatch submit_stress_test.sh` from Spartan filesystem

**CUDA out of memory**:
- Reduce `--opt-iterations` or `--sample-size`
- Try `--torch-device cpu` (slower but no memory limit)
- Use `--diag-approx-rank 30` to enable sparse eigen approximation

**Module load failures**:
- Check Spartan module environment: `module avail Python`
- Update module names in `submit_stress_test.sh` if needed

**venv activation fails**:
- Pre-create: `python -m venv /home/leova3397/projects/squlearn/.venv`
- Verify location in both bash scripts matches

**Results not appearing**:
- Check `logs/stress_test_<JOBID>.log` for errors
- Verify output directory path was created
- Ensure `/data/gpfs/projects/punim0613/zuzana/qphate/results/` is writable

---

## Quick Start Examples

### Local CPU (Quick Validation)
```bash
bash run_stress_test.sh --dataset inter_circles --seed 42 --mode one-factor --torch-device cpu
```

### Local GPU (Full Diagnostics)
```bash
bash run_stress_test.sh \
  --dataset inter_circles \
  --seed 42 \
  --mode one-factor \
  --sample-size 250 \
  --opt-iterations 100 \
  --torch-device cuda \
  --diag-rbf-gamma median \
  --save-matrix
```

### Spartan Submission (Recommended)
```bash
sbatch submit_stress_test.sh --dataset inter_circles --seed 42 --torch-device cuda
```

### Monitor Submitted Job
```bash
squeue -u leova3397
tail -f logs/stress_test_*.log
```
