#!/usr/bin/env bash
#
# Full 5-stage PQC pipeline for toy datasets using run_inter_circles_stress_test.py
# Uses full dataset sampling by setting --sample-size 999999 (clipped to n by the script).
#
# Example:
#   tmux new-session -s toy_sphere
#   tmux attach-session -t toy_sphere
#   cd /home/leova3397/projects/squlearn
#   bash run_full_pqc_pipeline.sh --dataset toy_sphere --torch-device cuda --seed 42

set -euo pipefail

PYTHON="${HOME}/.conda/envs/myenv/bin/python"
STRESS_SCRIPT="scripts/run_inter_circles_stress_test.py"
EXPORT_SCRIPT="scripts/export_stress_quantum_configs.py"

DATASET="toy_sphere"
DATA_PATH=""
SEED=42
TORCH_DEVICE="cuda"
OPT_ITER=100
ADAM_MULT=1
SPSA_SAMPLES=1
TOPK=4
OUTROOT="/data/gpfs/projects/punim0613/zuzana/qphate/results/pqc_full_pipeline"

usage() {
  cat <<'HELP'
Usage: bash run_full_pqc_pipeline.sh [options]

Options:
  --dataset NAME        toy_circle | eyeglasses | inter_circles | toy_sphere
  --data PATH           Optional explicit data file path
  --seed INT            Random seed (default: 42)
  --torch-device DEV    cpu | cuda (default: cuda)
  --opt-iterations INT  Base optimizer iterations (default: 100)
  --adam-multiplier INT Adam iteration multiplier (default: 1)
  --spsa-samples INT    SPSA samples per step (default: 1)
  --top-k INT           Number of finalists to keep (default: 4)
  --outroot DIR         Output root directory
  -h, --help            Show help
HELP
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset) DATASET="$2"; shift 2 ;;
    --data) DATA_PATH="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --torch-device) TORCH_DEVICE="$2"; shift 2 ;;
    --opt-iterations) OPT_ITER="$2"; shift 2 ;;
    --adam-multiplier) ADAM_MULT="$2"; shift 2 ;;
    --spsa-samples) SPSA_SAMPLES="$2"; shift 2 ;;
    --top-k) TOPK="$2"; shift 2 ;;
    --outroot) OUTROOT="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1"; usage; exit 1 ;;
  esac
done

case "$DATASET" in
  toy_circle)
    DEFAULT_DATA="/data/gpfs/projects/punim0613/zuzana/qphate/data/toy_data_x.npz"
    BQ=5; BL=3
    ;;
  eyeglasses)
    DEFAULT_DATA="/data/gpfs/projects/punim0613/zuzana/qphate/data/eyeglasses_raw.npz"
    BQ=6; BL=3
    ;;
  inter_circles)
    DEFAULT_DATA="/data/gpfs/projects/punim0613/zuzana/qphate/data/inter_circles_raw.npz"
    BQ=8; BL=2
    ;;
  toy_sphere)
    DEFAULT_DATA="/data/gpfs/projects/punim0613/zuzana/qphate/data/toy_sphere_raw.npz"
    BQ=5; BL=3
    ;;
  *)
    echo "Unsupported dataset: $DATASET"
    exit 1
    ;;
esac

if [[ -z "$DATA_PATH" ]]; then
  DATA_PATH="$DEFAULT_DATA"
fi

TS="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${OUTROOT}/${DATASET}_${TS}"
STAGE1_DIR="${RUN_DIR}/stage1_sweep"
STAGE2_DIR="${RUN_DIR}/stage2_candidates"
STAGE3_DIR="${RUN_DIR}/stage3_finalists"
STAGE4_DIR="${RUN_DIR}/stage4_final_manifest"
STAGE5_DIR="${RUN_DIR}/stage5_notebook_payload"

mkdir -p "$STAGE1_DIR" "$STAGE2_DIR" "$STAGE3_DIR" "$STAGE4_DIR" "$STAGE5_DIR"

echo "==============================================================="
echo "PQC FULL PIPELINE (5 STAGES)"
echo "==============================================================="
echo "Dataset:       $DATASET"
echo "Data:          $DATA_PATH"
echo "Seed:          $SEED"
echo "Device:        $TORCH_DEVICE"
echo "Opt iterations:$OPT_ITER"
echo "Adam mult:     $ADAM_MULT"
echo "SPSA samples:  $SPSA_SAMPLES"
echo "Top K:         $TOPK"
echo "Run dir:       $RUN_DIR"
echo "==============================================================="

# Stage 1: Full-dataset one-factor sweep with saved artifacts.
echo "[Stage 1/5] Running full-dataset one-factor sweep..."
"$PYTHON" "$STRESS_SCRIPT" \
  --data "$DATA_PATH" \
  --outdir "$STAGE1_DIR" \
  --mode one-factor \
  --sample-size 999999 \
  --pcs 8,12 \
  --alpha 0.8,1.2 \
  --gamma 4,8 \
  --nonlinearity arccos \
  --baseline-pcs 10 \
  --baseline-alpha 1.0 \
  --baseline-gamma 6 \
  --baseline-nonlinearity arctan \
  --baseline-qubits "$BQ" \
  --baseline-layers "$BL" \
  --optimizer-method torch_adam \
  --opt-iterations "$OPT_ITER" \
  --adam-iter-multiplier "$ADAM_MULT" \
  --spsa-samples "$SPSA_SAMPLES" \
  --diag-rbf-gamma config \
  --diag-approx-rank 200 \
  --torch-device "$TORCH_DEVICE" \
  --seed "$SEED" \
  --save-matrix

RESULTS_JSON="${STAGE1_DIR}/inter_circles_sensitivity_results.json"

# Stage 2: Candidate export to notebook-ready quantum configs.
echo "[Stage 2/5] Exporting top candidate parameter sets..."
"$PYTHON" "$EXPORT_SCRIPT" \
  --results-json "$RESULTS_JSON" \
  --output-dir "$STAGE2_DIR" \
  --top-k "$TOPK" \
  --dataset-name "$DATASET"

CANDIDATES_JSON="${STAGE2_DIR}/${DATASET}_final_quantum_configs.json"

# Stage 3: Re-train finalists on full data as single-config confirmation runs.
echo "[Stage 3/5] Re-training finalists as single-config confirmation runs..."
"$PYTHON" - <<PY
import json
import subprocess
from pathlib import Path

py = ${PYTHON@Q}
stress_script = ${STRESS_SCRIPT@Q}
candidates = Path(${CANDIDATES_JSON@Q})
stage3_dir = Path(${STAGE3_DIR@Q})
seed = int(${SEED@Q})
opt_iter = int(${OPT_ITER@Q})
adam_mult = int(${ADAM_MULT@Q})
spsa_samples = int(${SPSA_SAMPLES@Q})
torch_device = ${TORCH_DEVICE@Q}
data_path = ${DATA_PATH@Q}

with candidates.open("r") as f:
    payload = json.load(f)

for cfg in payload.get("quantum_configs", []):
    name = cfg["source_experiment"]
    outdir = stage3_dir / name
    outdir.mkdir(parents=True, exist_ok=True)

    cmd = [
        py, stress_script,
        "--data", data_path,
        "--outdir", str(outdir),
        "--mode", "one-factor",
        "--sample-size", "999999",
        "--pcs", str(cfg["pcs"]),
        "--alpha", str(cfg["alpha_param"]),
        "--gamma", str(cfg["gamma_param"]),
        "--nonlinearity", str(cfg["nonlinearity_param"]),
        "--baseline-pcs", str(cfg["pcs"]),
        "--baseline-alpha", str(cfg["alpha_param"]),
        "--baseline-gamma", str(cfg["gamma_param"]),
        "--baseline-nonlinearity", str(cfg["nonlinearity_param"]),
        "--baseline-qubits", str(cfg["n_qubits"]),
        "--baseline-layers", str(cfg["n_layers"]),
        "--optimizer-method", "torch_adam",
        "--opt-iterations", str(opt_iter),
        "--adam-iter-multiplier", str(adam_mult),
        "--spsa-samples", str(spsa_samples),
        "--diag-rbf-gamma", "config",
        "--diag-approx-rank", "200",
        "--torch-device", torch_device,
        "--seed", str(seed),
        "--save-matrix",
    ]
    print("Running finalist:", name)
    subprocess.run(cmd, check=True)
PY

# Stage 4: Aggregate finalist manifests into final notebook config list.
echo "[Stage 4/5] Building final aggregated quantum config manifest..."
"$PYTHON" - <<PY
import json
from pathlib import Path

stage3 = Path(${STAGE3_DIR@Q})
stage4 = Path(${STAGE4_DIR@Q})
dataset = ${DATASET@Q}
final_configs = []

for exp_dir in sorted([p for p in stage3.iterdir() if p.is_dir()]):
    result_json = exp_dir / "inter_circles_sensitivity_results.json"
    if not result_json.exists():
        continue
    data = json.loads(result_json.read_text())
    rows = data.get("experiments", [])
    ok_rows = [r for r in rows if r.get("status") == "ok"]
    if not ok_rows:
        continue
    row = ok_rows[0]
    artifact = row.get("artifact_npz")
    if not artifact:
        continue
    final_configs.append({
        "name": f"quantum_{dataset}_{row['name']}",
        "path": artifact,
        "n_qubits": int(row["qubits"]),
        "n_layers": int(row["layers"]),
        "pcs": int(row["pcs"]),
        "nonlinearity_param": str(row["nonlinearity"]),
        "alpha_param": float(row["alpha"]),
        "gamma_param": float(row["gamma"]),
        "source_experiment": row["name"],
        "optimization_loss_final": row.get("optimization_loss_final"),
        "model_complexity_quantum_s": row.get("model_complexity_quantum_s"),
        "geometric_difference_g": row.get("geometric_difference_g"),
    })

payload = {
    "dataset": dataset,
    "generated_from": str(stage3),
    "quantum_configs": final_configs,
}
out = stage4 / f"{dataset}_quantum_configs_final.json"
out.write_text(json.dumps(payload, indent=2))
print("Wrote:", out)
print("Configs:", len(final_configs))
PY

# Stage 5: Write notebook drop-in payload for Section 2.7.
echo "[Stage 5/5] Writing notebook drop-in JSON and usage note..."
cp "${STAGE4_DIR}/${DATASET}_quantum_configs_final.json" "${STAGE5_DIR}/quantum_configs_for_notebook.json"
cat > "${STAGE5_DIR}/README.txt" <<TXT
Use this file in Computing_Outlier_Scores_Refactored.ipynb Section 2.7:
  ${STAGE5_DIR}/quantum_configs_for_notebook.json

Python snippet:
  import json
  with open("${STAGE5_DIR}/quantum_configs_for_notebook.json", "r") as f:
      quantum_configs = json.load(f)["quantum_configs"]
TXT

echo ""
echo "==============================================================="
echo "Pipeline complete"
echo "  Stage 1 results: $STAGE1_DIR"
echo "  Stage 2 candidates: $STAGE2_DIR"
echo "  Stage 3 finalists: $STAGE3_DIR"
echo "  Stage 4 final manifest: ${STAGE4_DIR}/${DATASET}_quantum_configs_final.json"
echo "  Stage 5 notebook payload: ${STAGE5_DIR}/quantum_configs_for_notebook.json"
echo "==============================================================="
