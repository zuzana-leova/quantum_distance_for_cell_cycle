#!/usr/bin/env python3
"""Export notebook-ready quantum config manifest from stress test outputs.

This script reads a stress-test JSON manifest and associated matrix artifacts
(saved via --save-matrix), selects top experiments, and writes:
1) standardized opt-parameter .npz files
2) a quantum config JSON compatible with Section 2.7 style entries

Usage example:
  python scripts/export_stress_quantum_configs.py \
    --results-json /path/to/inter_circles_sensitivity_results.json \
    --output-dir /path/to/PQCtrained/eyeglasses \
    --top-k 5 \
    --include-baseline
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


def _safe_float(value: Any, default: float) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def rank_experiments(rows: List[Dict[str, Any]], include_baseline: bool, top_k: int) -> List[Dict[str, Any]]:
    ok_rows = [r for r in rows if str(r.get("status", "")).lower() == "ok"]
    if not include_baseline:
        ok_rows = [r for r in ok_rows if r.get("name") != "baseline"]

    # Prefer low final loss, low model complexity, and high geometric difference.
    ranked = sorted(
        ok_rows,
        key=lambda r: (
            _safe_float(r.get("optimization_loss_final"), 1e18),
            _safe_float(r.get("model_complexity_quantum_s"), 1e18),
            -_safe_float(r.get("geometric_difference_g"), -1e18),
        ),
    )
    return ranked[: max(1, int(top_k))]


def build_entry(dataset_name: str, row: Dict[str, Any], saved_npz_path: Path) -> Dict[str, Any]:
    name = str(row.get("name"))
    return {
        "name": f"quantum_{dataset_name}_{name}",
        "path": str(saved_npz_path.resolve()),
        "n_qubits": int(row["qubits"]),
        "n_layers": int(row["layers"]),
        "pcs": int(row["pcs"]),
        "nonlinearity_param": str(row["nonlinearity"]),
        "alpha_param": float(row["alpha"]),
        "gamma_param": float(row["gamma"]),
        "source_experiment": name,
        "optimization_loss_final": _safe_float(row.get("optimization_loss_final"), np.nan),
        "model_complexity_quantum_s": _safe_float(row.get("model_complexity_quantum_s"), np.nan),
        "geometric_difference_g": _safe_float(row.get("geometric_difference_g"), np.nan),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Export stress-test winners to notebook-ready quantum config manifest")
    parser.add_argument("--results-json", type=str, required=True, help="Path to inter_circles_sensitivity_results.json")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to store exported opt-params and manifest")
    parser.add_argument("--top-k", type=int, default=5, help="Number of best runs to export")
    parser.add_argument("--include-baseline", action="store_true", help="Include baseline row in candidates")
    parser.add_argument("--dataset-name", type=str, default="", help="Optional dataset label override")
    args = parser.parse_args()

    results_json = Path(args.results_json)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    with results_json.open("r") as f:
        manifest = json.load(f)

    experiments = manifest.get("experiments", [])
    if not experiments:
        raise ValueError("No experiments found in results JSON")

    selected = rank_experiments(experiments, include_baseline=args.include_baseline, top_k=args.top_k)
    if not selected:
        raise ValueError("No eligible experiments after filtering")

    data_path = Path(str(manifest.get("data", "unknown")))
    dataset_name = args.dataset_name.strip() or data_path.stem.replace("_raw", "")

    exported_dir = outdir / "exported_opt_params"
    exported_dir.mkdir(parents=True, exist_ok=True)

    quantum_configs: List[Dict[str, Any]] = []
    for row in selected:
        exp_name = str(row.get("name"))
        artifact = row.get("artifact_npz")
        if not artifact:
            raise ValueError(
                "Missing artifact_npz in results. Re-run stress test with --save-matrix to export opt_params."
            )

        artifact_path = Path(str(artifact))
        if not artifact_path.exists():
            raise FileNotFoundError(f"Artifact not found for experiment '{exp_name}': {artifact_path}")

        source = np.load(artifact_path, allow_pickle=True)
        if "opt_params" not in source:
            raise KeyError(f"Artifact missing 'opt_params': {artifact_path}")

        out_npz = exported_dir / f"opt_params_{dataset_name}_{exp_name}.npz"
        metadata = {
            "dataset": dataset_name,
            "experiment": exp_name,
            "config": {
                "pcs": int(row["pcs"]),
                "alpha": float(row["alpha"]),
                "gamma": float(row["gamma"]),
                "nonlinearity": str(row["nonlinearity"]),
                "qubits": int(row["qubits"]),
                "layers": int(row["layers"]),
            },
            "selection_metrics": {
                "optimization_loss_final": _safe_float(row.get("optimization_loss_final"), np.nan),
                "model_complexity_quantum_s": _safe_float(row.get("model_complexity_quantum_s"), np.nan),
                "geometric_difference_g": _safe_float(row.get("geometric_difference_g"), np.nan),
            },
            "source_results_json": str(results_json.resolve()),
            "source_artifact_npz": str(artifact_path.resolve()),
        }

        np.savez_compressed(out_npz, opt_params=source["opt_params"], metadata=metadata)
        quantum_configs.append(build_entry(dataset_name, row, out_npz))

    manifest_out = {
        "dataset": dataset_name,
        "source_results_json": str(results_json.resolve()),
        "selection": {
            "top_k": int(args.top_k),
            "include_baseline": bool(args.include_baseline),
            "ranking": [
                "optimization_loss_final (asc)",
                "model_complexity_quantum_s (asc)",
                "geometric_difference_g (desc)",
            ],
        },
        "quantum_configs": quantum_configs,
    }

    out_manifest = outdir / f"{dataset_name}_final_quantum_configs.json"
    with out_manifest.open("w") as f:
        json.dump(manifest_out, f, indent=2)

    print("Export complete")
    print(f"  Dataset: {dataset_name}")
    print(f"  Selected runs: {len(quantum_configs)}")
    print(f"  Manifest: {out_manifest}")
    print(f"  Params dir: {exported_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
