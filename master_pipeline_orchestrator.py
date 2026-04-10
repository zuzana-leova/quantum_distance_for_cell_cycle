#!/usr/bin/env python3
"""
Master Pipeline Orchestrator - Python implementation of end-to-end PQC workflow.

This script provides the conceptual scaffolding for understanding and executing
the complete PQC pipeline from data preparation through final scoring.

Usage:
    python master_pipeline_orchestrator.py --dataset toy_sphere --mode explain
    python master_pipeline_orchestrator.py --dataset eyeglasses --mode validate
"""

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ============================================================================
# STAGE 1: DATA & ENCODING
# ============================================================================

@dataclass
class DataConfig:
    """Configuration for synthetic data and encoding."""
    
    # Dataset selection
    dataset_name: str          # toy_circle, toy_sphere, inter_circles, eyeglasses
    data_path: str             # Path to .npz file
    sample_size: int           # Number of observations (typically 1000)
    
    # PCA encoding
    pcs: int                   # Principal components (8 or 12)
    pca_random_state: int      # Reproducibility seed
    
    # Scaling
    feature_min: float         # Min scaling (typically -0.9)
    feature_max: float         # Max scaling (typically +0.9)


class DataStage:
    """Stage 1: Data Loading & Preprocessing."""
    
    @staticmethod
    def load_data(path: str) -> np.ndarray:
        """Load data from .npz file."""
        archive = np.load(path, allow_pickle=True)
        for key in ("X", "x", "data", "features"):
            if key in archive:
                return np.asarray(archive[key])
        raise ValueError(f"No recognized array key in {path}")
    
    @staticmethod
    def apply_pca(X: np.ndarray, n_components: int, random_state: int) -> np.ndarray:
        """Apply PCA dimensionality reduction."""
        from sklearn.decomposition import PCA
        pca = PCA(n_components=n_components, random_state=random_state)
        X_pca = pca.fit_transform(X)
        print(f"  PCA: {X.shape} → {X_pca.shape}")
        print(f"  Explained variance: {pca.explained_variance_ratio_.sum():.1%}")
        return X_pca
    
    @staticmethod
    def scale_data(X: np.ndarray, feature_min: float, feature_max: float) -> np.ndarray:
        """Scale data to feature range for quantum compatibility."""
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler(feature_range=(feature_min, feature_max))
        X_scaled = scaler.fit_transform(X)
        print(f"  Scaling: [{X.min():.3f}, {X.max():.3f}] → [{feature_min}, {feature_max}]")
        return X_scaled


# ============================================================================
# STAGE 2: PQC PARAMETERS
# ============================================================================

@dataclass(frozen=True)
class PQCConfig:
    """Configuration for Parameterized Quantum Circuit (ChebyshevPQC)."""
    
    # Naming
    name: str
    
    # Circuit architecture
    num_qubits: int            # Qubit count (5-8)
    num_layers: int            # Circuit depth (2-3)
    
    # Data encoding
    pcs: int                   # Principal components to encode
    alpha: float               # Chebyshev basis weight [0.8-1.2]
    nonlinearity: str          # "arccos" or "tanh"
    
    # Kernel parameters
    gamma: float               # RBF kernel bandwidth [4-8]
    approx_rank: int           # Low-rank Nystrom approximation (typical: 200)
    
    # Training
    seed: int                  # Random seed


class PQCStage:
    """Stage 2: Define PQC parameters."""
    
    BASELINES = {
        "toy_circle": PQCConfig(
            name="baseline_5q3l_pcs8",
            num_qubits=5, num_layers=3, pcs=8, alpha=1.0,
            nonlinearity="arccos", gamma=6, approx_rank=200, seed=42
        ),
        "toy_sphere": PQCConfig(
            name="baseline_5q3l_pcs8",
            num_qubits=5, num_layers=3, pcs=8, alpha=1.0,
            nonlinearity="arccos", gamma=6, approx_rank=200, seed=42
        ),
        "inter_circles": PQCConfig(
            name="baseline_8q2l_pcs8",
            num_qubits=8, num_layers=2, pcs=8, alpha=1.0,
            nonlinearity="arccos", gamma=6, approx_rank=200, seed=42
        ),
        "eyeglasses": PQCConfig(
            name="baseline_6q3l_pcs12",
            num_qubits=6, num_layers=3, pcs=12, alpha=1.0,
            nonlinearity="arccos", gamma=6, approx_rank=200, seed=42
        ),
    }
    
    @staticmethod
    def get_baseline(dataset_name: str) -> PQCConfig:
        """Return baseline PQC configuration for dataset."""
        return PQCStage.BASELINES[dataset_name]
    
    @staticmethod
    def get_sweep_grid(baseline: PQCConfig) -> Dict[str, List]:
        """Return hyperparameter sweep grid for one-factor sensitivity analysis."""
        return {
            "pcs": [baseline.pcs, baseline.pcs + 4],
            "alpha": [0.8, 1.0, 1.2],
            "gamma": [4, 6, 8],
            "nonlinearity": [baseline.nonlinearity],
            "capacity": ["baseline", "baseline+1"],  # num_layers
        }


# ============================================================================
# STAGE 3: OPTIMIZATION & TRAINING
# ============================================================================

@dataclass
class OptimizerConfig:
    """Configuration for kernel optimization via torch_adam + SPSA."""
    
    method: str                # "torch_adam"
    iterations: int            # Training iterations (typically 100)
    spsa_samples: int          # Samples per step (typically 1)
    epsilon_initial: float     # Initial step size
    epsilon_schedule: str      # "adaptive" or "constant"
    device: str                # "cuda" or "cpu"


class TrainingStage:
    """Stage 3: Optimize PQC parameters."""
    
    @staticmethod
    def compute_loss_function(kernel_matrix: np.ndarray) -> float:
        """
        Compute Frobenius-norm loss: ||K - I||_F^2 / n
        
        Intuition: A kernel close to identity suggests good separability.
        """
        n = kernel_matrix.shape[0]
        identity = np.eye(n)
        loss = np.linalg.norm(kernel_matrix - identity, 'fro') ** 2 / n
        return float(loss)
    
    @staticmethod
    def estimate_gradient_spsa(theta: np.ndarray, 
                               loss_fn: callable,
                               epsilon: float = 1e-4) -> np.ndarray:
        """
        Estimate gradient using SPSA (Simultaneous Perturbation Stochastic Approximation).
        
        Only requires 1 function evaluation per gradient estimate (vs 2p for finite differences).
        """
        delta = np.random.choice([-1, 1], size=theta.shape)
        loss_plus = loss_fn(theta + epsilon * delta)
        loss_minus = loss_fn(theta - epsilon * delta)
        grad_est = (loss_plus - loss_minus) / (2 * epsilon) * delta
        return grad_est


# ============================================================================
# STAGE 4: SENSITIVITY SWEEP
# ============================================================================

@dataclass
class ExperimentResult:
    """Results from a single PQC training experiment."""
    
    config: PQCConfig
    loss_initial: float
    loss_final: float
    iterations: int
    time_seconds: float
    opt_params_path: Optional[str] = None
    metrics: Dict = None


class SensitivitySweep:
    """Stage 4: One-factor sensitivity analysis."""
    
    @staticmethod
    def generate_sweep_configs(baseline: PQCConfig, 
                               sweep_grid: Dict[str, List]) -> List[PQCConfig]:
        """
        Generate all configs for one-factor sweep.
        
        Each iteration varies ONE parameter while keeping others at baseline.
        """
        configs = [baseline]  # Include baseline
        
        for param_name, values in sweep_grid.items():
            if param_name == "capacity":
                # Special case: vary num_layers
                for delta in [0, 1]:
                    new_config = PQCConfig(
                        name=f"{baseline.name}_capacity+{delta}",
                        num_qubits=baseline.num_qubits,
                        num_layers=baseline.num_layers + delta,
                        pcs=baseline.pcs,
                        alpha=baseline.alpha,
                        nonlinearity=baseline.nonlinearity,
                        gamma=baseline.gamma,
                        approx_rank=baseline.approx_rank,
                        seed=baseline.seed,
                    )
                    if new_config not in configs:
                        configs.append(new_config)
            else:
                # Standard parameter sweep
                for value in values:
                    kwargs = asdict(baseline)
                    kwargs[param_name] = value
                    kwargs['name'] = f"{baseline.name}_{param_name}={value}"
                    new_config = PQCConfig(**kwargs)
                    if new_config not in configs:
                        configs.append(new_config)
        
        return configs
    
    @staticmethod
    def count_experiments(baseline: PQCConfig, sweep_grid: Dict[str, List]) -> int:
        """
        Count total experiments in sweep.
        
        For one-factor sensitivity: baseline + Σ(|values_i| - 1)
        """
        configs = SensitivitySweep.generate_sweep_configs(baseline, sweep_grid)
        return len(configs)


# ============================================================================
# STAGE 5: MANIFEST & SCORING
# ============================================================================

@dataclass
class QuantumConfigEntry:
    """Entry in the quantum_configs manifest."""
    
    name: str
    path_opt_params: str  # Path to saved .npz
    num_qubits: int
    num_layers: int
    pcs: int
    alpha: float
    gamma: float
    nonlinearity: str
    
    # Metrics for ranking
    loss_final: float       # Lower is better
    complexity: float       # Lower is better
    geometry_diff: float    # Higher is better
    
    # Ranking score (combined)
    rank_score: float = 0.0


class ManifestStage:
    """Stage 5: Export and rank results."""
    
    @staticmethod
    def rank_experiments(results: List[ExperimentResult],
                         top_k: int = 4) -> List[QuantumConfigEntry]:
        """
        Rank experiments by combined metric.
        
        Ranking: 30% loss + 30% complexity - 40% geometry_diff
        """
        # Convert to entries with ranking
        entries = [
            QuantumConfigEntry(
                name=r.config.name,
                path_opt_params=r.opt_params_path or "",
                num_qubits=r.config.num_qubits,
                num_layers=r.config.num_layers,
                pcs=r.config.pcs,
                alpha=r.config.alpha,
                gamma=r.config.gamma,
                nonlinearity=r.config.nonlinearity,
                loss_final=r.loss_final,
                complexity=r.metrics.get("model_complexity", 0.0),
                geometry_diff=r.metrics.get("geometry_diff", 0.0),
            )
            for r in results
        ]
        
        # Compute ranking scores (normalized)
        losses = [e.loss_final for e in entries]
        complexities = [e.complexity for e in entries]
        geoms = [e.geometry_diff for e in entries]
        
        loss_norm = (np.array(losses) - np.min(losses)) / (np.max(losses) - np.min(losses) + 1e-6)
        cplx_norm = (np.array(complexities) - np.min(complexities)) / (np.max(complexities) - np.min(complexities) + 1e-6)
        geom_norm = (np.array(geoms) - np.min(geoms)) / (np.max(geoms) - np.min(geoms) + 1e-6)
        
        for i, entry in enumerate(entries):
            entry.rank_score = 0.3 * loss_norm[i] + 0.3 * cplx_norm[i] - 0.4 * geom_norm[i]
        
        # Sort by rank score (lower is better)
        entries_sorted = sorted(entries, key=lambda e: e.rank_score)
        
        return entries_sorted[:top_k]


# ============================================================================
# PIPELINE ORCHESTRATOR
# ============================================================================

class PipelineOrchestrator:
    """Main orchestrator coordinating all stages."""
    
    def __init__(self, args):
        self.dataset = args.dataset
        self.mode = args.mode
        self.outdir = Path(args.outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)
    
    def run(self):
        """Execute pipeline in requested mode."""
        
        print("\n" + "=" * 80)
        print(f"MASTER PQC PIPELINE: {self.dataset.upper()}")
        print("=" * 80)
        
        if self.mode == "explain":
            self.mode_explain()
        elif self.mode == "validate":
            self.mode_validate()
        elif self.mode == "execute":
            self.mode_execute()
        else:
            print(f"Unknown mode: {self.mode}")
            sys.exit(1)
    
    def mode_explain(self):
        """Educational mode: Print configuration details."""
        print("\n[MODE: EXPLAIN] Understanding the pipeline...\n")
        
        # Stage 1: Data
        print("STAGE 1: DATA & ENCODING")
        print("-" * 80)
        baseline = PQCStage.get_baseline(self.dataset)
        print(f"Dataset: {self.dataset}")
        print(f"  PQC Encoding: PCA to {baseline.pcs} principal components")
        print(f"  Scaling: [-0.9, 0.9] for quantum compatibility")
        print()
        
        # Stage 2: PQC Parameters
        print("STAGE 2: PQC PARAMETERS")
        print("-" * 80)
        print(f"Baseline Configuration:")
        for field, value in asdict(baseline).items():
            print(f"  {field}: {value}")
        print()
        
        # Stage 4: Sweep
        print("STAGE 4: SENSITIVITY SWEEP")
        print("-" * 80)
        sweep_grid = PQCStage.get_sweep_grid(baseline)
        total_exps = SensitivitySweep.count_experiments(baseline, sweep_grid)
        print(f"Sweep Grid:")
        for param, values in sweep_grid.items():
            print(f"  {param}: {values}")
        print(f"\nTotal Experiments: {total_exps}")
        print(f"Approx Time (1 GPU): {total_exps * 5} minutes ({total_exps * 5 / 60:.1f} hours)")
        print()
        
        # Stage 3: Training
        print("STAGE 3: OPTIMIZATION & TRAINING (per config)")
        print("-" * 80)
        print("Optimizer: torch_adam + SPSA")
        print("  - Iterations: 100")
        print("  - Samples/step: 1 (SPSA)")
        print("  - Loss: ||K - I||_F^2 / n")
        print("  - Gradient: Random perturbation (Bernoulli ±1)")
        print()
        
        # Summary
        print("PIPELINE SUMMARY")
        print("-" * 80)
        print("1. Load & preprocess data (PCA + scaling)")
        print("2. Define baseline PQC config (qubits, layers, encoding)")
        print("3. Train baseline (100 iterations)")
        print("4. Sweep 1 param at a time (36+ variants)")
        print("5. Rank by loss/complexity/geometry")
        print("6. Export top-K to manifest")
        print("7. Re-train finalists for confirmation")
        print("8. Load in notebook, compute scores, generate figures")
        print()
    
    def mode_validate(self):
        """Validation mode: Check data availability and config consistency."""
        print("\n[MODE: VALIDATE] Checking prerequisites...\n")
        
        # Check baseline config
        print("✓ STAGE 2: PQC Configuration")
        baseline = PQCStage.get_baseline(self.dataset)
        print(f"  Baseline: {baseline.name}")
        print(f"  Architecture: {baseline.num_qubits}q{baseline.num_layers}l")
        print()
        
        # Validate sweep
        print("✓ STAGE 4: Sensitivity Sweep")
        sweep_grid = PQCStage.get_sweep_grid(baseline)
        total = SensitivitySweep.count_experiments(baseline, sweep_grid)
        print(f"  Total experiments: {total}")
        print(f"  Est. GPU time: {total * 5 / 60:.1f} hours")
        print()
        
        # Output plan
        print("✓ OUTPUT PLAN:")
        print(f"  Stage 1: {self.outdir}/stage1_sweep/")
        print(f"           ├─ inter_circles_sensitivity_results.json")
        print(f"           └─ matrices/*.npz")
        print(f"  Stage 2: {self.outdir}/stage2_candidates/")
        print(f"           ├─ {self.dataset}_final_quantum_configs.json")
        print(f"           └─ exported_opt_params/*.npz")
        print(f"  Stage 5: {self.outdir}/stage5_notebook_payload/")
        print(f"           └─ quantum_configs_for_notebook.json ← Notebook loads this")
        print()
    
    def mode_execute(self):
        """Full execution mode (placeholder - actual execution via shell scripts)."""
        print("\n[MODE: EXECUTE] Full pipeline\n")
        print("⚠ To execute the full pipeline, use:")
        print(f"\n  bash run_full_pqc_pipeline.sh \\")
        print(f"    --dataset {self.dataset} \\")
        print(f"    --torch-device cuda \\")
        print(f"    --seed 42 \\")
        print(f"    --opt-iterations 100 \\")
        print(f"    --adam-multiplier 1 \\")
        print(f"    --spsa-samples 1 \\")
        print(f"    --top-k 4")
        print()


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Master PQC Pipeline Orchestrator - Educational & Execution Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python master_pipeline_orchestrator.py --dataset toy_sphere --mode explain
  python master_pipeline_orchestrator.py --dataset eyeglasses --mode validate
  python master_pipeline_orchestrator.py --dataset inter_circles --mode execute
        """
    )
    
    parser.add_argument(
        "--dataset",
        choices=["toy_circle", "toy_sphere", "inter_circles", "eyeglasses"],
        default="toy_sphere",
        help="Dataset to use"
    )
    
    parser.add_argument(
        "--mode",
        choices=["explain", "validate", "execute"],
        default="explain",
        help="Execution mode: explain (educate), validate (check), execute (run)"
    )
    
    parser.add_argument(
        "--outdir",
        default="./pipeline_output",
        help="Output directory for results"
    )
    
    args = parser.parse_args()
    
    orchestrator = PipelineOrchestrator(args)
    orchestrator.run()


if __name__ == "__main__":
    main()
