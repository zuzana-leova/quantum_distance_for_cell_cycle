#!/usr/bin/env python3
"""Run a structured sensitivity analysis for inter_circles-style data.

The script keeps a single baseline configuration and then sweeps one parameter
family at a time so results can be compared cleanly. It also includes a small
qubit/layer capacity grid to test expressivity separately from data compression.

Outputs:
- a CSV with one row per experiment
- a JSON manifest with configs, metrics, and the baseline reference
- an optional Markdown summary for quick inspection
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import MinMaxScaler

from squlearn.encoding_circuit import ChebyshevPQC
from squlearn.kernel import ProjectedQuantumKernel
from squlearn.kernel import KernelOptimizer
from squlearn.util import Executor

import pennylane as qml
import torch
import scipy.linalg as sla
import scipy.sparse.linalg as spla
from scipy.optimize import minimize as scipy_minimize


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    pcs: int
    alpha: float
    gamma: float
    nonlinearity: str
    qubits: int
    layers: int


@dataclass
class ExperimentMetrics:
    status: str
    shape: Tuple[int, int]
    symmetry_error: float
    diag_mean: float
    diag_std: float
    off_diag_mean: float
    off_diag_std: float
    kernel_mean: float
    kernel_std: float
    kernel_min: float
    kernel_max: float
    condition_number: Optional[float]
    effective_rank_90: Optional[int]
    effective_rank_95: Optional[int]
    spectral_entropy: Optional[float]
    model_complexity_s: Optional[float]
    frobenius_norm: float
    optimization_loss_initial: Optional[float]
    optimization_loss_final: Optional[float]
    optimization_iterations: int
    optimization_time_seconds: float


def parse_int_list(value: str) -> List[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_float_list(value: str) -> List[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_str_list(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def load_array(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        data = np.load(path)
        return np.asarray(data)
    if suffix == ".npz":
        archive = np.load(path, allow_pickle=True)
        for key in ("X", "x", "data", "features"):
            if key in archive:
                return np.asarray(archive[key])
        if len(archive.files) == 1:
            return np.asarray(archive[archive.files[0]])
        raise ValueError(f"No supported array key found in {path}; keys={archive.files}")
    if suffix == ".h5ad":
        try:
            import scanpy as sc
        except Exception as exc:
            raise RuntimeError("scanpy is required to read .h5ad files") from exc
        adata = sc.read_h5ad(path)
        return np.asarray(adata.X)
    raise ValueError(f"Unsupported data format: {path.suffix}")


def make_scaled_data(x: np.ndarray, pcs: int, random_state: int) -> np.ndarray:
    pca = PCA(n_components=pcs, random_state=random_state)
    x_pca = pca.fit_transform(x)
    scaler = MinMaxScaler((-0.9, 0.9))
    return scaler.fit_transform(x_pca)


def kernel_metrics(kernel: np.ndarray, opt_loss_init: Optional[float] = None, opt_loss_final: Optional[float] = None, opt_iters: int = 0, opt_time: float = 0.0) -> ExperimentMetrics:
    k = np.asarray(kernel, dtype=np.float64)
    k = 0.5 * (k + k.T)
    k = np.nan_to_num(k, nan=0.0, posinf=0.0, neginf=0.0)

    diag = np.diag(k)
    off_diag = k[~np.eye(k.shape[0], dtype=bool)] if k.shape[0] > 1 else np.array([], dtype=np.float64)
    symmetry_error = float(np.max(np.abs(k - k.T))) if k.size else 0.0

    condition_number: Optional[float]
    effective_rank_90: Optional[int]
    effective_rank_95: Optional[int]
    spectral_entropy: Optional[float]
    model_complexity_s: Optional[float]

    try:
        eigvals = np.linalg.eigvalsh(k)
        eigvals = np.sort(eigvals)[::-1]
        eigvals_pos = np.clip(eigvals, 0.0, None)
        total = float(np.sum(eigvals_pos))
        if total > 0.0:
            cumsum = np.cumsum(eigvals_pos) / total
            effective_rank_90 = int(np.searchsorted(cumsum, 0.90) + 1)
            effective_rank_95 = int(np.searchsorted(cumsum, 0.95) + 1)
            probs = np.clip(eigvals_pos / total, 1e-12, None)
            spectral_entropy = float(-np.sum(probs * np.log(probs)))
        else:
            effective_rank_90 = None
            effective_rank_95 = None
            spectral_entropy = None

        if eigvals.size and np.min(np.abs(eigvals)) > 1e-12:
            condition_number = float(np.max(np.abs(eigvals)) / np.min(np.abs(eigvals)))
        else:
            condition_number = None

        trace = float(np.trace(k))
        if trace > 1e-12:
            kn = k / trace
            model_complexity_s = float(np.trace(kn @ kn) / (np.trace(kn) ** 2))
        else:
            model_complexity_s = None
    except Exception:
        condition_number = None
        effective_rank_90 = None
        effective_rank_95 = None
        spectral_entropy = None
        model_complexity_s = None

    fro_norm = float(np.linalg.norm(k, ord="fro"))

    return ExperimentMetrics(
        status="ok",
        shape=tuple(k.shape),
        symmetry_error=symmetry_error,
        diag_mean=float(np.mean(diag)) if diag.size else 0.0,
        diag_std=float(np.std(diag)) if diag.size else 0.0,
        off_diag_mean=float(np.mean(off_diag)) if off_diag.size else 0.0,
        off_diag_std=float(np.std(off_diag)) if off_diag.size else 0.0,
        kernel_mean=float(np.mean(k)) if k.size else 0.0,
        kernel_std=float(np.std(k)) if k.size else 0.0,
        kernel_min=float(np.min(k)) if k.size else 0.0,
        kernel_max=float(np.max(k)) if k.size else 0.0,
        condition_number=condition_number,
        effective_rank_90=effective_rank_90,
        effective_rank_95=effective_rank_95,
        spectral_entropy=spectral_entropy,
        model_complexity_s=model_complexity_s,
        frobenius_norm=fro_norm,
        optimization_loss_initial=opt_loss_init,
        optimization_loss_final=opt_loss_final,
        optimization_iterations=opt_iters,
        optimization_time_seconds=opt_time,
    )


def kernel_psd_projection(
    K: np.ndarray,
    eig_tol: float = 1e-12,
    eps_diag: float = 1e-10,
    approx_rank: Optional[int] = None,
) -> np.ndarray:
    """Symmetrize and project a kernel matrix to PSD."""
    K = np.asarray(K)
    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        raise ValueError("K must be square")

    orig_dtype = K.dtype
    K = K.astype(np.float32, copy=False)
    K = 0.5 * (K + K.T)
    K = np.nan_to_num(
        K,
        nan=0.0,
        posinf=float(np.finfo(np.float32).max),
        neginf=-float(np.finfo(np.float32).max),
    )

    n = K.shape[0]
    idx = np.diag_indices(n)
    K = K.copy()
    K[idx] = K[idx] + eps_diag

    approx_rank_int = int(approx_rank) if approx_rank is not None else 0
    do_approx = 0 < approx_rank_int < n - 1
    if not do_approx:
        vals, vecs = sla.eigh(K.astype(np.float64))
        vals_clipped = np.clip(vals, a_min=0.0, a_max=None)
        K_psd = (vecs * vals_clipped) @ vecs.T
        K_psd = 0.5 * (K_psd + K_psd.T)
        K_psd[idx] += eps_diag
        return K_psd.astype(orig_dtype, copy=False)

    k = int(min(max(1, approx_rank_int), n - 2))
    try:
        vals, vecs = spla.eigsh(K, k=k, which="LA", maxiter=None)
    except Exception:
        vals, vecs = sla.eigh(K.astype(np.float64))
        vals_clipped = np.clip(vals, a_min=0.0, a_max=None)
        K_psd = (vecs * vals_clipped) @ vecs.T
        K_psd = 0.5 * (K_psd + K_psd.T)
        K_psd[idx] += eps_diag
        return K_psd.astype(orig_dtype, copy=False)

    vals_clipped = np.clip(vals, a_min=0.0, a_max=None)
    K_psd_approx = (vecs * vals_clipped) @ vecs.T
    K_psd_approx = 0.5 * (K_psd_approx + K_psd_approx.T)
    K_psd_approx[idx] += eps_diag
    return K_psd_approx.astype(orig_dtype, copy=False)


def compute_model_complexity_s(K: np.ndarray) -> Optional[float]:
    """Model complexity score from Huang et al.: s = Tr(K^2) / Tr(K)^2."""
    K = 0.5 * (np.asarray(K, dtype=np.float64) + np.asarray(K, dtype=np.float64).T)
    K = np.nan_to_num(K, nan=0.0, posinf=0.0, neginf=0.0)
    trace = float(np.trace(K))
    if trace <= 1e-12:
        return None
    Kn = K / trace
    denom = float(np.trace(Kn) ** 2)
    if denom <= 1e-20:
        return None
    return float(np.trace(Kn @ Kn) / denom)


def compute_geometric_difference_g(K_classical: np.ndarray, K_quantum: np.ndarray, eps: float = 1e-7) -> float:
    """Compute geometric difference g between classical and quantum kernels."""
    Kc = 0.5 * (np.asarray(K_classical, dtype=np.float64) + np.asarray(K_classical, dtype=np.float64).T)
    Kq = 0.5 * (np.asarray(K_quantum, dtype=np.float64) + np.asarray(K_quantum, dtype=np.float64).T)
    Kc = np.nan_to_num(Kc, nan=0.0, posinf=0.0, neginf=0.0)
    Kq = np.nan_to_num(Kq, nan=0.0, posinf=0.0, neginf=0.0)

    n = Kc.shape[0]
    Kc_reg = Kc + float(eps) * np.eye(n)

    vals_q, vecs_q = np.linalg.eigh(Kq)
    vals_q = np.clip(vals_q, a_min=0.0, a_max=None)
    sqrt_vals_q = np.sqrt(vals_q)
    sqrt_Kq = (vecs_q * sqrt_vals_q) @ vecs_q.T

    vals_c, vecs_c = np.linalg.eigh(Kc_reg)
    vals_c = np.clip(vals_c, a_min=float(eps), a_max=None)
    inv_vals_c = 1.0 / vals_c
    Kc_inv = (vecs_c * inv_vals_c) @ vecs_c.T

    M = sqrt_Kq @ Kc_inv @ sqrt_Kq
    M = 0.5 * (M + M.T)
    max_eigval = float(np.max(np.linalg.eigvalsh(M)))
    return float(np.sqrt(max(max_eigval, 0.0)))


def build_rbf_reference_kernel(x_eval: np.ndarray, gamma: float) -> np.ndarray:
    """Construct a classical RBF reference kernel from evaluated features."""
    D2 = pairwise_distances(x_eval, metric="sqeuclidean")
    K_rbf = np.exp(-float(gamma) * D2)
    K_rbf = 0.5 * (K_rbf + K_rbf.T)
    K_rbf = np.nan_to_num(K_rbf, nan=0.0, posinf=0.0, neginf=0.0)
    return K_rbf


def _torch_adam_spsa_optimize(
    fun: Callable[[np.ndarray], float],
    x0: np.ndarray,
    maxiter: int,
    lr: float,
    eps: float,
    spsa_samples: int,
    spsa_eps_start: float,
    spsa_eps_end: float,
    grad_clip: Optional[float],
    random_state: int,
    device: str,
) -> Tuple[np.ndarray, float, int]:
    """Torch Adam with SPSA gradient estimates, aligned with TEST_Damrich behavior."""
    x0 = np.asarray(x0, dtype=float).ravel()
    if x0.size == 0:
        return x0, float(fun(x0)), 0

    rng = np.random.RandomState(random_state)
    params_t = torch.nn.Parameter(torch.tensor(x0, dtype=torch.float32, device=device))
    opt = torch.optim.Adam([params_t], lr=float(lr), betas=(0.9, 0.999), eps=float(eps))

    best_x = x0.copy()
    best_f = float(fun(best_x))

    for it in range(int(maxiter)):
        p_cpu = params_t.detach().cpu().numpy().astype(float)
        grad_est = np.zeros_like(p_cpu)

        steps = max(1, int(maxiter))
        t = float(min(it, steps)) / float(steps)
        cur_eps = float(spsa_eps_start + (spsa_eps_end - spsa_eps_start) * t)

        for _ in range(max(1, int(spsa_samples))):
            delta = rng.choice([-1.0, 1.0], size=p_cpu.size)
            p_plus = p_cpu + cur_eps * delta
            p_minus = p_cpu - cur_eps * delta
            f_plus = float(fun(p_plus))
            f_minus = float(fun(p_minus))
            grad_est += ((f_plus - f_minus) / (2.0 * cur_eps)) * delta

        grad_est /= float(max(1, int(spsa_samples)))
        grad_t = torch.tensor(grad_est.astype(np.float32), device=device)
        grad_t = torch.nan_to_num(grad_t, nan=0.0, posinf=1e8, neginf=-1e8)
        params_t.grad = grad_t

        if grad_clip is not None:
            try:
                torch.nn.utils.clip_grad_norm_([params_t], max_norm=float(grad_clip))
            except Exception:
                pass

        opt.step()
        opt.zero_grad()

        cur_x = params_t.detach().cpu().numpy().astype(float)
        cur_f = float(fun(cur_x))
        if cur_f < best_f:
            best_f = cur_f
            best_x = cur_x.copy()

    return best_x, float(best_f), int(maxiter)


def make_kernel_and_optimize(
    x: np.ndarray,
    config: ExperimentConfig,
    seed: int,
    opt_iterations: int = 50,
    optimizer_method: str = "torch_adam",
    adam_lr: float = 1e-3,
    adam_eps: float = 1e-8,
    adam_grad_clip: Optional[float] = 2.0,
    spsa_samples: int = 1,
    spsa_eps_start: float = 1e-4,
    spsa_eps_end: float = 1e-2,
    adam_iter_multiplier: int = 3,
    device: str = "cpu",
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray, float, float, int, float]:
    """Optimize PQC parameters and return kernel matrix + optimization info.
    
    Returns: (kernel_matrix, opt_params, loss_initial, loss_final, num_iterations)
    """
    start_time = time.time()
    
    enc = ChebyshevPQC(
        num_qubits=config.qubits,
        num_layers=config.layers,
        num_features=x.shape[1],
        entangling_gate="rzz",
        nonlinearity=config.nonlinearity,
        alpha=config.alpha,
    )
    dev = qml.device("lightning.qubit", wires=config.qubits)
    executor = Executor(dev)
    
    rng = np.random.RandomState(seed)
    initial_params = rng.normal(0.0, 0.05, enc.num_parameters)
    
    kernel_obj = ProjectedQuantumKernel(
        encoding_circuit=enc,
        executor=executor,
        measurement="XYZ",
        outer_kernel="gaussian",
        initial_parameters=initial_params,
        gamma=config.gamma,
    )
    
    # Use the same loss fallback pattern as in the large optimization script.
    loss_fn_eval: Optional[Callable[[np.ndarray], float]] = None
    try:
        from squlearn.kernel.loss import MMDPairwiseDistance
        loss_fn = MMDPairwiseDistance(random_state=seed)
        loss_fn.set_quantum_kernel(kernel_obj)
    except Exception:
        # Fallback: use a simple custom loss
        def fallback_loss_fn_eval(params: np.ndarray) -> float:
            kernel_obj.assign_parameters(params)
            K = kernel_obj.evaluate(x)
            K = np.asarray(K, dtype=np.float64)
            K = 0.5 * (K + K.T)
            # MMD-like: variance of off-diagonal vs diagonal
            off_diag = K[~np.eye(K.shape[0], dtype=bool)]
            if off_diag.size > 0:
                return float(1.0 / (np.std(off_diag) + 1e-8))
            return 1.0
        loss_fn_eval = fallback_loss_fn_eval
        loss_fn = None
    
    # Initial loss
    try:
        if loss_fn is not None:
            loss_initial = float(loss_fn.compute(initial_params, data=x))
        else:
            if loss_fn_eval is None:
                raise RuntimeError("No loss function available for optimization")
            loss_initial = float(loss_fn_eval(initial_params))
    except Exception:
        loss_initial = np.inf
    
    # Shared scalar objective for all optimizer modes.
    def loss_wrapper(params):
        try:
            if loss_fn is not None:
                return float(loss_fn.compute(params, data=x))
            else:
                if loss_fn_eval is None:
                    return 1e10
                return float(loss_fn_eval(params))
        except Exception:
            return 1e10
    
    try:
        chosen = optimizer_method.lower().strip()
        if chosen == "torch_adam":
            adam_steps = int(max(1, opt_iterations * max(1, int(adam_iter_multiplier))))
            opt_params, loss_final, n_evals = _torch_adam_spsa_optimize(
                fun=loss_wrapper,
                x0=initial_params,
                maxiter=adam_steps,
                lr=adam_lr,
                eps=adam_eps,
                spsa_samples=spsa_samples,
                spsa_eps_start=spsa_eps_start,
                spsa_eps_end=spsa_eps_end,
                grad_clip=adam_grad_clip,
                random_state=seed,
                device=device,
            )
        elif chosen == "cobyla":
            opt_result = scipy_minimize(
                loss_wrapper,
                initial_params,
                method="COBYLA",
                options={
                    "maxiter": int(opt_iterations),
                    "rhobeg": 1.0,
                    "tol": 1e-6,
                    "disp": False,
                },
            )
            opt_params = np.asarray(opt_result.x)
            loss_final = float(opt_result.fun)
            n_evals = int(opt_result.nfev) if hasattr(opt_result, "nfev") else int(opt_iterations)
        else:
            raise ValueError(f"Unsupported optimizer_method={optimizer_method}; use 'torch_adam' or 'cobyla'.")
    except Exception as exc:
        if verbose:
            print(f"Optimization failed: {exc}")
        opt_params = initial_params
        loss_final = loss_initial
        n_evals = 0
    
    opt_time = time.time() - start_time
    
    # Evaluate kernel with optimized parameters
    kernel_obj.assign_parameters(opt_params)
    K_opt = np.asarray(kernel_obj.evaluate(x), dtype=np.float64)
    
    return K_opt, opt_params, loss_initial, loss_final, n_evals, opt_time


def build_experiments(baseline: ExperimentConfig, mode: str, pcs_values: Sequence[int], alpha_values: Sequence[float], gamma_values: Sequence[float], nonlinearities: Sequence[str], qubits_values: Sequence[int], layers_values: Sequence[int]) -> List[ExperimentConfig]:
    experiments: List[ExperimentConfig] = [baseline]

    def add_if_new(cfg: ExperimentConfig) -> None:
        if cfg not in experiments:
            experiments.append(cfg)

    if mode in {"one-factor", "all", "sweep"}:
        for pcs in pcs_values:
            add_if_new(replace(baseline, name=f"pcs_{pcs}", pcs=pcs))
        for alpha in alpha_values:
            add_if_new(replace(baseline, name=f"alpha_{alpha:g}", alpha=alpha))
        for gamma in gamma_values:
            add_if_new(replace(baseline, name=f"gamma_{gamma:g}", gamma=gamma))
        for nonlinearity in nonlinearities:
            add_if_new(replace(baseline, name=f"nonlinearity_{nonlinearity}", nonlinearity=nonlinearity))

    if mode in {"capacity", "all", "sweep"}:
        for qubits in qubits_values:
            for layers in layers_values:
                add_if_new(replace(baseline, name=f"capacity_q{qubits}_l{layers}", qubits=qubits, layers=layers))

    if mode == "grid":
        for pcs in pcs_values:
            for alpha in alpha_values:
                for gamma in gamma_values:
                    for nonlinearity in nonlinearities:
                        for qubits in qubits_values:
                            for layers in layers_values:
                                name = f"grid_p{pcs}_a{alpha:g}_g{gamma:g}_{nonlinearity}_q{qubits}_l{layers}"
                                add_if_new(
                                    ExperimentConfig(
                                        name=name,
                                        pcs=pcs,
                                        alpha=alpha,
                                        gamma=gamma,
                                        nonlinearity=nonlinearity,
                                        qubits=qubits,
                                        layers=layers,
                                    )
                                )

    return experiments


def summarize_against_baseline(results: List[Dict[str, object]], baseline_name: str) -> None:
    baseline_row = next((row for row in results if row["name"] == baseline_name), None)
    if baseline_row is None:
        return
    baseline_std = float(baseline_row["kernel_std"])
    baseline_rank = baseline_row.get("effective_rank_90")
    print("\nBaseline reference:")
    print(f"  {baseline_name}: std={baseline_std:.6f}, rank90={baseline_rank}")
    ordered = sorted(results, key=lambda row: (row["family"], row["name"]))
    print("\nTop deviations from baseline by Frobenius distance:")
    baseline_fro = float(baseline_row["frobenius_norm"])
    scored = []
    for row in ordered:
        scored.append((abs(float(row["frobenius_norm"]) - baseline_fro), row))
    for _, row in sorted(scored, key=lambda item: item[0], reverse=True)[:10]:
        print(
            f"  {row['name']:<24s} family={row['family']:<14s} "
            f"std={row['kernel_std']:.6f} rank90={row.get('effective_rank_90')} "
            f"fro={row['frobenius_norm']:.4f}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Structured sensitivity analysis for inter_circles-like data")
    parser.add_argument("--data", type=str, required=True, help="Path to .npz, .npy, or .h5ad data file")
    parser.add_argument("--outdir", type=str, default="results/inter_circles_stress_test", help="Directory for CSV/JSON output")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--sample-size", type=int, default=250, help="Number of rows used for each kernel evaluation")
    parser.add_argument("--mode", type=str, default="one-factor", choices=["one-factor", "capacity", "all", "grid"], help="How aggressively to sweep parameters")
    parser.add_argument("--max-experiments", type=int, default=0, help="Optional cap on the total number of experiments; 0 means no cap")
    parser.add_argument("--pcs", type=str, default="5,10,15", help="PC values to test")
    parser.add_argument("--alpha", type=str, default="0.7,1.0,1.3", help="Alpha values to test")
    parser.add_argument("--gamma", type=str, default="2,4,6,8,12", help="Gamma values to test")
    parser.add_argument("--nonlinearity", type=str, default="arctan,arccos", help="Comma-separated nonlinearity values")
    parser.add_argument("--qubits", type=str, default="5,8,10,12", help="Qubit counts to test")
    parser.add_argument("--layers", type=str, default="1,2,3,4", help="Layer counts to test")
    parser.add_argument("--baseline-pcs", type=int, default=10, help="Baseline number of PCs")
    parser.add_argument("--baseline-alpha", type=float, default=1.0, help="Baseline alpha")
    parser.add_argument("--baseline-gamma", type=float, default=6.0, help="Baseline gamma")
    parser.add_argument("--baseline-nonlinearity", type=str, default="arctan", help="Baseline nonlinearity")
    parser.add_argument("--baseline-qubits", type=int, default=8, help="Baseline qubits")
    parser.add_argument("--baseline-layers", type=int, default=2, help="Baseline layers")
    parser.add_argument("--save-matrix", action="store_true", help="Save sampled kernel matrices for each experiment")
    parser.add_argument("--optimizer-method", type=str, default="torch_adam", choices=["torch_adam", "cobyla"], help="Optimizer to use for parameter fitting")
    parser.add_argument("--opt-iterations", type=int, default=50, help="Base optimizer iterations per experiment")
    parser.add_argument("--adam-lr", type=float, default=1e-3, help="Learning rate for torch_adam mode")
    parser.add_argument("--adam-eps", type=float, default=1e-8, help="Adam epsilon for torch_adam mode")
    parser.add_argument("--adam-grad-clip", type=float, default=2.0, help="L2 gradient clipping for torch_adam mode")
    parser.add_argument("--adam-iter-multiplier", type=int, default=3, help="torch_adam runs opt_iterations * multiplier steps")
    parser.add_argument("--spsa-samples", type=int, default=1, help="Number of SPSA repeats per step")
    parser.add_argument("--spsa-eps-start", type=float, default=1e-4, help="Initial SPSA perturbation size")
    parser.add_argument("--spsa-eps-end", type=float, default=1e-2, help="Final SPSA perturbation size")
    parser.add_argument("--torch-device", type=str, default="cpu", help="Torch device for torch_adam mode, e.g. cpu or cuda")
    parser.add_argument("--diag-rbf-gamma", type=str, default="config", choices=["config", "median"], help="RBF gamma strategy for diagnostic comparison")
    parser.add_argument("--diag-approx-rank", type=int, default=0, help="Optional low-rank PSD projection for diagnostics; 0 disables")
    args = parser.parse_args()

    np.random.seed(args.seed)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    x = load_array(Path(args.data))
    if x.ndim != 2:
        raise ValueError(f"Expected a 2D array, got shape={x.shape}")

    n_samples = x.shape[0]
    sample_size = min(int(args.sample_size), n_samples)
    rng = np.random.RandomState(args.seed)
    sample_idx = rng.choice(n_samples, size=sample_size, replace=False)

    baseline = ExperimentConfig(
        name="baseline",
        pcs=args.baseline_pcs,
        alpha=args.baseline_alpha,
        gamma=args.baseline_gamma,
        nonlinearity=args.baseline_nonlinearity,
        qubits=args.baseline_qubits,
        layers=args.baseline_layers,
    )

    experiments = build_experiments(
        baseline=baseline,
        mode=args.mode,
        pcs_values=parse_int_list(args.pcs),
        alpha_values=parse_float_list(args.alpha),
        gamma_values=parse_float_list(args.gamma),
        nonlinearities=parse_str_list(args.nonlinearity),
        qubits_values=parse_int_list(args.qubits),
        layers_values=parse_int_list(args.layers),
    )

    if args.max_experiments and len(experiments) > args.max_experiments:
        experiments = experiments[: args.max_experiments]

    print("=" * 88)
    print("INTER_CIRCLES SENSITIVITY ANALYSIS")
    print("=" * 88)
    print(f"Data: {args.data}")
    print(f"Shape: {x.shape}")
    print(f"Sample size per experiment: {sample_size}")
    print(f"Mode: {args.mode}")
    print(f"Optimizer: {args.optimizer_method}")
    print(f"Baseline: pcs={baseline.pcs}, alpha={baseline.alpha}, gamma={baseline.gamma}, nonlinearity={baseline.nonlinearity}, qubits={baseline.qubits}, layers={baseline.layers}")
    print(f"Planned experiments: {len(experiments)}")
    print("=" * 88)

    rows: List[Dict[str, object]] = []
    matrices_dir = outdir / "matrices"
    if args.save_matrix:
        matrices_dir.mkdir(exist_ok=True)

    baseline_matrix = None

    for idx, config in enumerate(experiments, start=1):
        family = "baseline"
        if config.name.startswith("pcs_"):
            family = "pcs"
        elif config.name.startswith("alpha_"):
            family = "alpha"
        elif config.name.startswith("gamma_"):
            family = "gamma"
        elif config.name.startswith("nonlinearity_"):
            family = "nonlinearity"
        elif config.name.startswith("capacity_"):
            family = "capacity"
        elif config.name.startswith("grid_"):
            family = "grid"

        x_scaled = make_scaled_data(x, pcs=config.pcs, random_state=args.seed)
        x_eval = x_scaled[sample_idx]

        print(
            f"[{idx:03d}/{len(experiments):03d}] {config.name:<24s} "
            f"pcs={config.pcs:<2d} alpha={config.alpha:<4g} gamma={config.gamma:<4g} "
            f"nl={config.nonlinearity:<7s} q={config.qubits:<2d} layers={config.layers:<2d}",
            end="",
            flush=True,
        )

        try:
            kernel, opt_params, loss_init, loss_final, opt_iters, opt_time = make_kernel_and_optimize(
                x_eval,
                config,
                seed=args.seed,
                opt_iterations=args.opt_iterations,
                optimizer_method=args.optimizer_method,
                adam_lr=args.adam_lr,
                adam_eps=args.adam_eps,
                adam_grad_clip=args.adam_grad_clip,
                spsa_samples=args.spsa_samples,
                spsa_eps_start=args.spsa_eps_start,
                spsa_eps_end=args.spsa_eps_end,
                adam_iter_multiplier=args.adam_iter_multiplier,
                device=args.torch_device,
                verbose=False,
            )
            metrics = kernel_metrics(
                kernel,
                opt_loss_init=loss_init,
                opt_loss_final=loss_final,
                opt_iters=opt_iters,
                opt_time=opt_time,
            )
            approx_rank = int(args.diag_approx_rank) if int(args.diag_approx_rank) > 0 else None
            kernel_psd = kernel_psd_projection(kernel, eig_tol=1e-12, eps_diag=1e-10, approx_rank=approx_rank)

            if args.diag_rbf_gamma == "median":
                D2 = pairwise_distances(x_eval, metric="sqeuclidean")
                upper = D2[np.triu_indices_from(D2, k=1)]
                median_d2 = float(np.median(upper)) if upper.size else 1.0
                rbf_gamma = 1.0 / (2.0 * max(median_d2, 1e-12))
            else:
                rbf_gamma = float(config.gamma)

            kernel_rbf = build_rbf_reference_kernel(x_eval, gamma=rbf_gamma)
            model_complexity_classical_s = compute_model_complexity_s(kernel_rbf)
            geometric_difference_g = compute_geometric_difference_g(kernel_rbf, kernel_psd, eps=1e-7)

            status = "ok"
            if baseline_matrix is None and config.name == baseline.name:
                baseline_matrix = kernel
            frob_delta = 0.0
            if baseline_matrix is not None and config.name != baseline.name and baseline_matrix.shape == kernel.shape:
                frob_delta = float(np.linalg.norm(kernel - baseline_matrix, ord="fro") / (np.linalg.norm(baseline_matrix, ord="fro") + 1e-12))

            row = {
                "name": config.name,
                "family": family,
                **asdict(config),
                "status": status,
                "sample_size": sample_size,
                "relative_frobenius_to_baseline": frob_delta,
                "artifact_npz": str((matrices_dir / f"{config.name}.npz").resolve()) if args.save_matrix else None,
                "diag_rbf_gamma": rbf_gamma,
                "model_complexity_quantum_s": metrics.model_complexity_s,
                "model_complexity_classical_s": model_complexity_classical_s,
                "geometric_difference_g": geometric_difference_g,
                **asdict(metrics),
            }
            rows.append(row)

            if args.save_matrix:
                np.savez_compressed(matrices_dir / f"{config.name}.npz", kernel=kernel, x_eval=x_eval, config=json.dumps(asdict(config)), opt_params=opt_params)

            print(
                f" | loss_init={loss_init:.6f} loss_final={loss_final:.6f} iters={opt_iters} | "
                f"std={metrics.kernel_std:.6f} off_std={metrics.off_diag_std:.6f} "
                f"rank90={metrics.effective_rank_90} delta={frob_delta:.4f} g={geometric_difference_g:.4f}"
            )
        except Exception as exc:
            row = {
                "name": config.name,
                "family": family,
                **asdict(config),
                "status": f"error:{type(exc).__name__}",
                "sample_size": sample_size,
                "relative_frobenius_to_baseline": None,
                "artifact_npz": None,
                "diag_rbf_gamma": None,
                "model_complexity_quantum_s": None,
                "model_complexity_classical_s": None,
                "geometric_difference_g": None,
                "shape": None,
                "symmetry_error": None,
                "diag_mean": None,
                "diag_std": None,
                "off_diag_mean": None,
                "off_diag_std": None,
                "kernel_mean": None,
                "kernel_std": None,
                "kernel_min": None,
                "kernel_max": None,
                "condition_number": None,
                "effective_rank_90": None,
                "effective_rank_95": None,
                "spectral_entropy": None,
                "model_complexity_s": None,
                "frobenius_norm": None,
                "optimization_loss_initial": None,
                "optimization_loss_final": None,
                "optimization_iterations": 0,
                "optimization_time_seconds": 0.0,
                "error": str(exc),
            }
            rows.append(row)
            print(f" | ERROR {type(exc).__name__}: {str(exc)[:120]}")

    csv_path = outdir / "inter_circles_sensitivity_results.csv"
    json_path = outdir / "inter_circles_sensitivity_results.json"
    md_path = outdir / "inter_circles_sensitivity_summary.md"

    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    manifest = {
        "data": str(Path(args.data).resolve()),
        "seed": args.seed,
        "sample_size": sample_size,
        "mode": args.mode,
        "diagnostics": {
            "rbf_gamma_strategy": args.diag_rbf_gamma,
            "psd_projection_approx_rank": int(args.diag_approx_rank),
        },
        "baseline": asdict(baseline),
        "experiments": rows,
    }
    with json_path.open("w") as handle:
        json.dump(manifest, handle, indent=2)

    summarize_against_baseline(rows, baseline.name)

    ranked = sorted(
        [row for row in rows if row["status"] == "ok"],
        key=lambda row: (
            abs(float(row.get("relative_frobenius_to_baseline") or 0.0)),
            -float(row.get("off_diag_std") or 0.0),
        ),
        reverse=True,
    )

    with md_path.open("w") as handle:
        handle.write("# Inter-Circles Sensitivity Summary\n\n")
        handle.write(f"- Data: `{args.data}`\n")
        handle.write(f"- Shape: `{x.shape}`\n")
        handle.write(f"- Sample size: `{sample_size}`\n")
        handle.write(f"- Mode: `{args.mode}`\n")
        handle.write(f"- Baseline: `{baseline}`\n\n")
        handle.write("## Top Deviations\n\n")
        for row in ranked[:10]:
            handle.write(
                f"- {row['name']}: family={row['family']}, std={row['kernel_std']:.6f}, "
                f"off_std={row['off_diag_std']:.6f}, rank90={row.get('effective_rank_90')}, "
                f"delta={row.get('relative_frobenius_to_baseline', 0.0):.4f}, "
                f"g={row.get('geometric_difference_g')}\n"
            )

    print("\nSaved:")
    print(f"  CSV: {csv_path}")
    print(f"  JSON: {json_path}")
    print(f"  Summary: {md_path}")
    if args.save_matrix:
        print(f"  Matrices: {matrices_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())