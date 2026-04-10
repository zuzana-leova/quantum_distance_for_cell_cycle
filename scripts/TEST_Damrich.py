# necessary imports
import os
import random
import numpy as np
import time
import json
from datetime import datetime
import datetime as dt_module

# Fix HDF5/h5py version incompatibility issues on some systems (e.g., Spartan HPC)
os.environ['HDF5_DISABLE_VERSION_CHECK'] = '1'

# Reproducibility: define a global SEED and fix RNGs and thread settings.
# You can change SEED once at the top of the notebook if desired.
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
# Optionally limit BLAS/parallel threads to reduce nondeterminism (uncomment if needed)
os.environ.setdefault('OMP_NUM_THREADS', '24')
os.environ.setdefault('MKL_NUM_THREADS', '24')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '24')

# sQUlearn helpers
from squlearn.util import Executor
import os
import numpy as np
import threading
import time
import pandas as pd
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_samples, pairwise_distances
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

def benchmark_trajectory(embedding, labels, root_label="Day 00-03"):
    """
    Calculates Trajectory Correlation to benchmark lineage preservation.
    
    Args:
        embedding: (N, 2) or (N, 3) numpy array of the PHATE/Q-PHATE coordinates.
        labels: (N,) array of string labels (e.g., 'Day 00-03').
        root_label: The string label representing the start of the lineage.
        
    Returns:
        spearman_corr: Correlation between embedding distance-from-root and true time.
    """
    # 1. Convert Labels to Integers (True Time)
    # Ensure they are sorted biologically, not alphabetically!
    unique_ordered = sorted(np.unique(labels)) 
    # Check if sorting is correct manually if needed, e.g.:
    # unique_ordered = ['Day 00-03', 'Day 06-09', ...]
    
    label_map = {val: i for i, val in enumerate(unique_ordered)}
    true_time = np.array([label_map[l] for l in labels])
    
    # 2. Identify Root Centroid in Embedding
    root_indices = np.where(labels == root_label)[0]
    if len(root_indices) == 0:
        raise ValueError(f"Root label '{root_label}' not found in labels.")
        
    root_centroid = np.mean(embedding[root_indices], axis=0).reshape(1, -1)
    
    # 3. Calculate Pseudotime (Distance from Root in Embedding)
    # Using Euclidean distance in the embedding space
    dists_from_root = pairwise_distances(embedding, root_centroid).ravel()
    
    # 4. Calculate Correlation
    # We use Spearman because the relationship is monotonic, not necessarily linear
    corr, p_val = spearmanr(dists_from_root, true_time)
    
    return corr

# Guarded Qiskit/Aer/primitives imports so the notebook runs across different Qiskit stacks.
_HAS_QISKIT = False
_HAS_QISKIT_AER = False
_HAS_QISKIT_PRIMITIVES = False
try:
    import qiskit
    _HAS_QISKIT = True
    # primitives may not exist in older qiskit bundles
    try:
        import qiskit.primitives as _qprims  # type: ignore
        _HAS_QISKIT_PRIMITIVES = True
    except Exception:
        _HAS_QISKIT_PRIMITIVES = False
    # Aer can be provided either via the standalone qiskit-aer package or bundled under qiskit
    try:
        from qiskit_aer import Aer, StatevectorSimulator  # preferred standalone layout
        _HAS_QISKIT_AER = True
    except Exception:
        try:
            # older / alternate layout
            from qiskit import Aer
            try:
                from qiskit.providers.aer import AerSimulator as StatevectorSimulator
            except Exception:
                StatevectorSimulator = None
            _HAS_QISKIT_AER = True
        except Exception:
            Aer = None
            StatevectorSimulator = None
            _HAS_QISKIT_AER = False
except Exception:
    qiskit = Nones
    Aer = None
    StatevectorSimulator = None

# qiskit.circuit.Store may not exist in older Terra releases; provide a safe shim
try:
    from qiskit.circuit import Store  # type: ignore
except Exception:
    Store = None

# Debugging: print qiskit availability and versions (useful after kernel restart)
try:
    _qv = getattr(qiskit, '__version__', None)
except Exception:
    _qv = None
print('qiskit version:', _qv, 'HAS_AER:', _HAS_QISKIT_AER, 'HAS_PRIMITIVES:', _HAS_QISKIT_PRIMITIVES)

from squlearn.encoding_circuit import ChebyshevPQC, HubregtsenEncodingCircuit, HighDimEncodingCircuit
from squlearn.kernel import FidelityKernel, ProjectedQuantumKernel
from squlearn.kernel import QGPR
from squlearn.kernel import KernelOptimizer
from squlearn.kernel.loss import NLL
from squlearn.optimizers import LBFGSB
from scipy.optimize import minimize as scipy_minimize

from typing import Optional
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
import torch
from joblib import Parallel, delayed
from multiprocessing import cpu_count

from squlearn.kernel.loss.kernel_loss_base import KernelLossBase

# Device selection is deferred until after CLI parsing so --use_gpu can be honored
device = None

# Config

# subsampling params - for now using all digits and samples
# N_SAMPLES = 300  # total samples to use
DIGITS = list(range(10))

# kernel related
#KERNEL_NAME ='fidelity_kernel'# 'projected quantum kernel' #other option is "projected quantum kernel"
KERNEL_NAME ='projected quantum kernel'

#circuit relateds
N_QUBITS = 5  # PQK/FQK qubits (3 data + 2 expressivity for 3 PCs)
SEED = 42
NO_LAYERS = 2 # PQK layers (lighter for quick evaluation without optimization)

# optimization related
#LOSS = 'PHATERankPreservationLoss'#'DEMaPLoss' 
#LOSS = 'DEMaPLoss' 
#LOSS = 'QuantumVonNeumannRatioLoss'
#LOSS = 'KTALoss'
LOSS =  'MMDPairwiseDistanceLoss'
loss = LOSS
ITER = 500
#LR = 0.05

JAX = False # whether to use JAX-based FQK

from sklearn import manifold, datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import argparse

# --- Command-line arguments (simple, with sane defaults) ---
parser = argparse.ArgumentParser(description="Run sQUlearn optimization script (batch-friendly)")
parser.add_argument("--data", type=str, default="/home/leova3397/projects/squlearn/examples/tutorials/data/data_sqrt.h5ad", help="Path to .h5ad or .npy data file")
parser.add_argument("--outdir", type=str, default="/data/gpfs/projects/punim0613/zuzana/qphate/results", help="Directory to write outputs")
parser.add_argument("--n_qubits", type=int, default=N_QUBITS, help="Number of qubits")
parser.add_argument("--num_layers", type=int, default=NO_LAYERS, help="Starting number of encoding layers")
parser.add_argument("--max_layers", type=int, default=5, help="Maximum number of layers to iterate to")
parser.add_argument("--iterations", type=int, default=ITER, help="Optimizer iterations per layer")
parser.add_argument("--seed", type=int, default=SEED, help="Random seed")
parser.add_argument("--use_gpu", action="store_true", help="Prefer GPU for torch if available")
parser.add_argument("--adaptive_iterations", type=bool, default=False, help="Use fewer iterations for early layers (speeds up). Default: False (full iterations)")
parser.add_argument("--top3_phate", action="store_true", help="If set, also evaluate and plot PHATE for top-3 candidates (extra full kernel evals). Default: only best model PHATE.")
parser.add_argument("--opt_samples", type=int, default=None, help="Number of cells to use for optimization (None = use all). Recommended: 2000-5000 for speed. Full dataset used for final evaluation.")
parser.add_argument("--loss_subsample", type=int, default=None, help="Row subsample size for loss computation (applies to DEMaP and MMD). Recommend 1000-5000 on large datasets to reduce compute/memory.")
parser.add_argument("--cover_full_dataset", action="store_true", help="Cycle over disjoint batches of size opt_samples until all points are seen each layer. Keeps params between batches.")
parser.add_argument("--gamma_sweep", action="store_true", help="Run logarithmic gamma sweep around median heuristic for ProjectedQuantumKernel and pick best gamma")
# New: control output directory naming/behavior
parser.add_argument("--run_name", type=str, default=None, help="Optional name for run directory. Uses optimization_<run_name> instead of timestamp.")
parser.add_argument("--no_timestamp", action="store_true", help="Do not append timestamp; write to a stable folder (optimization/ or optimization_<run_name>).")
parser.add_argument("--reuse_output", action="store_true", help="If target output folder exists, reuse it instead of creating a new one.")
parser.add_argument("--run_diagnostics", action="store_true", help="Run short diagnostics (disable cache, FD gradients, param deltas).")
parser.add_argument("--gamma_fixed", type=float, default=None, help="Use fixed gamma value (e.g., 50.0) without optimization. If set, skips optimization loop and uses this gamma directly.")
parser.add_argument("--optimizer_method", type=str, default="adam", choices=["adam", "cobyla", "lbfgsb"], help="Optimizer method to use: 'adam' (TorchAdamOptimizer - SPSA), 'cobyla' (scipy COBYLA), or 'lbfgsb' (L-BFGS-B).")
parser.add_argument("--dataset", type=str, default="eyeglasses", choices=["toy_circle", "eyeglasses", "inter_circles", "toy_sphere"], help="Dataset to use: toy_circle, eyeglasses, inter_circles, or toy_sphere. Default: eyeglasses")
args, _ = parser.parse_known_args()

# override global knobs with CLI args so the script is runnable as-is
N_QUBITS = args.n_qubits
NO_LAYERS = args.num_layers
MAX_LAYERS_ARG = args.max_layers
ITER = args.iterations
SEED = args.seed
DATA_PATH = args.data
# Normalize outdir to absolute path to avoid surprises
BASE_OUTDIR = os.path.abspath(args.outdir)
ADAPTIVE_ITERATIONS = args.adaptive_iterations
TOP3_PHATE = args.top3_phate
LOSS_SUBSAMPLE = args.loss_subsample
COVER_FULL_DATASET = args.cover_full_dataset
RUN_DIAGNOSTICS = getattr(args, 'run_diagnostics', False)
GAMMA_FIXED = args.gamma_fixed
OPTIMIZER_METHOD = args.optimizer_method.lower()
DATASET = args.dataset  # Get dataset from command-line argument

# Create output directory according to flags
current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
if args.run_name:
    target_name = f"optimization_{args.run_name}"
elif args.no_timestamp:
    target_name = "optimization"
else:
    target_name = f"optimization_{current_datetime}"

TARGET_OUTDIR = os.path.join(BASE_OUTDIR, target_name)
# If exists and reuse requested, keep; otherwise, if conflict and named, append timestamp
if os.path.exists(TARGET_OUTDIR) and not args.reuse_output:
    if args.run_name or args.no_timestamp:
        TARGET_OUTDIR = f"{TARGET_OUTDIR}_{current_datetime}"
# Ensure base and target exist
os.makedirs(TARGET_OUTDIR, exist_ok=True)

# Select torch device now that CLI args are available (honor --use_gpu)
if getattr(args, 'use_gpu', False):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
        print('Warning: --use_gpu requested but CUDA/MPS not available; falling back to CPU')
else:
    # default auto-selection prefers MPS (mac) then CUDA then CPU
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

print('Using device for tensor operations:', device)

# atomic save helper
def atomic_save_npz(path: str, **arrays):
    tmp = str(path) + ".tmp"
    try:
        np.savez_compressed(tmp, **arrays)
        os.replace(tmp, str(path))
    finally:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass


def periodic_saver(stop_event: threading.Event, interval: float, kernel_opt_obj, out_dir: str):
    """Periodically save best-known parameters from KernelOptimizer to out_dir."""
    last_saved = None
    while not stop_event.is_set():
        try:
            params = None
            # try common attribute names that may hold current best params
            if hasattr(kernel_opt_obj, '_optimal_parameters') and getattr(kernel_opt_obj, '_optimal_parameters') is not None:
                params = getattr(kernel_opt_obj, '_optimal_parameters')
            elif hasattr(kernel_opt_obj, 'parameters') and getattr(kernel_opt_obj, 'parameters') is not None:
                params = getattr(kernel_opt_obj, 'parameters')
            elif hasattr(kernel_opt_obj, '_parameters') and getattr(kernel_opt_obj, '_parameters') is not None:
                params = getattr(kernel_opt_obj, '_parameters')
            if params is not None:
                params = np.asarray(params).ravel()
                # try to capture loss / iteration info from kernel optimizer if available
                loss_val = None
                it_val = None
                try:
                    if hasattr(kernel_opt_obj, '_loss_history') and getattr(kernel_opt_obj, '_loss_history'):
                        loss_val = float(getattr(kernel_opt_obj, '_loss_history')[-1])
                    elif hasattr(kernel_opt_obj, '_last_loss'):
                        loss_val = float(getattr(kernel_opt_obj, '_last_loss'))
                    elif hasattr(kernel_opt_obj, 'optimizer') and hasattr(kernel_opt_obj.optimizer, 'history') and getattr(kernel_opt_obj.optimizer, 'history'):
                        loss_val = float(getattr(kernel_opt_obj.optimizer, 'history')[-1])
                except Exception:
                    loss_val = None
                try:
                    # iteration counters naming varies; probe common names
                    if hasattr(kernel_opt_obj, 'nit'):
                        it_val = int(getattr(kernel_opt_obj, 'nit'))
                    elif hasattr(kernel_opt_obj, 'iteration'):
                        it_val = int(getattr(kernel_opt_obj, 'iteration'))
                    elif hasattr(kernel_opt_obj, '_iteration'):
                        it_val = int(getattr(kernel_opt_obj, '_iteration'))
                    elif hasattr(kernel_opt_obj, '_nit'):
                        it_val = int(getattr(kernel_opt_obj, '_nit'))
                    elif hasattr(kernel_opt_obj, 'optimizer') and hasattr(kernel_opt_obj.optimizer, 'iteration'):
                        it_val = int(getattr(kernel_opt_obj.optimizer, 'iteration'))
                except Exception:
                    it_val = None

                if last_saved is None or not np.array_equal(last_saved, params):
                    ts = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
                    fname = f'checkpoint_params_{ts}.npz'
                    out_path = os.path.join(out_dir, fname)
                    meta = {'timestamp': ts}
                    if loss_val is not None:
                        meta['loss'] = float(loss_val)
                    if it_val is not None:
                        meta['iteration'] = int(it_val)
                    try:
                        atomic_save_npz(out_path, opt_params=params, metadata=meta)
                    except Exception:
                        try:
                            np.savez_compressed(out_path, opt_params=params, metadata=meta)
                        except Exception:
                            pass
                    # also write (overwrite) a consistent 'latest' file for quick access
                    latest_path = os.path.join(out_dir, 'opt_params_latest.npz')
                    try:
                        atomic_save_npz(latest_path, opt_params=params, metadata=meta)
                    except Exception:
                        try:
                            np.savez_compressed(latest_path, opt_params=params, metadata=meta)
                        except Exception:
                            pass
                    last_saved = params.copy()
        except Exception:
            pass
        stop_event.wait(interval)


# ------------------------
# 0) Load data
# ------------------------


import numpy as np
dataset = DATASET  # Get dataset from command-line argument
nonlinearity_param = 'arccos'

if dataset =="toy_circle":
    data = np.load('/data/gpfs/projects/punim0613/zuzana/qphate/data/toy_data_x.npz')
    x = data['x']
    pcs = 6
    #alpha_param = 1.0
    alpha_param = 0.7
    gamma_param = 2.0
    nonlinearity_param = 'arctan'

if dataset =="eyeglasses":
    data = np.load('/data/gpfs/projects/punim0613/zuzana/qphate/data/eyeglasses_raw.npz')
    x = data['X']
    pcs = 8
    alpha_param = 1.0
    gamma_param = 7.0
    nonlinearity_param = 'arctan'


if dataset =="inter_circles":
    data = np.load('/data/gpfs/projects/punim0613/zuzana/qphate/data/inter_circles_raw.npz')
    x = data['X']
    pcs = 10
    alpha_param = 1.0
    gamma_param = 6.0
    nonlinearity_param = 'arctan'


if dataset =="toy_sphere":
    data = np.load('/data/gpfs/projects/punim0613/zuzana/qphate/data/toy_sphere_raw.npz')
    x = data['X']
    pcs = 6
    alpha_param = 0.7
    gamma_param = 1.2
    nonlinearity_param = 'arctan'
#x_outliers = data['x_outliers']

pca = PCA(n_components=pcs, random_state=SEED)
X_pca = pca.fit_transform(x)
scaler = MinMaxScaler((-0.9, 0.9))
X_scaled= scaler.fit_transform(X_pca)
print(f"Final data shape after PCA+scaling: {X_scaled.shape}")

X_scaled_opt = X_scaled
opt_sample_idx = None

print(f"Optimization data shape (full dataset): {X_scaled_opt.shape}")
print(f"Final evaluation data shape: {X_scaled.shape}")
print(f"Full data shape (for final results): {X_scaled.shape}")

# Executor with the Aer statevector simulator
#executor = StatevectorSimulator()
num_qubits = N_QUBITS


enc_circ = ChebyshevPQC(num_qubits, num_layers=NO_LAYERS,
                        num_features=X_scaled.shape[1],  # Match the 20 PCA components
                        entangling_gate = 'rzz',
                         nonlinearity =  nonlinearity_param,
                          alpha = alpha_param )  # Reduced for lower-dimensional encoding
    

# --- OPTION 2: The "Standard" Benchmark (ZZ-Feature Map style) ---
# This circuit uses Ry rotations for features and CNOTs for entanglement
# It is a standard "Hardware Efficient Ansatz" often used as a baseline.

# enc_circ = HubregtsenEncodingCircuit(
#     num_qubits=N_QUBITS,
#     num_features=20,
#     num_layers=NO_LAYERS,
#     closed=False,     # False = Linear Entanglement (Chain). True = Ring.
#     final_encoding=False
# )
#enc_circ = HighDimEncodingCircuit(num_qubits=num_qubits,num_layers=NO_LAYERS)

#Robust DEMaPLoss implementation: sanitizes inputs and avoids ConstantInputWarning
# and SciPy procrustes errors by short-circuiting degenerate inputs and using
# a closed-form classical MDS (double-centering) instead of iterative SMACOF/procrustes.
import pennylane as qml
dev = qml.device("lightning.qubit", wires=N_QUBITS)  # backend param is illustrative
executor = Executor(dev)
jax = JAX
#executor = Executor(Aer.get_backend("aer_simulator"),qpu_parallelization='auto' )


if KERNEL_NAME == 'fidelity_kernel':
    q_kernel = FidelityKernel(
        encoding_circuit=enc_circ, 
        executor=executor, 
        #executor=executor, # to compare with Qiskit we need to downgrade to 1.0
        parameter_seed=SEED,
        evaluate_duplicates='off_diagonal',
        caching=True
    )
else:
    # Use median heuristic gamma instead of hardcoded value
    temp_K = rbf_kernel(X_scaled[:min(500, len(X_scaled))], gamma=1.0)  # temp kernel for heuristic
    temp_dists = 1.0 - temp_K
    median_dist = np.median(temp_dists[temp_dists > 0])
    initial_gamma = 1.0 / (2.0 * (median_dist ** 2)) if median_dist > 0 else 50.0
    
    q_kernel = ProjectedQuantumKernel(
    encoding_circuit=enc_circ,
    #executor=executor,
    executor = executor,
    measurement="XYZ",
    #measurement="YZX",
    #measurement=obs,
    outer_kernel="gaussian",
    initial_parameters=2 * np.pi * np.random.rand(enc_circ.num_parameters),
    gamma = gamma_param ,  # Median heuristic instead of hardcoded 50
    #gamma= 1.0 / (3.0 * enc_circ.num_qubits),
)

# Optimization cell with optional Gram-matrix caching and JAX linear-algebra toggles
# Toggle knobs (set to True to enable, defaults preserve existing behaviour)
REUSE_KERNEL_CACHE = True            # cache exact kernel.evaluate results keyed by parameter vector
APPROXIMATE_REUSE = False            # allow approximate reuse for small-perturbations (risky; use only if you understand the approximation)
APPROX_TOL = 1e-6                    # L2 tolerance for approximate reuse when APPROXIMATE_REUSE=True
USE_JAX_LINEARALG = True             # prefer JAX-backed eig/decomp paths when available (DEMaPLoss/process_kernels)

# Lightweight kernel-evaluate caching proxy that wraps an existing quantum-kernel object.
# It implements assign_parameters(params) and evaluate(X) to mirror the expected interface.
import numpy as _np

def _kernel_to_distance(K: np.ndarray) -> np.ndarray:
    K = 0.5*(K + K.T)
    K[~np.isfinite(K)] = 0.0
    diag = np.diag(K).copy()
    diag[diag <= 0] = 1.0
    K_norm = K / np.sqrt(np.outer(diag, diag))
    np.fill_diagonal(K_norm, 1.0)
    dist = 1.0 - K_norm
    # jitter if needed:
    if np.nanstd(dist) < 1e-10:
        eps = 1e-8
        jitter = np.random.RandomState(SEED).normal(scale=eps, size=dist.shape)
        jitter = 0.5*(jitter + jitter.T); np.fill_diagonal(jitter, 0.0)
        dist = np.clip(dist + jitter, 0.0, None)
    return dist


def _median_heuristic_gamma(x: np.ndarray):
    """Median heuristic for RBF gamma for 1D arrays (e.g., pairwise distances flattened).
    Adds eps to avoid zero gamma.
    """
    x = np.asarray(x).reshape(-1, 1)
    if len(x) < 2:
        return 1.0
    dists = pdist(x, metric="euclidean")
    med = np.median(dists)
    eps = 1e-8
    return 1.0 / (2.0 * (max(med, eps) ** 2))

class KernelCacheProxy:
    def __init__(self, kernel, cache_exact=True, cache_approx=False, approx_tol=1e-6):
        self._kernel = kernel
        self.cache_exact = bool(cache_exact)
        self.cache_approx = bool(cache_approx)
        self.approx_tol = float(approx_tol)
        # cache keyed by (params_key) -> K (host numpy array)
        self._last_params = None
        self._last_K = None
        # lightweight input signature for cached K (shape + small checksum)
        self._last_input_signature = None

    def assign_parameters(self, params):
        # keep a copy of the params used to produce the cached K (host numpy)
        try:
            p = _np.asarray(params).ravel().astype(float)
        except Exception:
            p = params
        self._last_assign = p
        # forward to underlying kernel; many kernel implementations store params internally
        try:
            self._kernel.assign_parameters(params)
        except Exception:
            # some kernels accept set_parameters or similar; try best-effort
            if hasattr(self._kernel, 'set_parameters'):
                try:
                    self._kernel.set_parameters(params)
                except Exception:
                    pass

    def evaluate(self, X):
        # compute a lightweight signature of the input to avoid returning a K
        # computed on a different subsample/shape (source of rare broadcasting bugs)
        try:
            Xa = _np.asarray(X)
            cols = min(3, Xa.shape[1]) if Xa.ndim > 1 else 1
            input_sig = (Xa.shape, float(_np.sum(Xa[:, :cols])) if Xa.size > 0 else 0.0)
        except Exception:
            input_sig = None

        # if cache enabled and params identical to last cached, return cached K
        try:
            cur_p = _np.asarray(getattr(self, '_last_assign', None))
        except Exception:
            cur_p = None

        if self.cache_exact and (self._last_K is not None) and cur_p is not None and (self._last_params is not None) and _np.array_equal(cur_p, self._last_params):
            # ensure that cached K was produced for the same input signature
            if (self._last_input_signature is None) or (input_sig is None) or (self._last_input_signature == input_sig):
                print(f"[KernelCacheProxy] Exact cache HIT: params match. Returning cached K shape={getattr(self._last_K, 'shape', None)}")
                return self._last_K
            else:
                print(f"[KernelCacheProxy] Cache SKIP: params match but input signature differs. cached_sig={self._last_input_signature} req_sig={input_sig}")

        # approximate reuse path (use with caution)
        if self.cache_approx and (self._last_K is not None) and cur_p is not None and (self._last_params is not None):
            diff = cur_p - self._last_params
            if _np.linalg.norm(diff) <= float(self.approx_tol):
                # reuse last K (treat perturbation as negligible)
                return self._last_K

        # fallback: evaluate underlying kernel and cache result
        K = self._kernel.evaluate(X)
        try:
            K = _np.asarray(K)
        except Exception:
            pass
        # store cache copy (host), param snapshot and input signature
        try:
            self._last_K = _np.array(K, copy=True)
            self._last_params = _np.asarray(getattr(self, '_last_assign', None)).ravel().astype(float) if getattr(self, '_last_assign', None) is not None else None
            try:
                self._last_input_signature = input_sig
            except Exception:
                self._last_input_signature = None
            print(f"[KernelCacheProxy] Cached new K shape={getattr(self._last_K, 'shape', None)} input_sig={self._last_input_signature}")
        except Exception:
            self._last_K = K
            self._last_input_signature = input_sig
        return K


class MMDPairwiseDistanceLoss(KernelLossBase):
    """Compute MMD^2 between the distributions of pairwise distances (original vs kernel-derived).

    This implementation keeps the original sampled-pair machinery but adds several
    performance-oriented options:
      - embedding_dim: compute a low-rank embedding of kernel rows (randomized SVD) and
                       compute pairwise distances in that low-dim space (reduces memory).
      - use_linear_mmd: use a linear-time MMD estimator on 1D distance vectors (O(s) vs O(s^2)).
      - gamma_median_subsample: compute the median heuristic on a subsample to avoid O(s^2) pdist.

    The class also caches the last-processed parameter vector -> embedding/Kd to avoid repeated
    post-processing when finite-differences call compute() with identical parameters. The notebook
    includes a KernelCacheProxy to avoid re-running expensive quantum kernel.evaluate for identical params;
    this class's internal cache complements that by avoiding K->Kd/embedding recompute.
    """

    def __init__(self, subsample: Optional[int] = None, random_state: Optional[int] = None, kernel_scale: Optional[float] = None, pair_subsample: Optional[int] = 5000, embedding_dim: Optional[int] = None, use_linear_mmd: bool = True, gamma_median_subsample: int = 1000, use_torch: bool = True, torch_device: Optional[torch.device] = None):
        super().__init__()
        self.subsample = subsample
        self.random_state = random_state
        self.kernel_scale = kernel_scale
        self.pair_subsample = pair_subsample
        # new options
        self.embedding_dim = embedding_dim  # None or int (e.g. 16,32)
        self.use_linear_mmd = bool(use_linear_mmd)
        self.gamma_median_subsample = int(gamma_median_subsample)
        self.use_torch = bool(use_torch)
        self.torch_device = torch_device or device
        # caches: last parameter snapshot -> last embedding_or_Kd
        self._cached_params = None
        self._cached_embedding_or_Kd = None

    def _rbf_matrix(self, a: np.ndarray, b: np.ndarray, gamma: float):
        # a,b shape (n,1) or (n,) cast to column/row as needed
        a = np.asarray(a).reshape(-1, 1)
        b = np.asarray(b).reshape(-1, 1)
        diff = a - b.T
        return np.exp(-gamma * (diff ** 2))

    def _mmd2_from_vectors(self, x: np.ndarray, y: np.ndarray, gamma: float):
        # fallback quadratic unbiased estimator used for small sample sizes or when linear estimator fails
        x = x.reshape(-1, 1)
        y = y.reshape(-1, 1)
        Kxx = self._rbf_matrix(x, x, gamma)
        Kyy = self._rbf_matrix(y, y, gamma)
        Kxy = self._rbf_matrix(x, y, gamma)
        n = x.shape[0]
        m = y.shape[0]
        sum_kxx = (np.sum(Kxx) - np.trace(Kxx)) / (n * (n - 1)) if n > 1 else 0.0
        sum_kyy = (np.sum(Kyy) - np.trace(Kyy)) / (m * (m - 1)) if m > 1 else 0.0
        sum_kxy = np.sum(Kxy) / (n * m) if (n > 0 and m > 0) else 0.0
        return float(sum_kxx + sum_kyy - 2.0 * sum_kxy)

    def _median_heuristic_gamma_subsample(self, x: np.ndarray, max_samples: int = None):
        # median heuristic but computed on a subsample to avoid O(s^2) pdist on large s
        if max_samples is None:
            max_samples = max(256, int(self.gamma_median_subsample))
        x = np.asarray(x).reshape(-1, 1)
        s = x.shape[0]
        if s < 2:
            return 1.0
        if s <= max_samples:
            d = pdist(x, metric="euclidean")
            med = np.median(d) if d.size > 0 else 1.0
        else:
            rng = np.random.RandomState(self.random_state)
            idx = rng.choice(s, size=max_samples, replace=False)
            ds = pdist(x[idx], metric="euclidean")
            med = np.median(ds) if ds.size > 0 else 1.0
        med = max(float(med), 1e-12)
        return 1.0 / (2.0 * (med ** 2))

    def _linear_mmd_estimator(self, x: np.ndarray, y: np.ndarray, gamma: float, repeats: int = 3):
        # simple repeated random-pair linear-time estimator. For stability we average several repeats.
        rng = np.random.RandomState(self.random_state)
        nx, ny = int(x.shape[0]), int(y.shape[0])
        s = min(nx, ny)
        if s < 4:
            return None
        vals = []
        for _ in range(max(1, int(repeats))):
            # sample without replacement up to s elements
            idx_x = rng.choice(nx, size=s, replace=False) if nx > s else np.arange(nx)
            idx_y = rng.choice(ny, size=s, replace=False) if ny > s else np.arange(ny)
            xs = np.asarray(x)[idx_x].ravel()
            ys = np.asarray(y)[idx_y].ravel()
            order = rng.permutation(s)
            xs = xs[order]
            ys = ys[order]
            s_pairs = (s // 2) * 2
            xa = xs[:s_pairs:2]; xb = xs[1:s_pairs:2]
            ya = ys[:s_pairs:2]; yb = ys[1:s_pairs:2]
            k_xx = np.exp(-gamma * ((xa - xb) ** 2))
            k_yy = np.exp(-gamma * ((ya - yb) ** 2))
            k_xy = np.exp(-gamma * ((xa - yb) ** 2))
            k_yx = np.exp(-gamma * ((xb - ya) ** 2))
            vals.append(np.mean(k_xx) + np.mean(k_yy) - np.mean(k_xy) - np.mean(k_yx))
        return float(np.mean(vals)) if len(vals) > 0 else None

    def _kernel_to_embedding(self, K: np.ndarray, r: int = 32):
        # produce a low-rank embedding of kernel-derived rows using classical MDS on distance-of-K rows
        K = np.asarray(K)
        # reuse safe normalization from notebook-level helper
        try:
            Kd = _kernel_to_distance(K)  # (n,n) distance-like
        except Exception:
            # fallback: naive conversion
            Kd = 1.0 - (0.5 * (K + K.T))
        n = Kd.shape[0]
        if n == 0:
            return np.zeros((0, min(r, n)))
        D2 = np.asarray(Kd, dtype=float) ** 2
        J = np.eye(n) - np.ones((n, n)) / float(n)
        B = -0.5 * J.dot(D2).dot(J)
        # truncated randomized SVD on B to obtain top-r eigenspace (U, S)
        try:
            from sklearn.utils.extmath import randomized_svd
            r_eff = min(int(r), n)
            U, S, Vt = randomized_svd(B, n_components=r_eff, n_iter=5, random_state=self.random_state)
            emb = U * np.sqrt(np.maximum(S, 0.0))[np.newaxis, :]
            if emb.shape[1] < int(r):
                pad = np.zeros((n, int(r) - emb.shape[1]))
                emb = np.hstack([emb, pad])
            return emb
        except Exception:
            # fallback: use rows of K normalized
            diag = np.diag(K).copy()
            diag[diag <= 0] = 1.0
            K_norm = K / np.sqrt(np.outer(diag, diag))
            np.fill_diagonal(K_norm, 1.0)
            return K_norm[:, :min(K_norm.shape[1], int(r))]

    def compute(self, parameter_values: np.ndarray, data: np.ndarray, labels: np.ndarray = None) -> float:
        if self._quantum_kernel is None:
            raise ValueError("Quantum kernel not set. Call set_quantum_kernel(...) before using this loss.")

        # bind params
        self._quantum_kernel.assign_parameters(parameter_values)

        n_total = data.shape[0]
        if self.subsample and 0 < self.subsample < n_total:
            rng = np.random.RandomState(self.random_state)
            idx = rng.choice(n_total, size=self.subsample, replace=False)
            data_sub = data[idx]
        else:
            data_sub = data

        n = data_sub.shape[0]
        if n < 2:
            return 0.0

        rng = np.random.RandomState(self.random_state)
        total_pairs = n * (n - 1) // 2
        max_pairs = self.pair_subsample if self.pair_subsample is not None else total_pairs
        use_pairs = int(min(total_pairs, max_pairs))

        if use_pairs >= total_pairs:
            full_pairs_mode = True
            orig_pair = pdist(data_sub, metric="euclidean")
            i = j = None
        else:
            full_pairs_mode = False
            i = rng.randint(0, n, size=use_pairs)
            j = rng.randint(0, n, size=use_pairs)
            neq = i != j
            i = i[neq]; j = j[neq]
            if np.any(i > j):
                swap = i > j
                ti = i[swap].copy(); i[swap] = j[swap]; j[swap] = ti
            while i.shape[0] < use_pairs:
                extra_i = rng.randint(0, n, size=(use_pairs - i.shape[0]))
                extra_j = rng.randint(0, n, size=(use_pairs - j.shape[0]))
                mask = extra_i != extra_j
                extra_i = extra_i[mask]; extra_j = extra_j[mask]
                if extra_i.shape[0] == 0:
                    continue
                i = np.concatenate([i, extra_i[:use_pairs - i.shape[0]]])
                j = np.concatenate([j, extra_j[:use_pairs - j.shape[0]]])
            orig_pair = None
            orig_pair_sample = np.linalg.norm(data_sub[i] - data_sub[j], axis=1)

        # attempt to reuse cached embedding/Kd when parameters unchanged (fast)
        p_snapshot = np.asarray(parameter_values).ravel().astype(float)
        if (self._cached_params is not None) and np.array_equal(self._cached_params, p_snapshot) and (self._cached_embedding_or_Kd is not None):
            emb_or_Kd = self._cached_embedding_or_Kd
        else:
            # evaluate kernel (expensive) and postprocess once
            try:
                K = self._quantum_kernel.evaluate(data_sub)
            except Exception as e:
                print(f"MMDPairwiseDistanceLoss: kernel.evaluate raised: {e}")
                return 1.0
            if K is None:
                return 1.0
            K = np.asarray(K)
            if K.size == 0:
                return 1.0
            if self.embedding_dim is not None and self.embedding_dim > 0:
                try:
                    emb_or_Kd = self._kernel_to_embedding(K, r=self.embedding_dim)
                except Exception:
                    emb_or_Kd = _kernel_to_distance(K)
            else:
                emb_or_Kd = _kernel_to_distance(K)
            # cache snapshot
            self._cached_params = p_snapshot
            self._cached_embedding_or_Kd = emb_or_Kd

        # compute kernel-derived pair distance(s)
        if full_pairs_mode:
            if emb_or_Kd.ndim == 2 and emb_or_Kd.shape[1] < emb_or_Kd.shape[0]:
                kernel_dist = pdist(emb_or_Kd, metric="euclidean")
            else:
                # emb_or_Kd is an (n,n) distance matrix; pdist treats rows as vectors
                kernel_dist = pdist(emb_or_Kd, metric="euclidean")
            # compute gamma using subsampled median heuristic
            if self.kernel_scale is not None:
                gamma = self.kernel_scale
            else:
                gamma = self._median_heuristic_gamma_subsample(np.concatenate([orig_pair, kernel_dist]) if orig_pair is not None else kernel_dist, max_samples=self.gamma_median_subsample)
            if self.use_linear_mmd:
                mmd2 = self._linear_mmd_estimator(orig_pair, kernel_dist, gamma)
                if mmd2 is None:
                    mmd2 = self._mmd2_from_vectors(orig_pair, kernel_dist, gamma)
            else:
                mmd2 = self._mmd2_from_vectors(orig_pair, kernel_dist, gamma)
            return float(mmd2)
        else:
            # sampled pairs: compute only needed pair distances (cheap when embedding_dim provided)
            if self.use_torch:
                Et = _to_tensor(emb_or_Kd, dtype=torch.float32, dev=self.torch_device)
                it = torch.as_tensor(i, device=Et.device)
                jt = torch.as_tensor(j, device=Et.device)
                diffs = Et[it] - Et[jt]
                kernel_pair_sample = torch.linalg.norm(diffs, dim=1).detach().cpu().numpy()
            else:
                if emb_or_Kd.ndim == 2 and emb_or_Kd.shape[1] < emb_or_Kd.shape[0]:
                    kernel_pair_sample = np.linalg.norm(emb_or_Kd[i] - emb_or_Kd[j], axis=1)
                else:
                    kernel_pair_sample = np.linalg.norm(emb_or_Kd[i] - emb_or_Kd[j], axis=1)
            if self.kernel_scale is not None:
                gamma = self.kernel_scale
            else:
                gamma = self._median_heuristic_gamma_subsample(np.concatenate([orig_pair_sample, kernel_pair_sample]) if (orig_pair_sample is not None) else kernel_pair_sample, max_samples=self.gamma_median_subsample)
            if self.use_linear_mmd:
                mmd2 = self._linear_mmd_estimator(orig_pair_sample, kernel_pair_sample, gamma)
                if mmd2 is None:
                    mmd2 = self._mmd2_from_vectors(orig_pair_sample, kernel_pair_sample, gamma)
            else:
                mmd2 = self._mmd2_from_vectors(orig_pair_sample, kernel_pair_sample, gamma)
            return float(mmd2)

# Example file: src/squlearn/kernel/loss/phate_rank_preservation.py
from typing import Optional
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
#from phate import PHATE

from squlearn.kernel.loss.kernel_loss_base import KernelLossBase  # your existing base class

from sklearn.manifold import MDS
from sklearn.utils.extmath import randomized_svd
from scipy.sparse.linalg import eigsh

try:
    import graphtools
    _HAS_GRAPHTOOLS = True
except Exception:
    _HAS_GRAPHTOOLS = False

# Optional JAX import — we'll use it only when requested via use_jax=True
try:
    import jax
    import jax.numpy as jnp
    _HAS_JAX = True
except Exception:
    jax = None
    jnp = None
    _HAS_JAX = False


def geodesic_distance(data, knn=30, distance="data"):
    """
    Return shortest-path (geodesic) distance matrix for rows in `data`.
    Uses graphtools.Graph when available; otherwise falls back to sklearn
    NearestNeighbors + scipy shortest_path.
    """
    if _HAS_GRAPHTOOLS:
        G = graphtools.Graph(data, knn=knn, decay=None)
        return G.shortest_path(distance=distance)
    else:
        # fallback
        from sklearn.neighbors import NearestNeighbors
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import shortest_path
        from squlearn.kernel.loss.target_alignment import TargetAlignment
        from sklearn.manifold import MDS
        nbrs = NearestNeighbors(n_neighbors=min(knn, data.shape[0]-1)).fit(data)
        A = nbrs.kneighbors_graph(mode="distance")
        A = 0.5 * (A + A.T)
        return shortest_path(csgraph=csr_matrix(A), directed=False)

import numpy as np
import time
from typing import Optional

# Assuming KernelLossBase is defined elsewhere in your pipeline
# class KernelLossBase: ... 
import numpy as np
import torch
from typing import Optional
from squlearn.kernel.loss.kernel_loss_base import KernelLossBase

from squlearn.optimizers.optimizer_base import OptimizerBase, OptimizerResult, IterativeMixin

# -------- Tensor utilities for large-matrix losses --------
def _to_tensor(x, dtype=torch.float32, dev=None):
    dev = dev or device
    if isinstance(x, torch.Tensor):
        return x.to(dev)
    return torch.as_tensor(x, dtype=dtype, device=dev)


def evaluate_kernel_torch(kernel_obj, X, device=None, dtype=torch.float32):
    """Try to obtain the kernel Gram matrix directly as a torch.Tensor on `device`.

    Preferred behavior: call `kernel_obj.evaluate_torch(...)` if implemented by the
    kernel or a proxy. Otherwise, call `kernel_obj.evaluate(...)` and convert the
    returned numpy array to a torch tensor on the requested device.
    """
    try:
        import torch
    except Exception:
        raise RuntimeError("torch is required for evaluate_kernel_torch but is not available")

    dev = device or torch.device('cpu')
    # Prefer native torch evaluation if the kernel supports it
    try:
        if hasattr(kernel_obj, 'evaluate_torch'):
            return kernel_obj.evaluate_torch(X, device=dev, dtype=dtype)
    except Exception:
        # Fall back to numpy evaluate on any failure
        pass

    # Fallback: evaluate as numpy and convert
    K_np = kernel_obj.evaluate(X)
    return torch.as_tensor(K_np, dtype=dtype, device=dev)

def _spearman_corr_torch(x: torch.Tensor, y: torch.Tensor) -> float:
    """Compute Spearman correlation via rank transform on torch tensors.
    Uses argsort twice to get ranks; returns float on CPU.
    """
    # ranks for x
    x_argsort = torch.argsort(x)
    x_ranks = torch.empty_like(x_argsort, dtype=torch.float32)
    x_ranks[x_argsort] = torch.arange(x.numel(), dtype=torch.float32, device=x.device)
    # ranks for y
    y_argsort = torch.argsort(y)
    y_ranks = torch.empty_like(y_argsort, dtype=torch.float32)
    y_ranks[y_argsort] = torch.arange(y.numel(), dtype=torch.float32, device=y.device)
    # Pearson on ranks
    xr = (x_ranks - x_ranks.mean())
    yr = (y_ranks - y_ranks.mean())
    denom = (xr.norm() * yr.norm())
    if denom.item() == 0.0:
        return 0.0
    corr = (xr * yr).sum() / denom
    return float(torch.clamp(corr, -1.0, 1.0).detach().cpu())

def _pairwise_euclid(x: torch.Tensor) -> torch.Tensor:
    return torch.cdist(x, x, p=2)

def _median_heuristic_torch(vals: torch.Tensor, max_samples: int = 1000) -> float:
    v = vals.reshape(-1)
    if v.numel() == 0:
        return 1.0
    if v.numel() > max_samples:
        idx = torch.randperm(v.numel(), device=v.device)[:max_samples]
        v = v[idx]
    med = torch.median(torch.abs(v))
    med = float(max(med.item(), 1e-12))
    return 1.0 / (2.0 * (med ** 2))

class TorchAdamOptimizer(OptimizerBase, IterativeMixin):
    """Torch-backed optimizer that supports multiple gradient-estimation strategies.

    Improvements over the original implementation:
    - Added SPSA gradient estimator (O(1) evals per iteration) for large parameter vectors.
    - Preserves original central-finite-difference (FD) path for small problems or when
      exact FD is desired.
    - Uses analytic `grad` when provided.
    - Auto-selects SPSA when parameter count exceeds `spsa_auto_threshold`.
    """
    def __init__(self, maxiter=50, lr=1e-3, eps=1e-8, central_eps=1e-4, verbose=False, device=None, n_jobs: int = None, grad_clip: float = 1.0, grad_method: str = 'auto', spsa_samples: int = 1, spsa_eps: float = None, spsa_eps_end: float = 0.01, spsa_auto_threshold: int = 50, spsa_batch_size: Optional[int] = None, dataset_size: Optional[int] = None, random_state: Optional[int] = None):
        IterativeMixin.__init__(self)
        self.maxiter = int(maxiter)
        self.lr = float(lr)
        self.eps = float(eps)
        self.central_eps = float(central_eps)
        self.verbose = bool(verbose)
        self.device = device or torch.device('cpu')
        # keep last history on the instance for external inspection
        self.history = None
        # parallel workers for finite-difference evaluations (None -> auto).
        # NOTE: PennyLane's queuing is not thread-safe. To avoid
        # queuing / tape-corruption issues when finite-differences call
        # QNodes concurrently, default to single-worker. If you need
        # parallel FD for non-PennyLane backends, set n_jobs>1 explicitly.
        if n_jobs is None:
            self.n_jobs = 1
        else:
            self.n_jobs = int(n_jobs)
        # gradient clipping value (L2 norm). Set to None to disable.
        self.grad_clip = float(grad_clip) if grad_clip is not None else None

        # SPSA / gradient options
        self.grad_method = str(grad_method).lower()  # 'auto' | 'fd' | 'spsa' | 'analytic'
        self.spsa_samples = int(max(1, spsa_samples))
        # SPSA eps schedule (start -> end). By default start uses provided spsa_eps or central_eps.
        self.spsa_eps = float(spsa_eps) if spsa_eps is not None else float(self.central_eps)
        self.spsa_eps_end = float(spsa_eps_end)
        # minibatch for SPSA: if set, sample this many landmark indices per SPSA evaluation
        self.spsa_batch_size = int(spsa_batch_size) if spsa_batch_size is not None else None
        # if using minibatching, dataset_size must be provided so we can sample indices
        self.dataset_size = int(dataset_size) if dataset_size is not None else None
        self.spsa_auto_threshold = int(spsa_auto_threshold)
        self.random_state = random_state

        # deterministic rng for SPSA sampling
        self._rng = np.random.RandomState(self.random_state)

    def _fd_component(self, i: int, base_p: np.ndarray, eps: float, fun: callable):
        """Helper to compute central-difference for a single index. Returns (i, grad_i)."""
        p = base_p.copy()
        orig = p[i]
        p[i] = orig + eps
        f_plus = float(fun(p))
        p[i] = orig - eps
        f_minus = float(fun(p))
        grad_i = (f_plus - f_minus) / (2.0 * eps)
        return i, grad_i

    def _spsa_gradient(self, base_p: np.ndarray, eps: float, fun: callable, repeats: int = 1) -> np.ndarray:
        """Estimate gradient via SPSA: only 2 * repeats evaluations regardless of param count.

        Returns a numpy array of same shape as base_p containing the estimated gradient.
        """
        n = base_p.size
        grad_est = np.zeros(n, dtype=float)

        # Use a reproducible RNG that we advance each call
        rng = self._rng

        for rep in range(repeats):
            # ±1 Rademacher perturbations
            delta = rng.choice([-1.0, 1.0], size=n)

            p_plus = base_p + eps * delta
            p_minus = base_p - eps * delta

            # If minibatching is requested and dataset_size is provided, sample indices
            batch_kwargs = {}
            if (self.spsa_batch_size is not None) and (self.dataset_size is not None):
                m = min(self.spsa_batch_size, max(1, self.dataset_size))
                idx = rng.choice(self.dataset_size, size=m, replace=False)
                # try common kwarg names; we'll let fun raise TypeError if not supported
                batch_kwargs = {"batch": idx}

            try:
                f_plus = float(fun(p_plus, **batch_kwargs)) if batch_kwargs else float(fun(p_plus))
                f_minus = float(fun(p_minus, **batch_kwargs)) if batch_kwargs else float(fun(p_minus))
            except TypeError:
                # fall back to calling without batch kwarg
                f_plus = float(fun(p_plus))
                f_minus = float(fun(p_minus))

            # gradient estimate for this repeat
            # note: (f_plus - f_minus) / (2 eps) * delta
            g = (f_plus - f_minus) / (2.0 * eps) * delta
            grad_est += g

        grad_est /= float(repeats)
        return grad_est

    def minimize(self, fun: callable, x0: np.ndarray, grad: callable = None, bounds=None) -> OptimizerResult:
        # fun: callable(param_vector: np.ndarray) -> float (numpy)
        x0 = np.asarray(x0, dtype=float).ravel()
        n = x0.size
        res = OptimizerResult()
        if n == 0:
            res.x = x0
            res.nit = 0
            res.fun = float(fun(x0)) if callable(fun) else 0.0
            res.history = [res.fun]
            self.history = res.history
            return res

        # create torch parameter on device
        params_t = torch.nn.Parameter(torch.tensor(x0, dtype=torch.float32, device=self.device))
        opt = torch.optim.Adam([params_t], lr=self.lr, betas=(0.9, 0.999), eps=self.eps)

        best_x = x0.copy()
        try:
            best_f = float(fun(best_x))
        except Exception:
            raise

        # initialize history with initial evaluation
        history = [float(best_f)]

        # # auto-select gradient method if requested
        # chosen_method = self.grad_method
        # if chosen_method == 'auto':
        #     chosen_method = 'spsa' if n >= self.spsa_auto_threshold else 'fd'

        chosen_method = 'spsa'
        for it in range(self.maxiter):
            p_cpu = params_t.detach().cpu().numpy().astype(float)

            # 1) Analytic gradient path (user provided)
            if callable(grad):
                grad_np = np.asarray(grad(p_cpu)).astype(float).ravel()

            # 2) SPSA path (very cheap: 2*repeats evaluations per iter)
            elif chosen_method == 'spsa':
                # compute current SPSA eps with linear decay from start->end over maxiter
                start = float(self.spsa_eps)
                end = float(self.spsa_eps_end)
                steps = max(1, int(self.maxiter))
                t = float(min(it, steps)) / float(steps)
                cur_eps = float(start + (end - start) * t)
                grad_np = self._spsa_gradient(p_cpu, eps=cur_eps, fun=fun, repeats=self.spsa_samples)

            # 3) Finite-difference path (original behaviour)
            else:
                grad_np = np.zeros_like(p_cpu)
                try:
                    workers = max(1, min(self.n_jobs, n))
                    results = Parallel(n_jobs=workers, prefer='threads')(
                        delayed(self._fd_component)(i, p_cpu, self.central_eps, fun) for i in range(n)
                    )
                    for i, g in results:
                        grad_np[i] = g
                except Exception:
                    for i in range(n):
                        _, g = self._fd_component(i, p_cpu, self.central_eps, fun)
                        grad_np[i] = g

            # assign gradient to torch param
            grad_t = torch.tensor(grad_np.astype(np.float32), device=self.device)
            # sanitize NaN/inf to avoid blowups
            grad_t = torch.nan_to_num(grad_t, nan=0.0, posinf=1e8, neginf=-1e8)
            params_t.grad = grad_t

            # gradient clipping using torch utility (works on Parameter .grad fields)
            if self.grad_clip is not None:
                try:
                    torch.nn.utils.clip_grad_norm_([params_t], max_norm=float(self.grad_clip))
                except Exception:
                    # fallback: no-op if clipping fails for any reason
                    pass

            # perform optimizer step
            opt.step()
            opt.zero_grad()

            # evaluate the objective at the updated params
            cur_x = params_t.detach().cpu().numpy()
            cur_f = float(fun(cur_x))
            history.append(float(cur_f))
            if cur_f < best_f:
                best_f = cur_f
                best_x = cur_x.copy()
            self.iteration += 1
            if self.verbose:
                print(f'iter={it+1}/{self.maxiter}  loss={cur_f:.6g}  method={chosen_method}')

        res.x = best_x
        res.nit = int(self.iteration)
        res.fun = float(best_f)
        # attach history to result and instance
        res.history = history
        self.history = history
        return res


class COBYLAOptimizer(OptimizerBase, IterativeMixin):
    """Wrapper around scipy.optimize.minimize with COBYLA method.
    
    COBYLA (Constrained Optimization BY Linear Approximation) is a gradient-free method
    that works well on noisy objective functions and requires no derivative information.
    """
    def __init__(self, maxiter=50, lr=None, rhobeg=1.0, tol=1e-6, verbose=False, device=None, random_state=None):
        IterativeMixin.__init__(self)
        self.maxiter = int(maxiter)
        self.rhobeg = float(rhobeg)  # Initial step size for COBYLA
        self.tol = float(tol)
        self.verbose = bool(verbose)
        self.device = device or torch.device('cpu')
        self.history = None
        self.random_state = random_state
        self._rng = np.random.RandomState(self.random_state)

    def minimize(self, fun: callable, x0: np.ndarray, grad: callable = None, bounds=None) -> OptimizerResult:
        """Minimize using scipy COBYLA method."""
        x0 = np.asarray(x0, dtype=float).ravel()
        n = x0.size
        res = OptimizerResult()
        
        if n == 0:
            res.x = x0
            res.nit = 0
            res.fun = float(fun(x0)) if callable(fun) else 0.0
            res.history = [res.fun]
            self.history = res.history
            return res

        # Create wrapper to track evaluations
        eval_count = {'count': 0}
        history = []
        best_x = x0.copy()
        best_f = None
        
        def fun_wrapper(x):
            try:
                f = float(fun(x))
                eval_count['count'] += 1
                if best_f is None or f < best_f:
                    history.append(f)
                    if f < (best_f or float('inf')):
                        best_x[:] = x
                        best_f_arr[0] = f
                else:
                    history.append(best_f_arr[0])
                if self.verbose and eval_count['count'] % 10 == 0:
                    print(f"  COBYLA iter ~{eval_count['count']}: loss={f:.6g}")
                return f
            except Exception as e:
                if self.verbose:
                    print(f"Evaluation failed: {e}")
                return 1e10

        best_f_arr = np.array([float(fun(x0))])
        history.append(best_f_arr[0])
        best_x[:] = x0

        # Run COBYLA
        opt_res = scipy_minimize(
            fun_wrapper,
            x0,
            method='COBYLA',
            options={
                'maxiter': self.maxiter,
                'rhobeg': self.rhobeg,
                'tol': self.tol,
                'disp': self.verbose
            }
        )

        res.x = opt_res.x
        res.nit = opt_res.nit if hasattr(opt_res, 'nit') else eval_count['count']
        res.fun = best_f_arr[0]
        res.history = history
        self.history = history
        self.iteration = res.nit
        
        return res


# Usage notes:
# - To use this with sQUlearn KernelOptimizer the loss must be implemented through
#   a differentiable backend (PennyLane with interface='torch' or similar).
# - Numpy-based loss functions are not differentiable for autograd; they must be
#   ported to torch or wrapped by a differentiable QNode.
#loss = LOSS

if LOSS == "MMDPairwiseDistanceLoss":

    # 1) Wrap kernel with proxy to cache exact evaluations (helps FD & repeated calls)
    proxy = KernelCacheProxy(q_kernel, cache_exact=True, cache_approx=False, approx_tol=1e-6)
    # If you already set kernel into KernelOptimizer, instead set loss quantum kernel with proxy:
    # loss.set_quantum_kernel(proxy)

    # 2) Instantiate MMD loss (use full dataset for rows; keep pair subsampling for speed)
    loss = MMDPairwiseDistanceLoss(
        subsample=None,
        random_state=SEED,
        kernel_scale=None,         # None -> median heuristic (subsampled)
        pair_subsample=1000,       # when < total pairs we compute only sampled pairs
        embedding_dim=32,          # low-rank embedding dimension (set None to skip)
        use_linear_mmd=True,       # use O(s) linear-time MMD estimator
        gamma_median_subsample=500
    )

    # attach the proxy kernel to the loss
    loss.set_quantum_kernel(proxy)

elif LOSS == "KTALoss":

    # 1) Wrap kernel with proxy to cache exact evaluations (helps FD & repeated calls)
    proxy = KernelCacheProxy(q_kernel, cache_exact=True, cache_approx=False, approx_tol=1e-6)
    # If you already set kernel into KernelOptimizer, instead set loss quantum kernel with proxy:
    # loss.set_quantum_kernel(proxy)

    # 2) Instantiate KTA loss (use full dataset for rows)
    loss = KTALoss(
        subsample=None,
        random_state=SEED,
        rescale_class_labels=True
    )

    try:
        loss.set_quantum_kernel(proxy)
    except Exception:
        loss._quantum_kernel = proxy

    qk_for_optimizer = proxy
elif LOSS =="QuantumVonNeumannRatioLoss":
    loss = ApproxQuantumVonNeumannRatioLoss(
        subsample=None,
        random_state=SEED,
        use_torch=True
    )
else:
    # Instead of PHATERankPreservationLoss, use DeMAP
    # Use full dataset for DEMaP as well (no row subsampling)
    loss = DEMaPLoss(subsample=None, random_state=SEED)
    loss.set_quantum_kernel(q_kernel)   # KernelLossBase usually provides something like this

# Iterative layer-by-layer optimization
MAX_LAYERS = MAX_LAYERS_ARG  # Maximum number of layers to add
all_results = []  # Store results for each layer
best_overall_loss = float('inf')
best_overall_params = None
best_overall_layer = None
best_overall_history = None  # Track history of best run

# Start timing the optimization
optimization_start_time = time.time()
print(f"\n{'='*80}")
print(f"OPTIMIZATION STARTED AT: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"{'='*80}\n")

# ======================== FIXED GAMMA MODE ========================
# If GAMMA_FIXED is set, skip optimization and use fixed gamma value directly
if GAMMA_FIXED is not None:
    print(f"\n{'='*80}")
    print(f"FIXED GAMMA MODE: Using gamma={GAMMA_FIXED} WITHOUT OPTIMIZATION")
    print(f"{'='*80}\n")
    
    # Use the starting layer count (NO_LAYERS)
    enc_circ = ChebyshevPQC(N_QUBITS, num_layers=NO_LAYERS,
                              entangling_gate = 'rzz',
                            num_features=X_scaled.shape[1],
                            alpha=alpha_param)
    
    print(f"Creating circuit: {N_QUBITS} qubits, {NO_LAYERS} layers, {X_scaled.shape[1]} features")
    print(f"Circuit has {enc_circ.num_parameters} parameters")
    
    if KERNEL_NAME == 'fidelity_kernel':
        q_kernel = FidelityKernel(
            encoding_circuit=enc_circ,
            executor=executor,
            parameter_seed=SEED,
            evaluate_duplicates='off_diagonal',
            caching=True
        )
    else:
        # Create kernel with FIXED gamma
        q_kernel = ProjectedQuantumKernel(
            encoding_circuit=enc_circ,
            executor=executor,
            measurement="XYZ",
            outer_kernel="gaussian",
            initial_parameters=np.random.normal(0, 0.05, enc_circ.num_parameters),
            gamma=gamma_param  # Use fixed gamma without optimization
        )
    
    # Initialize with random parameters (no optimization)
    init_params = np.random.normal(0, 0.05, enc_circ.num_parameters)
    q_kernel.assign_parameters(init_params)
    
    best_overall_params = init_params.copy()
    best_overall_layer = NO_LAYERS
    best_overall_history = [0.0]  # Placeholder history
    
    print(f"\n✓ Fixed gamma mode complete: gamma={GAMMA_FIXED}, params initialized randomly")
    print(f"  Using parameters: {best_overall_layer} layers with {len(best_overall_params)} parameters\n")
    
    opt_params = best_overall_params
    NO_LAYERS = best_overall_layer

# ======================== OPTIMIZATION MODE ========================
# Standard iterative layer-by-layer optimization (skipped if GAMMA_FIXED is set)
else:
    for current_layer in range(NO_LAYERS, MAX_LAYERS + 1):
        print(f"\n{'='*80}")
        print(f"STARTING OPTIMIZATION WITH {current_layer} LAYER(S)")
        print(f"{'='*80}\n")
    
        # Create new encoding circuit with current number of layers
        enc_circ = ChebyshevPQC(N_QUBITS, num_layers=current_layer,
                                num_features=X_scaled.shape[1],  # Match 2 principal components
                             entangling_gate = 'rzz',
                              alpha = alpha_param)  # Reduced for lower-dimensional encoding

        # enc_circ = HubregtsenEncodingCircuit(
        #     num_qubits=N_QUBITS,
        #     num_features=20,
        #     num_layers=NO_LAYERS,
        #     closed=False,     # False = Linear Entanglement (Chain). True = Ring.
        #     final_encoding=False
        # )
        #enc_circ = HighDimEncodingCircuit(N_QUBITS, num_layers=current_layer)
        # Create new quantum kernel with updated circuit
        if KERNEL_NAME == 'fidelity_kernel':
            q_kernel = FidelityKernel(
                encoding_circuit=enc_circ, 
                executor=executor,
                parameter_seed=SEED,
                evaluate_duplicates='off_diagonal',
                caching=True
            )
        else:
            # Use median heuristic gamma for optimization iterations
            try:
                temp_K_opt = rbf_kernel(X_scaled_opt[:min(500, len(X_scaled_opt))], gamma=1.0)
                temp_dists_opt = 1.0 - temp_K_opt
                median_dist_opt = np.median(temp_dists_opt[temp_dists_opt > 0])
                new_gamma = 1.0 / (2.0 * (median_dist_opt ** 2)) if median_dist_opt > 0 else 2.0
            except Exception:
                new_gamma = 2.0  # Fallback if median heuristic fails
            
            q_kernel = ProjectedQuantumKernel(
                encoding_circuit=enc_circ,
                executor=executor,
                measurement="XYZ",
                outer_kernel="gaussian",
                initial_parameters=np.random.normal(0, 0.05, enc_circ.num_parameters),
                #gamma= 1.0 / (3.0 * enc_circ.num_qubits),
                gamma = gamma_param # Median heuristic instead of hardcoded 50
            )
        # Recreate loss with new kernel
        # No row subsampling: always use full dataset; keep pair subsampling schedule for speed/precision tradeoff
        is_final_layer = (current_layer == MAX_LAYERS)

        if is_final_layer:
            # Final layer: higher pair subsampling for better loss estimates
            subsample_size = None
            pair_subsample_size = 50000  # Increased from 2000 for better estimates on full data
            gamma_subsample_size = 2000  # Increased from 800 for better median heuristic
        else:
            # Early layers: reduced pair subsampling for faster optimization
            subsample_size = None
            pair_subsample_size = 5000   # Reduced from 10000 for 2x faster loss computation
            gamma_subsample_size = 500   # Reduced from 1000
    
        if LOSS == "MMDPairwiseDistanceLoss":
            proxy = KernelCacheProxy(q_kernel, cache_exact=True, cache_approx=False, approx_tol=1e-6)
            loss_obj = MMDPairwiseDistanceLoss(
                subsample=subsample_size,
                random_state=SEED,
                kernel_scale=None,
                pair_subsample=pair_subsample_size,
                embedding_dim=32,
                use_linear_mmd=True,
                gamma_median_subsample=gamma_subsample_size
            )
            loss_obj.set_quantum_kernel(proxy)
    
        # Use adaptive or full iterations based on CLI argument (will be divided across batches if enabled)
        current_iter = ITER if (is_final_layer or not ADAPTIVE_ITERATIONS) else max(50, ITER // 2)

        # Prepare batch schedule if asked to cover full dataset during optimization
        if COVER_FULL_DATASET and (opt_sample_idx is not None) and (MAX_OPTIMIZATION_SAMPLES < X_scaled.shape[0]):
            all_idx = np.arange(X_scaled.shape[0])
            rng_batches = np.random.RandomState(SEED)
            rng_batches.shuffle(all_idx)
            batches = [all_idx[i:i+MAX_OPTIMIZATION_SAMPLES] for i in range(0, len(all_idx), MAX_OPTIMIZATION_SAMPLES)]
        else:
            batches = [opt_sample_idx if opt_sample_idx is not None else np.arange(X_scaled.shape[0])]

        # iterations per batch to roughly keep total iterations near current_iter
        iters_per_batch = max(1, int(np.ceil(current_iter / len(batches))))

        # Create optimizer for this layer based on OPTIMIZER_METHOD
        adaptive_lr = 1e-3 if is_final_layer else 5e-3
    
        if OPTIMIZER_METHOD == 'cobyla':
            # COBYLA optimizer: gradient-free, good for noisy objectives
            torch_opt = COBYLAOptimizer(
                maxiter=iters_per_batch*3,
                rhobeg=1.0,
                tol=1e-6,
                verbose=True,
                device=device,
                random_state=SEED
            )
            print(f"  Using optimizer: COBYLA (gradient-free method)")
        elif OPTIMIZER_METHOD == 'lbfgsb':
            # L-BFGS-B optimizer via scipy
            torch_opt = LBFGSB(options={'maxiter': iters_per_batch, 'tol': 1e-6, 'ftol': 1e-9})
            print(f"  Using optimizer: L-BFGS-B (quasi-Newton method)")
        else:  # 'adam' or default
            # TorchAdamOptimizer with SPSA: efficient for many parameters
            torch_opt = TorchAdamOptimizer(
                maxiter=iters_per_batch*3,
                lr=adaptive_lr,
                central_eps=5e-5,
                verbose=True,
                device=device,
                n_jobs=1,
                grad_clip=2.0
            )
            print(f"  Using optimizer: TorchAdam with SPSA (default method)")
    
        def safe_evaluate(kernel_obj, X):
            X = np.asarray(X)
            n = X.shape[0]
            # Try proxy.evaluate -> kernel.evaluate -> callable(kernel)
            K = None
            # 1) try evaluate on kernel_obj (proxy or kernel)
            try:
                if hasattr(kernel_obj, "evaluate"):
                    K = kernel_obj.evaluate(X)
                    if K is not None:
                        K = np.asarray(K)
                        if K.ndim == 2 and K.shape[0] == n and K.shape[1] == n:
                            return K
            except Exception:
                K = None
            # 2) try callable interface
            try:
                if callable(kernel_obj):
                    K = kernel_obj(X)
                    if K is not None:
                        K = np.asarray(K)
                        if K.ndim == 2 and K.shape[0] == n and K.shape[1] == n:
                            return K
            except Exception:
                K = None
            # 3) fallback to underlying kernel if proxy wrappers are used
            underlying = getattr(kernel_obj, "_kernel", None) or getattr(kernel_obj, "_quantum_kernel", None) or kernel_obj
            try:
                if hasattr(underlying, "evaluate"):
                    K = np.asarray(underlying.evaluate(X))
                    if K.ndim == 2 and K.shape[0] == n and K.shape[1] == n:
                        # update proxy cache if present
                        if hasattr(kernel_obj, "_last_K"):
                            try:
                                kernel_obj._last_K = np.array(K, copy=True)
                                if getattr(kernel_obj, "_last_assign", None) is not None:
                                    kernel_obj._last_params = np.asarray(kernel_obj._last_assign).ravel().astype(float)
                            except Exception:
                                pass
                        return K
            except Exception:
                K = None
            # 4) final callable fallback on underlying
            try:
                if callable(underlying):
                    K = np.asarray(underlying(X))
                    if K.ndim == 2 and K.shape[0] == n and K.shape[1] == n:
                        if hasattr(kernel_obj, "_last_K"):
                            try:
                                kernel_obj._last_K = np.array(K, copy=True)
                                if getattr(kernel_obj, "_last_assign", None) is not None:
                                    kernel_obj._last_params = np.asarray(kernel_obj._last_assign).ravel().astype(float)
                            except Exception:
                                pass
                        return K
            except Exception:
                pass
            raise RuntimeError("safe_evaluate: all kernel evaluation attempts failed or returned mismatched shapes.")


        #Warm-up: find a small-random init that yields non-collapsed off-diagonal variance.
        def _warmup_find_noncollapsed_params(kernel_obj, X, rng_seed=SEED, tries=5, scales=(1e-3,1e-2,5e-2,1e-1,2e-1), subset=200, thresh=1e-5):
            rng = _np.random.RandomState(rng_seed)
            n = X.shape[0]
            idx = rng.choice(n, min(subset, n), replace=False)
            X_sub = X[idx]
            p_len = getattr(kernel_obj, "num_parameters", None) or getattr(getattr(kernel_obj, "_kernel", None), "num_parameters", None)
            if p_len is None:
                return None
            p_len = int(p_len)
            last_cand = None
            for s in scales[:tries]:
                cand = rng.normal(scale=float(s), size=p_len)
                last_cand = cand
                # assign params if supported
                try:
                    kernel_obj.assign_parameters(cand)
                except Exception:
                    try:
                        if hasattr(kernel_obj, "set_parameters"):
                            kernel_obj.set_parameters(cand)
                    except Exception:
                        pass
                # evaluate safely
                try:
                    K = safe_evaluate(kernel_obj, X_sub)
                except Exception:
                    K = None
                if K is None:
                    continue
                off = K[~_np.eye(K.shape[0], dtype=bool)]
                if off.size == 0:
                    continue
                if _np.std(off) > float(thresh):
                    print("Warmup found params with off-diag std=", _np.std(off))
                    return cand
            print("Warmup did not find high-variance init; returning last candidate")
            return last_cand if last_cand is not None else rng.normal(scale=1e-2, size=p_len)

        # # run warmup and assign to proxy
        # try:
        #     candidate_params = _warmup_find_noncollapsed_params(proxy, X_scaled, rng_seed=SEED, tries=5, subset=min(200, X_scaled.shape[0]), thresh=1e-5)
        #     if candidate_params is not None:
        #         try:
        #             proxy.assign_parameters(candidate_params)
        #         except Exception:
        #             try:
        #                 proxy.set_parameters(candidate_params)
        #             except Exception:
        #                 pass
        # except Exception as e:
        #     print("Warmup routine failed:", e)
        #     import traceback; traceback.print_exc()

    
        # prefer to pass proxy (qk_for_optimizer) when available so optimizer and loss share the same cache
        qk_for_opt = locals().get('qk_for_optimizer', None) or q_kernel
        kernel_optimizer = KernelOptimizer(quantum_kernel=qk_for_opt, loss=loss_obj, optimizer=torch_opt)
    
        # Only start periodic saver for final 2 layers (skip for early layers to save I/O)
        save_checkpoints = (current_layer >= MAX_LAYERS - 1)
        stop_event = threading.Event()
        if save_checkpoints:
            saver_thread = threading.Thread(target=periodic_saver, args=(stop_event, 60.0, kernel_optimizer, TARGET_OUTDIR), daemon=True)
            saver_thread.start()
        else:
            saver_thread = None
    
        # Run optimization across batches, carrying parameters forward
        try:
            # Diagnostics: optional quick checks before running full optimization
            if RUN_DIAGNOSTICS:
                print("\n=== RUNNING DIAGNOSTICS: disabling kernel cache and probing FD gradients ===")
                try:
                    REUSE_KERNEL_CACHE = False
                except Exception:
                    pass
                # Probe finite-difference sensitivity for first few params
                try:
                    params0 = None
                    if getattr(kernel_optimizer, '_initial_parameters', None) is not None:
                        params0 = np.asarray(kernel_optimizer._initial_parameters).ravel().copy()
                    elif getattr(kernel_optimizer, '_parameters', None) is not None:
                        params0 = np.asarray(kernel_optimizer._parameters).ravel().copy()
                    else:
                        params0 = np.asarray(kernel_optimizer.get_optimal_parameters() or kernel_optimizer.quantum_kernel.parameters or [])
                    params0 = params0.ravel()
                    print(f"Diagnostic: using params length={len(params0)}")
                    loss0 = None
                    try:
                        loss0 = loss_obj.compute(params0, data=X_scaled_opt, labels=None)
                        print(f"Diagnostic: baseline loss on opt-data = {loss0:.6f}")
                    except Exception as e:
                        print(f"Diagnostic: baseline loss computation failed: {e}")
                    eps = 1e-5
                    for i in range(min(5, len(params0))):
                        p = params0.copy(); p[i] += eps
                        try:
                            l1 = loss_obj.compute(p, data=X_scaled_opt, labels=None)
                        except Exception as e:
                            l1 = float('nan')
                        p = params0.copy(); p[i] -= eps
                        try:
                            l2 = loss_obj.compute(p, data=X_scaled_opt, labels=None)
                        except Exception as e:
                            l2 = float('nan')
                        grad_central = (l1 - l2) / (2*eps) if (np.isfinite(l1) and np.isfinite(l2)) else float('nan')
                        print(f"param[{i}] grad_central ~ {grad_central:.3e} (l_plus={l1:.6f} l_minus={l2:.6f})")
                except Exception as e:
                    print("Diagnostic FD probe failed:", e)
            prev_params_for_diagnostics = None
            last_params = None
            best_batch_loss = float('inf')
            best_batch_result = None
        
            for b_idx, batch in enumerate(batches, start=1):
                X_batch = X_scaled[batch]

                # warm start with last_params if available
                if last_params is not None:
                    try:
                        kernel_optimizer._optimal_parameters = last_params.copy()
                    except Exception:
                        pass
                    try:
                        kernel_optimizer.quantum_kernel.assign_parameters(last_params)
                    except Exception:
                        pass

                batch_result = kernel_optimizer.run_optimization(X_batch)
                # Extract loss from this batch
                batch_loss = getattr(batch_result, 'fun', None)
                print(f'\nLayer {current_layer} batch {b_idx}/{len(batches)} finished. nit={getattr(batch_result, "nit", None)}, loss={batch_loss}')
            
                # Track best result across all batches
                if batch_loss is not None and batch_loss < best_batch_loss:
                    best_batch_loss = batch_loss
                    best_batch_result = batch_result

                # capture parameters for next batch (use last batch's params for warm start)
                cur_params = getattr(kernel_optimizer, '_optimal_parameters', None)
                if RUN_DIAGNOSTICS:
                    try:
                        if prev_params_for_diagnostics is None and cur_params is not None:
                            prev_params_for_diagnostics = np.asarray(cur_params).ravel().copy()
                        elif (prev_params_for_diagnostics is not None) and (cur_params is not None):
                            cur = np.asarray(cur_params).ravel()
                            delta = np.linalg.norm(cur - prev_params_for_diagnostics)
                            print(f"[Diagnostics] After batch {b_idx}: param_norm={np.linalg.norm(cur):.6e} delta_norm={delta:.6e}")
                            prev_params_for_diagnostics = cur.copy()
                    except Exception:
                        pass
                last_params = cur_params

            # Use the best result across all batches
            opt_result = best_batch_result if best_batch_result is not None else batch_result
            final_loss = getattr(opt_result, 'fun', None) if opt_result is not None else None
        
            if opt_result is None or final_loss is None:
                print(f'\n⚠️  WARNING: Layer {current_layer} - no valid optimization result from any batch')
            else:
                print(f'\nLayer {current_layer} complete. Best loss across batches: {final_loss}')
        
            # Extract history (from best batch result)
            history = None
            if opt_result is not None:
                if hasattr(opt_result, 'history'):
                    history = getattr(opt_result, 'history')
                elif isinstance(opt_result, dict) and ('history' in opt_result):
                    history = opt_result.get('history')
                elif hasattr(kernel_optimizer, '_loss_history'):
                    history = getattr(kernel_optimizer, '_loss_history')
        
            # Note: Individual layer plots are not saved here.
            # Final plot will be created after loop using best_overall_history
        
            # Store results only if we have valid output
            if opt_result is not None and final_loss is not None:
                opt_params_current = kernel_optimizer._optimal_parameters
                all_results.append({
                    'layer': current_layer,
                    'loss': final_loss,
                    'params': opt_params_current,
                    'history': history
                })
            
                # Track best overall
                if final_loss < best_overall_loss:
                    best_overall_loss = final_loss
                    best_overall_params = opt_params_current.copy() if opt_params_current is not None else None
                    best_overall_layer = current_layer
                    best_overall_history = history  # Store history of best run
                    print(f"*** New best loss: {best_overall_loss} at layer {current_layer} ***")
            else:
                print(f"⚠️  Skipping layer {current_layer} - invalid result")
        
        except Exception as e:
            print(f'Layer {current_layer} optimization failed:', e)
            import traceback; traceback.print_exc()
    
        # Stop periodic saver thread (only if it was started)
        if save_checkpoints:
            stop_event.set()
            saver_thread.join(timeout=5.0)

# Calculate total optimization time
optimization_end_time = time.time()
total_time_seconds = optimization_end_time - optimization_start_time
total_hours = total_time_seconds / 3600
total_minutes = (total_time_seconds % 3600) / 60

print(f"\n{'='*80}")
if GAMMA_FIXED is not None:
    print(f"FIXED GAMMA MODE COMPLETE")
    print(f"Gamma value used: {GAMMA_FIXED}")
else:
    print(f"OPTIMIZATION COMPLETE")
    print(f"Best loss: {best_overall_loss} at layer {best_overall_layer}")
print(f"Total time: {total_hours:.2f} hours ({total_time_seconds:.2f} seconds)")
if GAMMA_FIXED is None:
    print(f"Time breakdown: {int(total_hours)} hours, {int(total_minutes)} minutes")
print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"{'='*80}\n")

# Plot the optimization history from the best run
if best_overall_history is not None:
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(8,4))
        plt.plot(range(1, len(best_overall_history)+1), best_overall_history, marker='o', linewidth=2, markersize=4)
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title(f'Best Optimization History (Layer {best_overall_layer}, Final Loss: {best_overall_loss:.6f})', fontsize=13)
        plt.grid(True, alpha=0.3)
        fname = os.path.join(TARGET_OUTDIR, f'best_loss_history_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
        fig.savefig(fname, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved best optimization history plot to: {fname}")
    except Exception as e:
        print(f"Warning: Could not save optimization history plot: {e}")

save_results = True

# Check if optimization succeeded
if best_overall_params is None or best_overall_layer is None:
    print("\n⚠️  WARNING: Optimization did not produce valid results.")
    print("Skipping final evaluation. Check error messages above.")
    import sys
    sys.exit(1)

# Use best overall parameters
opt_params = best_overall_params
NO_LAYERS = best_overall_layer  # Update to best layer count

# Recreate circuit and kernel with best configuration
print(f"\nRecreating circuit with {N_QUBITS} qubits, {NO_LAYERS} layers, {X_scaled.shape[1]} features...")
# enc_circ = ChebyshevPQC(N_QUBITS, num_layers=NO_LAYERS,  
#                          #entangling_gate = 'rzz',
#                           alpha = 5.0,
#                         num_features=X_scaled.shape[1])
enc_circ = ChebyshevPQC(N_QUBITS, num_layers=NO_LAYERS,  
                         entangling_gate = 'rzz',
                          alpha = alpha_param,
                        num_features=X_scaled.shape[1])
# enc_circ = HubregtsenEncodingCircuit(
#     num_qubits=N_QUBITS,
#     num_features=20,
#     num_layers=NO_LAYERS,
#     closed=False,     # False = Linear Entanglement (Chain). True = Ring.
#     final_encoding=False
# )
#enc_circ = HighDimEncodingCircuit(N_QUBITS, num_layers=NO_LAYERS,)
# Ensure circuit is properly built
print(f"Circuit has {enc_circ.num_parameters} parameters")
if enc_circ.num_parameters is None or enc_circ.num_parameters == 0:
    print("⚠️  ERROR: Circuit num_parameters is None or 0!")
    import sys
    sys.exit(1)

if KERNEL_NAME == 'fidelity_kernel':
    q_kernel = FidelityKernel(
        encoding_circuit=enc_circ, 
        executor=executor,
        parameter_seed=SEED,
        evaluate_duplicates='off_diagonal',
        caching=True
    )
else:
    # For PKQ, generate random initial params if best_overall_params is None
    if best_overall_params is None:
        init_params = np.random.normal(0, 0.05, enc_circ.num_parameters)
    else:
        init_params = best_overall_params
    
    # Use median heuristic gamma for gamma tuning kernel
    temp_K_tune = rbf_kernel(X_scaled[:min(500, len(X_scaled))], gamma=1.0)
    temp_dists_tune = 1.0 - temp_K_tune
    median_dist_tune = np.median(temp_dists_tune[temp_dists_tune > 0])
    new_gamma = 1.0 / (2.0 * (median_dist_tune ** 2)) if median_dist_tune > 0 else 2.0
    
    q_kernel = ProjectedQuantumKernel(
        encoding_circuit=enc_circ,
        executor=executor,
        measurement="XYZ",
        outer_kernel="gaussian",
        initial_parameters=init_params,
        #gamma= 1.0 / (3.0 * enc_circ.num_qubits),
        gamma= gamma_param # Median heuristic (will be tuned below)
    )

# Assign optimal parameters to the quantum kernel
print(f"Assigning {len(opt_params)} optimal parameters to kernel...")
q_kernel.assign_parameters(opt_params)

# ==================== GAMMA TUNING LOOP ====================
# Find optimal gamma value that targets Rank(90%) in the 40s
print(f"\n{'='*60}")
print(f"=== GAMMA TUNING: Finding Optimal Gamma ===")
print(f"{'='*60}")

# Use a subset for fast gamma tuning
gamma_gamma_sample_size = min(500, X_scaled.shape[0])
gamma_rng = np.random.RandomState(SEED + 999)
gamma_tune_idx = gamma_rng.choice(X_scaled.shape[0], size=gamma_gamma_sample_size, replace=False)
X_gamma_tune = X_scaled[gamma_tune_idx]

gamma_candidates = [1.0, 2.0, 3.0, 5.0, 10.0, 15.0, 25.0, 50.0]
gamma_results = []

print(f"Tuning gamma on subset of {gamma_gamma_sample_size} samples")
print(f"Target: Rank(90%) in the 40s | Target Mean ~ 0.38")
print("-" * 70)

for g in gamma_candidates:
    try:
        # Create temporary kernel with current gamma for evaluation
        if KERNEL_NAME == 'fidelity_kernel':
            q_kernel_temp = FidelityKernel(
                encoding_circuit=enc_circ, 
                executor=executor,
                parameter_seed=SEED,
                evaluate_duplicates='off_diagonal',
                caching=True
            )
        else:
            q_kernel_temp = ProjectedQuantumKernel(
                encoding_circuit=enc_circ,
                executor=executor,
                measurement="XYZ",
                outer_kernel="gaussian",
                initial_parameters=opt_params if best_overall_params is not None else np.random.normal(0, 0.05, enc_circ.num_parameters),
                gamma=gamma_param
            )
        
        # Assign parameters to temporary kernel
        q_kernel_temp.assign_parameters(opt_params)
        
        # Evaluate kernel matrix on subset
        K_test = q_kernel_temp.evaluate(X_gamma_tune)
        
        # Calculate off-diagonal mean
        n_test = K_test.shape[0]
        off_diag_mean = (np.sum(K_test) - np.trace(K_test)) / (n_test * (n_test - 1)) if n_test > 1 else 0.0
        
        # Calculate Effective Rank (90%)
        eigenvalues = np.linalg.eigvalsh(K_test)
        eigenvalues = np.sort(eigenvalues)[::-1]  # Sort descending
        total_energy = np.sum(eigenvalues) if np.sum(eigenvalues) > 0 else 1.0
        energy = np.cumsum(eigenvalues) / total_energy
        rank_90 = np.searchsorted(energy, 0.90) + 1
        
        gamma_results.append({
            'gamma': g,
            'mean': off_diag_mean,
            'rank_90': rank_90,
            'top_10_eigs': eigenvalues[:10]
        })
        
        print(f"Gamma: {g:6.1f} | Mean: {off_diag_mean:.4f} | Rank(90%): {rank_90:3d}")
        
    except Exception as e:
        print(f"Gamma: {g:6.1f} | ERROR: {str(e)[:40]}")
        continue

# Find gamma closest to target Rank(90%) = 45 (mid of 40s)
print("-" * 70)
if gamma_results:
    # Find gamma with Rank(90%) closest to 45
    target_rank = 45
    best_gamma_idx = np.argmin([abs(r['rank_90'] - target_rank) for r in gamma_results])
    best_gamma_result = gamma_results[best_gamma_idx]
    optimal_gamma = best_gamma_result['gamma']
    
    print(f"\n✓ OPTIMAL GAMMA FOUND:")
    print(f"  Gamma: {optimal_gamma:.1f}")
    print(f"  Rank(90%): {best_gamma_result['rank_90']}")
    print(f"  Off-diagonal Mean: {best_gamma_result['mean']:.4f}")
    print(f"  Top 10 eigenvalues: {best_gamma_result['top_10_eigs']}")
else:
    print("⚠️  WARNING: Gamma tuning failed. Using default gamma = 2")
    optimal_gamma = 2.0

print(f"{'='*60}\n")

# ==================== RECREATE KERNEL WITH OPTIMAL GAMMA ====================
print(f"Recreating kernel with optimal gamma = {optimal_gamma}")

if KERNEL_NAME == 'fidelity_kernel':
    q_kernel = FidelityKernel(
        encoding_circuit=enc_circ, 
        executor=executor,
        parameter_seed=SEED,
        evaluate_duplicates='off_diagonal',
        caching=True
    )
else:
    q_kernel = ProjectedQuantumKernel(
        encoding_circuit=enc_circ,
        executor=executor,
        measurement="XYZ",
        outer_kernel="gaussian",
        initial_parameters=opt_params if best_overall_params is not None else np.random.normal(0, 0.05, enc_circ.num_parameters),
        gamma=optimal_gamma
    )

# Assign optimal parameters again
q_kernel.assign_parameters(opt_params)

# Evaluate kernel matrix on smart-subsampled X_scaled dataset using batch/landmark approach
print(f"\n{'='*60}")
print(f"=== FINAL KERNEL EVALUATION (with optimal gamma={optimal_gamma}) ===")
print(f"{'='*60}")
print(f"Dataset size: {X_scaled.shape[0]} samples")
print("Using batched/landmark computation for efficiency...")

def evaluate_kernel_batched(kernel, X, batch_size=500, use_landmarks=False, n_landmarks=2000):
    """
    Evaluate quantum kernel matrix in batches or using landmark approximation.
    
    Similar to graphtools.Graph() landmark approach:
    1. If use_landmarks=True: Select n_landmarks samples, compute K_landmarks, 
       then compute K_data_to_landmarks for out-of-sample extension
    2. Otherwise: Compute full kernel in batches to manage memory
    
    This is the PHATE/graphtools-compatible approach:
    - Landmark approximation: K_dtl @ K_lm^(-1) @ K_dtl^T (Nyström)
    - Preserves manifold structure for PHATE
    - Dramatically reduces computation time
    
    Args:
        kernel: Quantum kernel object with evaluate() method
        X: Full dataset (n_samples, n_features)
        batch_size: Number of samples per batch for full kernel computation
        use_landmarks: If True, use landmark approximation (faster, approximate)
        n_landmarks: Number of landmark points
    
    Returns:
        K: Kernel matrix (n_samples, n_samples) or (n_samples, n_landmarks) if use_landmarks
    """
    n_samples = X.shape[0]
    
    if use_landmarks and n_samples > n_landmarks:
        print(f"  → Using landmark approximation with {n_landmarks} landmarks")
        print(f"  → Speedup: {(n_samples**2)/(n_landmarks**2 + n_samples*n_landmarks):.1f}x faster")
        
        # Select landmarks using random sampling (similar to graphtools random_landmarking)
        # Could also use kmeans/spectral clustering like graphtools does by default
        rng = np.random.RandomState(SEED)
        landmark_idx = rng.choice(n_samples, size=n_landmarks, replace=False)
        X_landmarks = X[landmark_idx]
        
        print(f"\\n  Step 1/2: Computing landmark kernel ({n_landmarks} x {n_landmarks})...")
        K_landmarks = kernel.evaluate(X_landmarks)  # (n_landmarks, n_landmarks)
        print(f"    ✓ Landmark kernel computed: {K_landmarks.shape}")
        
        print(f"\\n  Step 2/2: Computing data-to-landmark kernel ({n_samples} x {n_landmarks})...")
        # Compute in batches to avoid memory issues
        K_data_to_landmarks = np.zeros((n_samples, n_landmarks))
        for i in range(0, n_samples, batch_size):
            end_i = min(i + batch_size, n_samples)
            K_data_to_landmarks[i:end_i] = kernel.evaluate(X[i:end_i], X_landmarks)
            if (i // batch_size) % 5 == 0 or end_i == n_samples:
                print(f"    Progress: {end_i}/{n_samples} samples ({100*end_i/n_samples:.1f}%)...")
        
        print(f"    ✓ Data-to-landmark kernel computed: {K_data_to_landmarks.shape}")
        
        # For PHATE/graph construction, we can use the Nystrom approximation:
        # K_approx ≈ K_data_to_landmarks @ K_landmarks^(-1) @ K_data_to_landmarks.T
        # But often just using K_data_to_landmarks is sufficient for graphtools
        return K_data_to_landmarks, landmark_idx, K_landmarks
    
    else:
        print(f"  → Computing full kernel in batches (batch_size={batch_size})")
        print(f"  → Total kernel evaluations: {n_samples**2}")
        K = np.zeros((n_samples, n_samples))
        
        total_blocks = ((n_samples + batch_size - 1) // batch_size) ** 2
        current_block = 0
        
        # Compute upper triangle in batches
        for i in range(0, n_samples, batch_size):
            end_i = min(i + batch_size, n_samples)
            for j in range(i, n_samples, batch_size):
                end_j = min(j + batch_size, n_samples)
                current_block += 1
                
                if i == j:
                    # Diagonal block - symmetric
                    K_block = kernel.evaluate(X[i:end_i])
                    K[i:end_i, i:end_i] = K_block
                else:
                    # Off-diagonal block
                    K_block = kernel.evaluate(X[i:end_i], X[j:end_j])
                    K[i:end_i, j:end_j] = K_block
                    K[j:end_j, i:end_i] = K_block.T  # Symmetry
                
                if current_block % 10 == 0 or current_block == total_blocks:
                    print(f"    Progress: block {current_block}/{total_blocks} ({100*current_block/total_blocks:.1f}%)...")
        
        print(f"    ✓ Full kernel computed: {K.shape}")
        return K, None, None

# Adaptive strategy based on dataset size
# Using all 18,203 cells with Nyström landmark approximation (Moon et al. 2019 approach)
LANDMARK_SIZE = 5000  # 22% of 18,203 cells - optimal balance of quality vs speed
BATCH_SIZE = 500  # Batch size for kernel computation
# Only use landmarks when the dataset is larger than the landmark count
USE_LANDMARKS = X_scaled.shape[0] > LANDMARK_SIZE

print(f"\\nKernel computation strategy:")
print(f"  Dataset size: {X_scaled.shape[0]} samples (ALL CELLS - no smart subsampling)")
print(f"  Use landmarks: {USE_LANDMARKS}")
if USE_LANDMARKS:
    print(f"  Landmark count: {LANDMARK_SIZE} ({100*LANDMARK_SIZE/X_scaled.shape[0]:.1f}% of data)")
    print(f"  Expected kernel evaluations: {LANDMARK_SIZE**2 + X_scaled.shape[0]*LANDMARK_SIZE:,}")
    print(f"  vs full kernel: {X_scaled.shape[0]**2:,}")
    print(f"  Speedup: {(X_scaled.shape[0]**2)/(LANDMARK_SIZE**2 + X_scaled.shape[0]*LANDMARK_SIZE):.1f}x faster")
    print(f"  Estimated time: 14-16 hours (vs 40+ hours for full kernel)")
    print(f"  Method: Nyström approximation as validated in Moon et al. 2019 (PHATE paper)")
else:
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Total kernel evaluations: {X_scaled.shape[0]**2:,}")

result = evaluate_kernel_batched(
    q_kernel, 
    X_scaled, 
    batch_size=BATCH_SIZE,
    use_landmarks=USE_LANDMARKS,
    n_landmarks=LANDMARK_SIZE
)

print(f"\\n{'='*60}")
if USE_LANDMARKS:
    K_data_to_landmarks, landmark_idx, K_landmarks = result
    if K_landmarks is None:
        # Fallback: landmark path was not used (dataset smaller than landmarks)
        K_opt = K_data_to_landmarks
        print(f"✓ Full kernel computation complete (no landmarks): {K_opt.shape}")
    else:
        print(f"✓ Landmark computation complete!")
        print(f"  - K_data_to_landmarks: {K_data_to_landmarks.shape}")
        print(f"  - K_landmarks: {K_landmarks.shape}")
    
        # Use Nyström approximation for full kernel
        print(f"\\n  Computing Nyström approximation...")
        from scipy.linalg import pinv
        K_opt = K_data_to_landmarks @ pinv(K_landmarks) @ K_data_to_landmarks.T
        print(f"  ✓ Full kernel approximation: {K_opt.shape}")
        print(f"  Quality: Nyström preserves manifold structure for PHATE")
else:
    K_opt, _, _ = result
    print(f"✓ Full kernel computation complete: {K_opt.shape}")
print(f"{'='*60}\\n")



# Convert kernel matrix to a distance-like matrix for PHATE
#K_dist = 1.0 - (K_opt - K_opt.min()) / (K_opt.max() - K_opt.min()) if K_opt.max() > K_opt.min() else np.zeros_like(K_opt)

# Fit PHATE embedding using the optimal kernel-derived distance
#phate_op = PHATE(n_components=2, knn_dist="precomputed_distance", random_state=SEED)
#phate_embedding_whole = phate_op.fit_transform(K_dist)

if save_results:

    # Save K_opt and PHATE embedding to cache files
    #np.save("K_opt.npy", K_opt)
    # Get current date-time string for filenames
    dt_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamp = dt_str  # Make timestamp available for all following code

    # Prepare metadata dictionary
    metadata = {
    # "N_SAMPLES": N_SAMPLES,
        "DIGITS": DIGITS,
        "KERNEL": KERNEL_NAME,
       # "PCA_DIM": PCA_DIM,
        #"TSNE_DIM": TSNE_DIM,
        "N_QUBITS": N_QUBITS,
        "NO_LAYERS": NO_LAYERS,
        "BEST_LAYER": best_overall_layer,
        "BEST_LOSS": float(best_overall_loss),
        "LOSS FUNCTION": LOSS,
        "SEED": SEED,
        "OPTIMAL_GAMMA": float(optimal_gamma),
        "opt_params": opt_params.tolist(),
        "loss": LOSS,
        "optimization_time_seconds": float(total_time_seconds),
        "optimization_time_hours": float(total_hours),
        "optimization_start": datetime.fromtimestamp(optimization_start_time).strftime('%Y-%m-%d %H:%M:%S'),
        "optimization_end": datetime.fromtimestamp(optimization_end_time).strftime('%Y-%m-%d %H:%M:%S'),
        "all_layer_results": [{
            'layer': r['layer'],
            'loss': float(r['loss']) if r['loss'] is not None else None,
            'num_params': len(r['params'])
        } for r in all_results],

    }

    # save outputs atomically to TARGET_OUTDIR
    target_dir = TARGET_OUTDIR
    fname = f"opt_params_{dt_str}_{KERNEL_NAME}_{LOSS}_bestlayer{best_overall_layer}.npz"
    out_path = os.path.join(target_dir, fname)
    try:
        # Save raw quantum kernel K_opt
        save_dict = {'opt_params': opt_params, 'K_opt': K_opt, 'metadata': metadata}
        atomic_save_npz(out_path, **save_dict)
        print(f"Saved results to {out_path}")
    except Exception:
        # fallback to plain save
        save_dict = {'opt_params': opt_params, 'K_opt': K_opt, 'metadata': metadata}
        np.savez_compressed(out_path, **save_dict)
        print(f"Saved results (fallback) to {out_path}")
    
    # Also save all layer results separately
    all_layers_fname = f"all_layers_{dt_str}_{KERNEL_NAME}_{LOSS}.npz"
    all_layers_path = os.path.join(target_dir, all_layers_fname)
    try:
        all_layer_data = {f'layer{r["layer"]}_params': r['params'] for r in all_results}
        all_layer_data['metadata'] = metadata
        atomic_save_npz(all_layers_path, **all_layer_data)
        print(f"Saved all layer results to {all_layers_path}")
    except Exception:
        pass

    # ==================== Post-processing: diagnostics and visualization ====================
    # This block runs ONLY ONCE after the best layer is determined and saved

    # PSD projection and distance computation
    from numpy.linalg import eigvalsh
    from sklearn.decomposition import PCA
    import scipy.linalg as sla
    import scipy.sparse.linalg as spla

    def kernel_psd_projection(K, eig_tol=1e-12, eps_diag=1e-10, verbose=False, approx_rank=None):
        """Symmetrize and project K to a PSD matrix."""
        K = np.asarray(K)
        if K.ndim != 2 or K.shape[0] != K.shape[1]:
            raise ValueError("K must be square")

        orig_dtype = K.dtype
        K = K.astype(np.float32, copy=False)
        K = 0.5 * (K + K.T)
        K = np.nan_to_num(K, nan=0.0, posinf=np.finfo(np.float32).max, neginf=-np.finfo(np.float32).max)

        n = K.shape[0]
        idx = np.diag_indices(n)
        K = K.copy()
        K[idx] = K[idx] + eps_diag

        do_approx = (approx_rank is not None) and (0 < approx_rank < n - 1)
        if not do_approx:
            vals, vecs = sla.eigh(K.astype(np.float64))
            neg_count = np.sum(vals < -eig_tol)
            if verbose:
                print("PSD projection full-eig: smallest eig =", float(vals.min()), "neg_count =", int(neg_count))
            vals_clipped = np.clip(vals, a_min=0.0, a_max=None)
            K_psd = (vecs * vals_clipped) @ vecs.T
            K_psd = 0.5 * (K_psd + K_psd.T)
            K_psd[idx] += eps_diag
            return K_psd.astype(orig_dtype, copy=False)
        else:
            k = int(min(max(1, approx_rank), n - 2))
            if verbose:
                print("PSD projection: using approx_rank=", k)
            try:
                vals, vecs = spla.eigsh(K, k=k, which='LA', tol=1e-6, maxiter=None)
            except Exception as e:
                if verbose:
                    print("eigsh failed, falling back to full eigh (reason=)", e)
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

    def kernel_to_rkhs_distance(K_psd, clamp_tol=1e-12, rescale_percentile=None, dtype_out=np.float32):
        """Convert PSD kernel -> RKHS Euclidean distance matrix."""
        K = np.asarray(K_psd, dtype=np.float32)
        n = K.shape[0]
        diag = np.diag(K).copy()
        diag = np.maximum(diag, 0.0)
        D2 = diag[:, None] + diag[None, :] - 2.0 * K
        D2 = np.maximum(D2, 0.0)
        np.fill_diagonal(D2, 0.0)
        D = np.sqrt(D2, dtype=np.float32)

        if rescale_percentile is not None:
            vals = D[np.triu_indices(n, k=1)]
            if vals.size:
                p = float(rescale_percentile)
                scale = np.percentile(vals, p)
                if scale > 0:
                    D = (D / scale).astype(dtype_out, copy=False)
        D = np.nan_to_num(D, nan=0.0, posinf=np.finfo(np.float32).max, neginf=0.0)
        return D.astype(dtype_out, copy=False)

    def kernel_diagnostics(K, name="K", save_dir=None):
        """Generate and optionally save kernel diagnostics plots."""
        K = np.asarray(K, dtype=float)
        print(f"--- {name} diagnostics ---")
        print("shape:", K.shape)
        print("symmetry: max|K-K.T| =", np.max(np.abs(K - K.T)))
        print("finite: any NaN/Inf ->", np.any(~np.isfinite(K)))
        print("min/max/mean/std:", np.nanmin(K), np.nanmax(K), np.nanmean(K), np.nanstd(K))
        
        diag = np.diag(K)
        print("diag: min/max/mean:", np.nanmin(diag), np.nanmax(diag), np.nanmean(diag))
        
        rows_rounded = np.round(K, decimals=8)
        uniq_rows = np.unique(rows_rounded, axis=0).shape[0]
        print("unique rows:", uniq_rows, " / ", K.shape[0])
        
        try:
            eigs = eigvalsh(K)
            print("eigenvalues: min, max, sum, near-zero-count:",
                  float(eigs[0]), float(eigs[-1]), float(np.sum(eigs)), np.sum(eigs < 1e-8))
        except Exception as e:
            print("eigvalsh failed:", e)
            eigs = None

        if eigs is not None:
            total = np.sum(np.abs(eigs))
            if total > 0:
                cumsum = np.cumsum(np.abs(eigs)) / total
                eff_rank = np.searchsorted(cumsum, 0.90) + 1
                print("effective rank (90% energy):", eff_rank, "of", len(eigs))

        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(1, 3, figsize=(14, 4))
        im = ax[0].imshow(K, cmap="viridis", aspect="auto")
        ax[0].set_title(f"{name} heatmap")
        fig.colorbar(im, ax=ax[0], fraction=0.046, pad=0.02)

        if eigs is not None:
            ax[1].plot(np.arange(len(eigs)), np.sort(eigs)[::-1], marker='.')
            ax[1].set_yscale('symlog', linthresh=1e-12)
            ax[1].set_title("Eigenvalue spectrum (desc)")
            ax[1].set_xlabel("component")
            ax[1].set_ylabel("eigval")

        try:
            pca = PCA(n_components=20)
            rows = K.copy()
            if rows.shape[1] < 2:
                rows = np.hstack([rows, np.zeros((rows.shape[0], 1))])
            proj = pca.fit_transform(rows)
            ax[2].scatter(proj[:, 0], proj[:, 1], s=20, alpha=0.8)
            ax[2].set_title("PCA of kernel rows")
        except Exception as e:
            ax[2].text(0.5, 0.5, f"PCA failed\n{e}", ha='center')
            ax[2].set_title("PCA failed")

        plt.tight_layout()
        
        if save_dir is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            pdf_path = os.path.join(save_dir, f"kernel_diagnostics_{name}_{timestamp}.pdf")
            try:
                fig.savefig(pdf_path, format='pdf', dpi=300, bbox_inches='tight')
                print(f"Saved kernel diagnostics plot to: {pdf_path}")
            except Exception as e:
                print(f"Failed to save kernel diagnostics plot: {e}")
        
        plt.close(fig)

        return {
            "shape": K.shape,
            "sym_max_diff": float(np.max(np.abs(K - K.T))),
            "has_naninf": bool(np.any(~np.isfinite(K))),
            "min": float(np.nanmin(K)),
            "max": float(np.nanmax(K)),
            "mean": float(np.nanmean(K)),
            "std": float(np.nanstd(K)),
            "diag_min": float(np.nanmin(diag)),
            "diag_max": float(np.nanmax(diag)),
            "unique_rows": int(uniq_rows),
            "eigs": eigs
        }

    # Process best kernel and create visualizations
    print(f"\n{'='*80}")
    print("FINAL POST-PROCESSING: Creating kernel diagnostics and visualizations")
    print(f"{'='*80}\n")
    
    print("K_opt diag (first 10):", np.diag(K_opt)[:10])

    n = K_opt.shape[0]
    if n <= 3000:
        H = np.eye(n) - np.ones((n, n)) / n
        Kc = H.dot(K_opt).dot(H)
        vals_c = np.linalg.eigvalsh(Kc)[::-1]
        total = np.sum(np.abs(vals_c)) if np.sum(np.abs(vals_c)) > 0 else 1.0
        explained = np.abs(vals_c) / total
        eff_rank_centered = np.searchsorted(np.cumsum(explained), 0.90) + 1
        print("centered top eigs:", vals_c[:10])
        print("effective rank (centered, 90%):", eff_rank_centered)
    else:
        print("Skipping full centered-spectrum analysis for large n=", n)

    K_sym = 0.5 * (K_opt + K_opt.T)
    K_sym[np.isnan(K_sym)] = 0.0
    approx_rank = None

    K_psd = kernel_psd_projection(K_sym, eig_tol=1e-12, eps_diag=1e-10, verbose=True, approx_rank=approx_rank)
    
    K_quantum = K_psd
    print(f"Using PSD-projected quantum kernel for benchmarking: {K_quantum.shape}")

    # ==================== Unsupervised Gamma Tuning ====================
    # Use median heuristic + spectrum analysis (no labels required)
    print("\n" + "="*80)
    print("UNSUPERVISED GAMMA TUNING (Median Heuristic)")
    print("="*80)
    
    # Compute median distance for base gamma (median heuristic)
    D2_full = pairwise_distances(X_pca, metric="sqeuclidean")
    distances_flat = D2_full[np.triu_indices_from(D2_full, k=1)]  # Upper triangle
    median_d2 = np.median(distances_flat) if len(distances_flat) > 0 else 1.0
    base_gamma = 1.0 / (2.0 * median_d2) if median_d2 > 0 else 1.0
    
    print(f"Median squared distance: {median_d2:.6f}")
    print(f"Base gamma (median heuristic): {base_gamma:.6e}")
    
    # Generate gamma grid around median heuristic
    gamma_grid = base_gamma * np.array([0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0])
    gamma_grid = np.unique(np.clip(gamma_grid, 1e-6, 1e6))
    
    def evaluate_kernel_spectrum(D2, gamma):
        """Evaluate kernel quality by spectrum properties (unsupervised)."""
        K = np.exp(-gamma * D2)
        K_sym = 0.5 * (K + K.T)
        K_sym[np.isnan(K_sym)] = 0.0
        
        # Centred kernel for effective rank
        n = K_sym.shape[0]
        H = np.eye(n) - np.ones((n, n)) / n
        Kc = H @ K_sym @ H
        
        # Eigenvalues and effective rank
        eigs = np.linalg.eigvalsh(Kc)[::-1]
        eigs = np.clip(eigs, 0, None)  # Remove numerical negatives
        
        # Effective rank at 90% variance explained
        total = np.sum(eigs) if np.sum(eigs) > 0 else 1.0
        explained = eigs / total
        eff_rank = np.searchsorted(np.cumsum(explained), 0.90) + 1
        
        # Spectrum entropy (measure of information)
        eigs_norm = eigs / np.sum(eigs) if np.sum(eigs) > 0 else eigs
        eigs_norm = np.clip(eigs_norm, 1e-12, None)
        entropy = -np.sum(eigs_norm * np.log(eigs_norm))
        
        return {
            'gamma': gamma,
            'eff_rank': int(eff_rank),
            'entropy': float(entropy),
            'max_eig': float(eigs[0]) if len(eigs) > 0 else 0.0
        }
    
    # Evaluate all gamma candidates
    print("\nEvaluating gamma candidates...")
    print("-" * 80)
    print(f"{'Gamma':>15} | {'Eff.Rank':>10} | {'Entropy':>12} | {'Max.Eig':>12}")
    print("-" * 80)
    
    spectrum_results = []
    for gamma in gamma_grid:
        result = evaluate_kernel_spectrum(D2_full, gamma)
        spectrum_results.append(result)
        print(f"{gamma:15.3e} | {result['eff_rank']:10d} | {result['entropy']:12.4f} | {result['max_eig']:12.4f}")
    
    # Select gamma based on spectrum: prefer moderate effective rank (not too small, not too large)
    # Target: effective rank in range of ~10-50% of data size for good expressivity
    target_rank_min = max(5, int(X_pca.shape[0] * 0.05))
    target_rank_max = max(20, int(X_pca.shape[0] * 0.30))
    
    # Find gamma with effective rank closest to target range midpoint
    target_rank = (target_rank_min + target_rank_max) // 2
    best_idx = np.argmin([abs(r['eff_rank'] - target_rank) for r in spectrum_results])
    best_gamma = spectrum_results[best_idx]['gamma']
    
    print("-" * 80)
    print(f"Target effective rank: {target_rank} (range: {target_rank_min}-{target_rank_max})")
    print(f"✓ Selected gamma: {best_gamma:.3e} (eff_rank={spectrum_results[best_idx]['eff_rank']})")
    print("="*80 + "\n")

    print("Computing heat kernel on full data...")
    # Heat kernel: G = exp(-(d²) / (4t))
    # Derived from Maggs et al.: Topology identifies concurrent cyclic processes
    # Using diffusion time t = 1 / (4 * gamma) to match RBF kernel behavior
    t = 1.0 / (4.0 * best_gamma)
    D_squared = pairwise_distances(X_pca, squared=True)
    K_rbf = np.exp(-D_squared / (4.0 * t))
    print(f"✓ Heat kernel computed: {K_rbf.shape}, diffusion_time t={t:.3e}")

    # ==================== Kernel Quality Metrics ====================
    print(f"\n{'='*80}")
    print("KERNEL QUALITY COMPARISON")
    print(f"{'='*80}\n")


    def compute_geometric_difference_g(K_classical, K_quantum, eps=1e-7):
        """Compute geometric difference g between classical and quantum kernels.

        g = sqrt( max_eig( sqrt(Kq) @ inv(Kc) @ sqrt(Kq) ) )
        Uses PSD-safe eigendecomposition with symmetrization and clipping.
        """
        Kc = 0.5 * (np.asarray(K_classical) + np.asarray(K_classical).T)
        Kq = 0.5 * (np.asarray(K_quantum) + np.asarray(K_quantum).T)
        Kc = np.nan_to_num(Kc, nan=0.0, posinf=0.0, neginf=0.0)
        Kq = np.nan_to_num(Kq, nan=0.0, posinf=0.0, neginf=0.0)

        n = Kc.shape[0]
        Kc_reg = Kc + eps * np.eye(n)

        # sqrt(Kq) via eigendecomposition (PSD-safe)
        vals_q, vecs_q = np.linalg.eigh(Kq)
        vals_q = np.clip(vals_q, a_min=0.0, a_max=None)
        sqrt_vals_q = np.sqrt(vals_q)
        sqrt_Kq = (vecs_q * sqrt_vals_q) @ vecs_q.T

        # inv(Kc) via eigendecomposition (stable for symmetric matrices)
        vals_c, vecs_c = np.linalg.eigh(Kc_reg)
        vals_c = np.clip(vals_c, a_min=eps, a_max=None)
        inv_vals_c = 1.0 / vals_c
        Kc_inv = (vecs_c * inv_vals_c) @ vecs_c.T

        M = sqrt_Kq @ Kc_inv @ sqrt_Kq
        M = 0.5 * (M + M.T)
        max_eigval = np.max(np.linalg.eigvalsh(M))
        return float(np.sqrt(max(max_eigval, 0.0)))

    # ==================== Kernel Quality Metrics ====================
    print(f"\n{'='*80}")
    print("KERNEL QUALITY COMPARISON")
    print(f"{'='*80}\n")

    def compute_kernel_quality_metrics(K, name="Kernel"):
        """Compute comprehensive quality metrics for a kernel matrix.
        
        Includes:
        - Traditional metrics: condition number, effective rank, sparsity
        - Model Complexity (s) from Huang et al. 2021 Nature Communications
        - Eigenvalue spectrum analysis
        """
        K = np.asarray(K, dtype=np.float64)
        n = K.shape[0]
        
        metrics = {'name': name}
        
        # Basic properties
        metrics['shape'] = K.shape
        metrics['symmetry_error'] = float(np.max(np.abs(K - K.T)))
        metrics['min'] = float(np.min(K))
        metrics['max'] = float(np.max(K))
        metrics['mean'] = float(np.mean(K))
        metrics['std'] = float(np.std(K))
        metrics['has_nan_inf'] = bool(np.any(~np.isfinite(K)))
        
        # Diagonal analysis
        diag = np.diag(K)
        metrics['diag_min'] = float(np.min(diag))
        metrics['diag_max'] = float(np.max(diag))
        metrics['diag_mean'] = float(np.mean(diag))
        metrics['diag_std'] = float(np.std(diag))
        
        # Off-diagonal variance (important for kernel expressiveness)
        off_diag_mask = ~np.eye(n, dtype=bool)
        off_diag = K[off_diag_mask]
        metrics['off_diag_mean'] = float(np.mean(off_diag))
        metrics['off_diag_std'] = float(np.std(off_diag))
        metrics['off_diag_min'] = float(np.min(off_diag))
        metrics['off_diag_max'] = float(np.max(off_diag))
        
        # Sparsity (entries close to zero)
        metrics['sparsity_1e-6'] = float(np.mean(np.abs(K) < 1e-6))
        metrics['sparsity_1e-3'] = float(np.mean(np.abs(K) < 1e-3))
        
        # Eigenvalue analysis
        try:
            if n <= 5000:
                eigs = eigvalsh(K)
            else:
                # For large matrices, use sparse eigenvalue solver
                print(f"  Using sparse eigensolver for large matrix (n={n})")
                from scipy.sparse.linalg import eigsh
                K_sparse = K.astype(np.float64)
                eigs = eigsh(K_sparse, k=min(500, n-2), which='LA', return_eigenvectors=False)
            
            eigs = np.sort(eigs)[::-1]  # descending order
            
            metrics['eig_min'] = float(eigs[-1]) if len(eigs) == n else None
            metrics['eig_max'] = float(eigs[0])
            metrics['eig_sum'] = float(np.sum(eigs))
            metrics['eig_mean'] = float(np.mean(eigs))
            
            # Condition number (ratio of largest to smallest eigenvalue)
            if len(eigs) == n and eigs[-1] > 1e-12:
                metrics['condition_number'] = float(eigs[0] / eigs[-1])
            else:
                metrics['condition_number'] = None
            
            # Effective rank (90% energy)
            eigs_abs = np.abs(eigs)
            total_energy = np.sum(eigs_abs)
            if total_energy > 0:
                cumsum = np.cumsum(eigs_abs) / total_energy
                metrics['effective_rank_90'] = int(np.searchsorted(cumsum, 0.90) + 1)
                metrics['effective_rank_95'] = int(np.searchsorted(cumsum, 0.95) + 1)
                metrics['effective_rank_99'] = int(np.searchsorted(cumsum, 0.99) + 1)
            else:
                metrics['effective_rank_90'] = 0
                metrics['effective_rank_95'] = 0
                metrics['effective_rank_99'] = 0
            
            # Number of near-zero eigenvalues
            metrics['n_eigs_near_zero_1e-8'] = int(np.sum(eigs < 1e-8))
            metrics['n_eigs_near_zero_1e-6'] = int(np.sum(eigs < 1e-6))
            metrics['n_eigs_negative'] = int(np.sum(eigs < 0))
            
            # ==================== MODEL COMPLEXITY (s) ====================
            # From Huang et al. 2021 Nature Communications:
            # "Power of data in quantum machine learning"
            # https://www.nature.com/articles/s41467-021-22539-9
            #
            # Model Complexity: s = Tr(K^2) / Tr(K)^2
            # where K is normalized: K <- K / Tr(K)
            #
            # Physical interpretation:
            # - s = 1/N (maximum): kernel is identity-like, minimal complexity
            # - s = 1 (minimum): kernel is rank-1, maximal complexity
            # - Lower s means more expressive/complex model
            # - Higher s means simpler model (closer to identity)
            
            trace_K = np.trace(K)
            if trace_K > 1e-12:
                K_normalized = K / trace_K
                trace_K_norm = np.trace(K_normalized)  # Should be 1
                trace_K2_norm = np.trace(K_normalized @ K_normalized)
                
                model_complexity_s = trace_K2_norm / (trace_K_norm ** 2)
                metrics['model_complexity_s'] = float(model_complexity_s)
                metrics['model_complexity_interpretation'] = (
                    'low (complex)' if model_complexity_s < 0.1 else
                    'medium' if model_complexity_s < 0.5 else
                    'high (simple)'
                )
            else:
                metrics['model_complexity_s'] = None
                metrics['model_complexity_interpretation'] = 'undefined (trace near zero)'
            
        except Exception as e:
            print(f"  Warning: Eigenvalue analysis failed for {name}: {e}")
            metrics['eig_min'] = None
            metrics['eig_max'] = None
            metrics['condition_number'] = None
            metrics['effective_rank_90'] = None
            metrics['model_complexity_s'] = None
        
        return metrics

    # Compute metrics for quantum and tuned RBF kernels
    print("Computing quality metrics for quantum kernel...")
    metrics_quantum = compute_kernel_quality_metrics(K_quantum, name="K_quantum (Chebyshev)")
    
    print("Computing quality metrics for tuned RBF kernel...")
    metrics_rbf = compute_kernel_quality_metrics(K_rbf, name="K_rbf (Tuned RBF)")

    # Geometric difference g (Huang et al. 2021) between quantum and classical kernels
    print("Computing geometric difference g (Quantum vs Heat Gaussian kernel)...")
    g_metric = compute_geometric_difference_g(K_rbf, K_quantum, eps=1e-7)
    print(f"g metric (Quantum vs Classical alpha-decay): {g_metric:.6f}")

    # Print comparison table
    print(f"\n{'='*80}")
    print("KERNEL QUALITY COMPARISON TABLE")
    print(f"{'='*80}\n")
    
    comparison_metrics = [
        ('Symmetry Error', 'symmetry_error', '{:.2e}'),
        ('Min Value', 'min', '{:.4f}'),
        ('Max Value', 'max', '{:.4f}'),
        ('Mean Value', 'mean', '{:.4f}'),
        ('Std Dev', 'std', '{:.4f}'),
        ('Diag Mean', 'diag_mean', '{:.4f}'),
        ('Off-Diag Std', 'off_diag_std', '{:.4f}'),
        ('Sparsity (< 1e-3)', 'sparsity_1e-3', '{:.2%}'),
        ('Max Eigenvalue', 'eig_max', '{:.4f}'),
        ('Min Eigenvalue', 'eig_min', '{:.4e}'),
        ('Condition Number', 'condition_number', '{:.2e}'),
        ('Effective Rank (90%)', 'effective_rank_90', '{:d}'),
        ('Effective Rank (95%)', 'effective_rank_95', '{:d}'),
        ('MODEL COMPLEXITY (s)', 'model_complexity_s', '{:.6f}'),
        ('Complexity Interpretation', 'model_complexity_interpretation', '{}'),
        ('Geometric Difference g', 'g_metric', '{:.6f}'),
    ]
    
    print(f"{'Metric':<30} {'Quantum (Chebyshev)':<22} {'RBF (Tuned)':<20}")
    print("-" * 90)

    s_quantum = metrics_quantum.get('model_complexity_s')
    s_rbf = metrics_rbf.get('model_complexity_s')
    if s_quantum is not None and s_rbf is not None:
        if s_quantum < s_rbf:
            print(f"\n✓ s-score result: s_quantum ({s_quantum:.6f}) < s_rbf ({s_rbf:.6f})")
            print("  This supports the claim that the quantum kernel captures the manifold with lower model complexity.")
        else:
            print(f"\n⚠️  s-score result: s_quantum ({s_quantum:.6f}) >= s_rbf ({s_rbf:.6f})")
            print("  Quantum kernel is not lower in model complexity under this metric.")
    
    for metric_name, metric_key, fmt_str in comparison_metrics:
        val_quantum = metrics_quantum.get(metric_key)
        val_rbf = metrics_rbf.get(metric_key)
        
        def format_val(v):
            if v is None:
                return 'N/A'
            elif isinstance(v, str):
                return v
            elif isinstance(v, (int, np.integer)):
                return fmt_str.format(v)
            elif isinstance(v, (float, np.floating)):
                return fmt_str.format(v)
            else:
                return str(v)
        
        print(f"{metric_name:<30} {format_val(val_quantum):<22} {format_val(val_rbf):<20}")
    
    print("-" * 90)
    
    # Save comparison to file
    comparison_data = {
        'timestamp': timestamp,
        'optimization': {
            'total_time_seconds': float(total_time_seconds),
            'total_time_hours': float(total_hours),
            'total_time_formatted': f"{int(total_hours)}h {int(total_minutes)}m",
            'best_loss': float(best_overall_loss),
            'best_layer': int(best_overall_layer),
            'optimization_start': datetime.fromtimestamp(optimization_start_time).strftime('%Y-%m-%d %H:%M:%S'),
            'optimization_end': datetime.fromtimestamp(optimization_end_time).strftime('%Y-%m-%d %H:%M:%S'),
        },
        'kernels': {
            'quantum_chebyshev': metrics_quantum,
            'rbf_tuned': metrics_rbf,
        },
        's_score_comparison': {
            's_quantum': metrics_quantum.get('model_complexity_s'),
            's_rbf': metrics_rbf.get('model_complexity_s'),
            'quantum_less_than_rbf': (
                metrics_quantum.get('model_complexity_s') is not None
                and metrics_rbf.get('model_complexity_s') is not None
                and metrics_quantum.get('model_complexity_s') < metrics_rbf.get('model_complexity_s')
            )
        },
        'rbf_tuning': {
            'best_gamma': best_gamma,
            'spectrum_results': spectrum_results,
            'tuning_method': 'unsupervised_spectrum_analysis',
        },
        'geometric_difference_g': {
            'value': g_metric,
            'description': 'Geometric difference g between quantum and classical kernels (Huang et al. 2021).'
        },
        'model_complexity_note': (
            'Model Complexity (s) from Huang et al. 2021: s = Tr(K^2) / Tr(K)^2. '
            'Lower values indicate more complex/expressive models. '
            's = 1/N (max) means identity-like kernel, s = 1 (min) means rank-1 kernel.'
        ),
    }
    
    comparison_json_path = os.path.join(TARGET_OUTDIR, f"kernel_comparison_{timestamp}.json")
    try:
        with open(comparison_json_path, 'w') as f:
            json.dump(comparison_data, f, indent=2)
        print(f"\n✓ Saved kernel comparison to: {comparison_json_path}")
    except Exception as e:
        print(f"Warning: Failed to save kernel comparison JSON: {e}")
    
    # Save comparison as CSV for easy viewing
    comparison_csv_path = os.path.join(TARGET_OUTDIR, f"kernel_comparison_{timestamp}.csv")
    try:
        df_comparison = pd.DataFrame([
            {
                'Metric': metric_name,
                'Quantum_Chebyshev': metrics_quantum.get(metric_key),
                'RBF_Tuned': metrics_rbf.get(metric_key),
            }
            for metric_name, metric_key, _ in comparison_metrics
        ])
        df_comparison.to_csv(comparison_csv_path, index=False)
        print(f"✓ Saved kernel comparison CSV to: {comparison_csv_path}")
    except Exception as e:
        print(f"Warning: Failed to save kernel comparison CSV: {e}")
    
    print(f"\n{'='*80}\n")

    # Save diagnostics plots for both kernels
    kernel_diagnostics(K_quantum, name="K_quantum (Chebyshev)", save_dir=TARGET_OUTDIR)
    kernel_diagnostics(K_rbf, name="K_rbf (Tuned)", save_dir=TARGET_OUTDIR)

    # ==================== Generate Comparison Figures ====================
    print(f"\n{'='*80}")
    print("GENERATING KERNEL COMPARISON FIGURES")
    print(f"{'='*80}\n")

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Helper function to safely get metric values (handles None)
    def safe_get(metrics_dict, key, default):
        """Get value from metrics dict, returning default if key is missing or value is None."""
        val = metrics_dict.get(key, default)
        return default if val is None else val

    # Figure 1: Heatmap comparison of both kernels
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Sample kernels for visualization (use every Nth sample for large matrices)
    sample_step = max(1, K_quantum.shape[0] // 500)
    sample_idx = np.arange(0, K_quantum.shape[0], sample_step)
    
    K_quantum_sampled = K_quantum[np.ix_(sample_idx, sample_idx)]
    K_rbf_sampled = K_rbf[np.ix_(sample_idx, sample_idx)]
    
    im0 = axes[0].imshow(K_quantum_sampled, cmap='viridis', aspect='auto')
    axes[0].set_title('K_quantum\n(Chebyshev)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Sample Index')
    axes[0].set_ylabel('Sample Index')
    fig.colorbar(im0, ax=axes[0], label='Kernel Value')

    im1 = axes[1].imshow(K_rbf_sampled, cmap='viridis', aspect='auto')
    axes[1].set_title('K_rbf\n(Tuned RBF)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Sample Index')
    axes[1].set_ylabel('Sample Index')
    fig.colorbar(im1, ax=axes[1], label='Kernel Value')
    
    plt.tight_layout()
    heatmap_path = os.path.join(TARGET_OUTDIR, f"kernel_heatmap_comparison_{timestamp}.png")
    fig.savefig(heatmap_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Saved kernel heatmap comparison to: {heatmap_path}")

    # Figure 2: Eigenvalue spectrum comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Compute eigenvalues for comparison
    eigs_quantum = np.sort(eigvalsh(K_quantum))[::-1][:500]
    eigs_rbf = np.sort(eigvalsh(K_rbf))[::-1][:500]
    
    ax.semilogy(np.arange(len(eigs_quantum)), eigs_quantum, marker='o', linewidth=2, markersize=4, label='K_quantum', alpha=0.7)
    ax.semilogy(np.arange(len(eigs_rbf)), eigs_rbf, marker='s', linewidth=2, markersize=4, label='K_rbf (Tuned)', alpha=0.7)
    
    ax.set_xlabel('Component Index', fontsize=12, fontweight='bold')
    ax.set_ylabel('Eigenvalue (log scale)', fontsize=12, fontweight='bold')
    ax.set_title('Eigenvalue Spectrum Comparison\n(First 500 components)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, loc='best')
    
    plt.tight_layout()
    eigs_path = os.path.join(TARGET_OUTDIR, f"kernel_eigenvalues_{timestamp}.png")
    fig.savefig(eigs_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Saved eigenvalue spectrum plot to: {eigs_path}")

    # Figure 3: Model Complexity (s) bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    
    kernels = ['Quantum\n(Chebyshev)', 'RBF\n(Tuned)']
    complexity_values = [
        safe_get(metrics_quantum, 'model_complexity_s', 0),
        safe_get(metrics_rbf, 'model_complexity_s', 0)
    ]
    colors = ['#1f77b4', '#ff7f0e']
    
    bars = ax.bar(kernels, complexity_values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, complexity_values)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.6f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_ylabel('Model Complexity (s)', fontsize=12, fontweight='bold')
    ax.set_title('Model Complexity Comparison\n(Lower = More Complex/Expressive)', fontsize=13, fontweight='bold')
    ax.set_ylim(0, max(complexity_values) * 1.2)
    ax.grid(True, axis='y', alpha=0.3)
    
    # Add interpretation text
    ax.text(0.5, 0.95, 's from Huang et al. 2021: s = Tr(K²) / Tr(K)²',
            transform=ax.transAxes, ha='center', va='top', fontsize=10, style='italic',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    complexity_path = os.path.join(TARGET_OUTDIR, f"model_complexity_comparison_{timestamp}.png")
    fig.savefig(complexity_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Saved model complexity bar chart to: {complexity_path}")

    # Figure 4: Effective rank comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    
    energy_levels = ['90%', '95%', '99%']
    
    quantum_ranks = [
        safe_get(metrics_quantum, 'effective_rank_90', 0),
        safe_get(metrics_quantum, 'effective_rank_95', 0),
        safe_get(metrics_quantum, 'effective_rank_99', 0)
    ]
    rbf_ranks = [
        safe_get(metrics_rbf, 'effective_rank_90', 0),
        safe_get(metrics_rbf, 'effective_rank_95', 0),
        safe_get(metrics_rbf, 'effective_rank_99', 0)
    ]
    
    x = np.arange(len(energy_levels))
    width = 0.35
    
    ax.bar(x - width / 2, quantum_ranks, width, label='Quantum (Chebyshev)', alpha=0.8, color='#1f77b4')
    ax.bar(x + width / 2, rbf_ranks, width, label='RBF (Tuned)', alpha=0.8, color='#ff7f0e')
    
    ax.set_xlabel('Energy Level', fontsize=12, fontweight='bold')
    ax.set_ylabel('Effective Rank', fontsize=12, fontweight='bold')
    ax.set_title('Effective Rank Comparison\n(Number of components needed to explain X% variance)', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(energy_levels)
    ax.legend(fontsize=11)
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    rank_path = os.path.join(TARGET_OUTDIR, f"effective_rank_comparison_{timestamp}.png")
    fig.savefig(rank_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Saved effective rank comparison to: {rank_path}")

    # Figure 5: Condition number and properties
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Panel 1: Condition number
    ax = axes[0, 0]
    kernels_list = ['Quantum\n(Chebyshev)', 'RBF\n(Tuned)']
    cond_nums = [
        safe_get(metrics_quantum, 'condition_number', 1e-10),
        safe_get(metrics_rbf, 'condition_number', 1e-10)
    ]
    colors_list = ['#1f77b4', '#ff7f0e']
    ax.bar(kernels_list, cond_nums, color=colors_list, alpha=0.7, edgecolor='black', linewidth=2)
    ax.set_ylabel('Condition Number', fontsize=11, fontweight='bold')
    ax.set_title('Condition Number (lower = better conditioned)', fontsize=12, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(True, axis='y', alpha=0.3)
    
    # Panel 2: Mean eigenvalue
    ax = axes[0, 1]
    mean_eigs = [
        safe_get(metrics_quantum, 'eig_mean', 0),
        safe_get(metrics_rbf, 'eig_mean', 0)
    ]
    ax.bar(kernels_list, mean_eigs, color=colors_list, alpha=0.7, edgecolor='black', linewidth=2)
    ax.set_ylabel('Mean Eigenvalue', fontsize=11, fontweight='bold')
    ax.set_title('Mean Eigenvalue', fontsize=12, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)
    
    # Panel 3: Off-diagonal std (expressiveness)
    ax = axes[1, 0]
    off_diag_stds = [
        safe_get(metrics_quantum, 'off_diag_std', 0),
        safe_get(metrics_rbf, 'off_diag_std', 0)
    ]
    ax.bar(kernels_list, off_diag_stds, color=colors_list, alpha=0.7, edgecolor='black', linewidth=2)
    ax.set_ylabel('Off-Diagonal Std Dev', fontsize=11, fontweight='bold')
    ax.set_title('Off-Diagonal Variability (higher = more expressive)', fontsize=12, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)
    
    # Panel 4: Sparsity
    ax = axes[1, 1]
    sparsities = [
        safe_get(metrics_quantum, 'sparsity_1e-3', 0) * 100,
        safe_get(metrics_rbf, 'sparsity_1e-3', 0) * 100
    ]
    ax.bar(kernels_list, sparsities, color=colors_list, alpha=0.7, edgecolor='black', linewidth=2)
    ax.set_ylabel('Sparsity (%)', fontsize=11, fontweight='bold')
    ax.set_title('Sparsity (% entries < 1e-3)', fontsize=12, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_ylim(0, 100)
    
    plt.tight_layout()
    props_path = os.path.join(TARGET_OUTDIR, f"kernel_properties_comparison_{timestamp}.png")
    fig.savefig(props_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Saved kernel properties comparison to: {props_path}")

    print(f"\n{'='*80}\n")
    print("KERNEL COMPARISON ANALYSIS COMPLETE")
    print(f"{'='*80}\n")
    print("Generated outputs:")
    print(f"  - Comparison CSV: {comparison_csv_path}")
    print(f"  - Comparison JSON: {comparison_json_path}")
    print(f"  - Heatmap comparison: {heatmap_path}")
    print(f"  - Eigenvalue spectrum: {eigs_path}")
    print(f"  - Model complexity chart: {complexity_path}")
    print(f"  - Effective rank chart: {rank_path}")
    print(f"  - Properties comparison: {props_path}")
    print(f"\n{'='*80}\n")


