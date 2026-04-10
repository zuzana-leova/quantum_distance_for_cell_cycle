#!/usr/bin/env bash
set -euo pipefail
cd /home/leova3397/projects/squlearn

# DIAGNOSTIC RUN: Test single case with no normalization to see raw kernel scales
echo "=== DIAGNOSTIC: Running with raw kernels (no normalization) ==="
echo "    This will show actual kernel value ranges to diagnose scale mismatch"
~/.conda/envs/myenv/bin/python scripts/Kernel_benchmark.py \
  --data examples/tutorials/data/data_sqrt.h5ad \
  --outdir /data/gpfs/projects/punim0613/zuzana/qphate/results/kernel_bench_out/DIAGNOSTIC_raw_kernels \
  --n_qubits 10 \
  --num_layers 3 \
  --norm-method none \
  --landmarks 200
echo "Check the output above for kernel value ranges!"
echo "If classical is ~0.1-1.0 and quantum is ~1e-6-1e-3, use --norm-method unit"
echo ""
read -p "Press Enter to continue with full grid search (or Ctrl+C to stop)..."

# Choose normalization method (default: unit for most cases with scale mismatch)
# Options: spectral, unit, diagonal, rows, none
NORM_METHOD="${NORM_METHOD:-unit}"

echo ""
echo "=== Running full grid search with normalization method: ${NORM_METHOD} ==="
echo ""

for Q in $(seq 10 10); do
  for L in $(seq 3 20); do
    echo "=== Running Kernel_benchmark for n_qubits=${Q}, num_layers=${L} ==="
    ~/.conda/envs/myenv/bin/python scripts/Kernel_benchmark.py \
      --data examples/tutorials/data/data_sqrt.h5ad \
      --outdir /data/gpfs/projects/punim0613/zuzana/qphate/results/kernel_bench_out/qubits_${Q}_layer_${L} \
      --n_qubits ${Q} \
      --num_layers ${L} \
      --norm-method diagonal \
      || { echo "Run failed for qubits ${Q}, layer ${L}"; exit 1; }
  done
done

echo ""
echo "=== Grid search complete! ==="

~/.conda/envs/myenv/bin/python scripts/Kernel_benchmark.py \
  --data examples/tutorials/data/data_sqrt.h5ad \
  --outdir /data/gpfs/projects/punim0613/zuzana/qphate/results/kernel_bench_out
  --n_qubits 10 \
  --num_layers 4 \
  --kernel fidelity \
  --landmarks 200 \
  --norm-method diagonal

~/.conda/envs/myenv/bin/python scripts/Kernel_benchmark.py \
    --data examples/tutorials/data/data_sqrt.h5ad \
    --n_qubits 20 \
    --num_layers 4 
    --landmarks 500 \
    --kernel fidelity \
    --outdir  /data/gpfs/projects/punim0613/zuzana/qphate/results/kernel_bench_out

# tmux attach-session -t squlearn10q
    ~/.conda/envs/myenv/bin/python scripts/Kernel_benchmark.py \
  --data examples/tutorials/data/data_sqrt.h5ad \
  --outdir /data/gpfs/projects/punim0613/zuzana/qphate/results/kernel_bench_out
  --n_qubits 10 \
  --num_layers 4 --max_layers 4 \
  --kernel fidelity \
  --landmarks 200 \
  --active_landmarks 300 \
  --opt_samples 2000\
  --adaptive_iterations True \
  --norm-method diagonal


  # session test on gpgpu187
  ~/.conda/envs/myenv/bin/python scripts/TEST_Kernel.py \
  --data examples/tutorials/data/data_sqrt.h5ad \
  --n_qubits 10 \
  --num_layers 4 --max_layers 4 \
  --kernel_name fidelity_kernel \
  --iter 100 \
  --loss_subsample 500 \
  --opt_samples 1000 \
  --seed 42 \
  --output /data/gpfs/projects/punim0613/zuzana/qphate/results/kernel_bench_out



  # session test on gpgpu186
  ~/.conda/envs/myenv/bin/python scripts/TEST_Kernel.py \
  --data examples/tutorials/data/data_sqrt.h5ad \
  --n_qubits 10 \
  --num_layers 4 --max_layers 4 \
  --kernel_name fidelity_kernel \
  --iter 100 \
  --loss_subsample 500 \
  --opt_samples 1000 \
  --seed 42 \
  --output /data/gpfs/projects/punim0613/zuzana/qphate/results/kernel_bench_out

    ~/.conda/envs/myenv/bin/python scripts/TEST_Kernel.py \
  --data examples/tutorials/data/data_sqrt.h5ad \
  --n_qubits 10 \
  --num_layers 4 --max_layers 4 \
  --kernel_name projected \
  --iter 500 \
  --loss_subsample 500 \
  --opt_samples 1000 \
  --seed 42 \
  --output /data/gpfs/projects/punim0613/zuzana/qphate/results/kernel_bench_out


    ~/.conda/envs/myenv/bin/python scripts/TEST_Kernel_alpha_decay.py \
  --data examples/tutorials/data/data_sqrt.h5ad \
  --n_qubits 10 \
  --num_layers 4 --max_layers 4 \
  --kernel_name projected \
  --iter 1000 \
  --loss_subsample 500 \
  --opt_samples 1000 \
  --seed 42 \
  --output /data/gpfs/projects/punim0613/zuzana/qphate/results/kernel_bench_out


# tmux attach-session -t test
# tmux new-session -s squlearn_KTA 
cd /home/leova3397/projects/squlearn
    ~/.conda/envs/myenv/bin/python scripts/TEST_Kernel_alpha_decay_new_optimizer.py \
  --data examples/tutorials/data/data_sqrt.h5ad \
  --use_gpu
  --use_nystrom \
  --parallel_fd 8 \
  --nystrom_landmarks 300 \
  --n_qubits 10 \
  --num_layers 4 --max_layers 4 \
  --kernel_name projected \
  --iter 100 \
  --loss_subsample 100 \
  --opt_samples 2000 \
  --seed 42 \
  --output /data/gpfs/projects/punim0613/zuzana/qphate/results/kernel_bench_out

tmux new-session -s squlearn_test


    ~/.conda/envs/myenv/bin/python scripts/TEST_Kernel_alpha_decay_new_optimizer.py \
  --data examples/tutorials/data/data_sqrt.h5ad \
  --use_spsa \
  --use_gpu \
  --use_nystrom \
  --parallel_fd 8 \
  --nystrom_landmarks 300 \
  --n_qubits 10 \
  --num_layers 4 --max_layers 4 \
  --kernel_name projected \
  --iter 80 \
  --loss_subsample 100 \
  --opt_samples 2000 \
  --seed 42 \
  --output /data/gpfs/projects/punim0613/zuzana/qphate/results/kernel_bench_out

 --use_spsa \
  --parallel_fd 4 \
  --use_nystrom \
  --iter 80 \
  --max_layers 4


  ~/.conda/envs/myenv/bin/python scripts/TEST_tcell.py \
  --data examples/tutorials/data/data_sqrt.h5ad \
  --use_spsa \
  --use_gpu \
  --use_nystrom \
  --parallel_fd 8 \
  --nystrom_landmarks 300 \
  --n_qubits 10 \
  --num_layers 4 --max_layers 4 \
  --kernel_name projected \
  --iter 100 \
  --loss_subsample 100 \
  --opt_samples 2000 \
  --seed 42 \
  --output /data/gpfs/projects/punim0613/zuzana/qphate/results/kernel_bench_out

    ~/.conda/envs/myenv/bin/python scripts/TEST_capolupo.py \
  --data examples/tutorials/data/data_sqrt.h5ad \
  --use_spsa \
  --parallel_fd 8 \
  --n_qubits 10 \
  --num_layers 3 --max_layers 3 \
  --kernel_name projected \
  --iter 100 \
  --loss_subsample 100 \
  --opt_samples 2000 \
  --seed 42 \
  --output /data/gpfs/projects/punim0613/zuzana/qphate/results/kernel_bench_out

  #tmux new-session -s squlearn10q
#tmux attach-session -t squlearn10q
cd /home/leova3397/projects/squlearn
    ~/.conda/envs/myenv/bin/python scripts/TEST_capolupo.py \
  --data examples/tutorials/data/data_sqrt.h5ad \
  --use_spsa \
  --parallel_fd 8 \
  --n_qubits 10 \
  --num_layers 4 --max_layers 4 \
  --kernel_name projected \
  --iter 100 \
  --loss_subsample 100 \
  --opt_samples 2000 \
  --seed 42 \
  --output /data/gpfs/projects/punim0613/zuzana/qphate/results/kernel_bench_out

    #tmux new-session -s squlearn2PCs
#tmux attach-session -t qulearn2PCs
cd /home/leova3397/projects/squlearn
    ~/.conda/envs/myenv/bin/python scripts/TEST_capolupo.py \
  --data examples/tutorials/data/data_sqrt.h5ad \
  --use_spsa \
  --parallel_fd 8 \
  --n_qubits 4 \
  --num_layers 4 --max_layers 4 \
  --kernel_name projected \
  --iter 100 \
  --loss_subsample 100 \
  --opt_samples 2000 \
  --seed 42 \
  --output /data/gpfs/projects/punim0613/zuzana/qphate/results/kernel_bench_out

      #tmux new-session -s squlearn2PCs
#tmux attach-session -t qulearn2PCs
cd /home/leova3397/projects/squlearn
    ~/.conda/envs/myenv/bin/python scripts/TEST_capolupo.py \
  --data examples/tutorials/data/data_sqrt.h5ad \
  --use_spsa \
  --parallel_fd 8 \
  --n_qubits 10 \
  --num_layers 4 --max_layers 4 \
  --kernel_name projected \
  --iter 300 \
  --loss_subsample 100 \
  --opt_samples 2000 \
  --seed 42 \
  --output /data/gpfs/projects/punim0613/zuzana/qphate/results/kernel_bench_out



  #!/bin/bash
#SBATCH --job-name=squlearn_cobyla
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

#tmux new-session -s cobyla_test
#tmux attach-session -t cobyla_test
cd /home/leova3397/projects/squlearn
~/.conda/envs/myenv/bin/python scripts/TEST_Damrich.py \
  --data examples/tutorials/data/data_sqrt.h5ad \
  --optimizer_method cobyla \
  --n_qubits 10 \
  --num_layers 4 --max_layers 4 \
  --iterations 150 \
  --loss_subsample 100 \
  --opt_samples 2000 \
  --gamma_fixed 50 \
  --seed 42 \
  --no_timestamp \
  --output /data/gpfs/projects/punim0613/zuzana/qphate/results/kernel_bench_out

  ~/.conda/envs/myenv/bin/python scripts/TEST_Damrich.py \
  --data data_sqrt.h5ad \
  --n_qubits 5 \
  --num_layers 4 --max_layers 4 \
  --iterations 150 \
  --loss_subsample 100 \
  --opt_samples 2000 \
  --seed 42 \
  --no_timestamp \
  --output /data/gpfs/projects/punim0613/zuzana/qphate/results/kernel_bench_out


  ~/.conda/envs/myenv/bin/python scripts/TEST_Damrich.py \
  --data examples/tutorials/data/data_sqrt.h5ad \
  --n_qubits 10 \
  --num_layers 2 --max_layers 2 \
  --iterations 100 \
  --loss_subsample 100 \
  --opt_samples 2000 \
  --seed 42 \
  --no_timestamp \
  --output /data/gpfs/projects/punim0613/zuzana/qphate/results/kernel_bench_out

  tmux attach-session -t toy_sphere
  #tmux new-session -s toy_sphere
  ~/.conda/envs/myenv/bin/python scripts/TEST_Damrich.py \
  --data data_sqrt.h5ad \
  --dataset toy_sphere \
  --n_qubits 5 \
  --num_layers 4 --max_layers 4 \
  --iterations 150 \
  --loss_subsample 100 \
  --opt_samples 2000 \
  --seed 42 \
  --no_timestamp \
  --output /data/gpfs/projects/punim0613/zuzana/qphate/results/kernel_bench_out


  tmux attach-session -t cobyla_test
    ~/.conda/envs/myenv/bin/python scripts/TEST_Damrich.py \
  --data data_sqrt.h5ad \
  --dataset eyeglasses \
  --n_qubits 10 \
  --num_layers 2 --max_layers 2 \
  --iterations 150 \
  --loss_subsample 100 \
  --opt_samples 2000 \
  --seed 42 \
  --no_timestamp \
  --output /data/gpfs/projects/punim0613/zuzana/qphate/results/kernel_bench_out

  #tmux attach-session -t inter_circles
  ~/.conda/envs/myenv/bin/python scripts/TEST_Damrich.py \
  --data data_sqrt.h5ad \
  --dataset inter_circles \
  --n_qubits 10 \
  --num_layers 2 --max_layers 2 \
  --iterations 150 \
  --loss_subsample 100 \
  --opt_samples 2000 \
  --seed 42 \
  --no_timestamp \
  --output /data/gpfs/projects/punim0613/zuzana/qphate/results/kernel_bench_out


   #tmux attach-session -t toy_sphere
  ~/.conda/envs/myenv/bin/python scripts/TEST_Damrich.py \
  --data data_sqrt.h5ad \
  --dataset toy_sphere \
  --n_qubits 10 \
  --num_layers 2 --max_layers 2 \
  --iterations 150 \
  --loss_subsample 100 \
  --opt_samples 2000 \
  --seed 42 \
  --no_timestamp \
  --output /data/gpfs/projects/punim0613/zuzana/qphate/results/kernel_bench_out

  ##### Traning 6th April #####
  #tmux new-session -s toy_circle
  ~/.conda/envs/myenv/bin/python scripts/TEST_Damrich.py \
  --data data_sqrt.h5ad \
  --dataset toy_circle \
  --n_qubits 4 \
  --num_layers 2 --max_layers 2 \
  --iterations 90 \
  --loss_subsample 100 \
  --opt_samples 2000 \
  --seed 42 \
  --no_timestamp \
  --output /data/gpfs/projects/punim0613/zuzana/qphate/results/kernel_bench_out

  #tmux new-session -s eyeglasses
  ~/.conda/envs/myenv/bin/python scripts/TEST_Damrich.py \
  --data data_sqrt.h5ad \
  --dataset eyeglasses \
  --n_qubits 6 \
  --num_layers 3 --max_layers 3 \
  --iterations 90 \
  --loss_subsample 100 \
  --opt_samples 2000 \
  --seed 42 \
  --no_timestamp \
  --output /data/gpfs/projects/punim0613/zuzana/qphate/results/kernel_bench_out

  #tmux new-session -s inter_circles
  ~/.conda/envs/myenv/bin/python scripts/TEST_Damrich.py \
  --data data_sqrt.h5ad \
  --dataset inter_circles \
  --n_qubits 7 \
  --num_layers 3 --max_layers 3 \
  --iterations 90 \
  --loss_subsample 100 \
  --opt_samples 2000 \
  --seed 42 \
  --no_timestamp \
  --output /data/gpfs/projects/punim0613/zuzana/qphate/results/kernel_bench_out

  #tmux new-session -s toy_sphere
  ~/.conda/envs/myenv/bin/python scripts/TEST_Damrich.py \
  --data data_sqrt.h5ad \
  --dataset toy_sphere \
  --n_qubits 5 \
  --num_layers 3 --max_layers 3 \
  --iterations 90 \
  --loss_subsample 100 \
  --opt_samples 2000 \
  --seed 42 \
  --no_timestamp \
  --output /data/gpfs/projects/punim0613/zuzana/qphate/results/kernel_bench_out


  ##### Traning 7th April #####
  #tmux attach-session -t  toy_circle
  ~/.conda/envs/myenv/bin/python scripts/TEST_Damrich.py \
  --data data_sqrt.h5ad \
  --dataset toy_circle \
  --n_qubits 5 \
  --num_layers 3 --max_layers 3 \
  --iterations 90 \
  --loss_subsample 100 \
  --opt_samples 2000 \
  --seed 42 \
  --no_timestamp \
  --output /data/gpfs/projects/punim0613/zuzana/qphate/results/kernel_bench_out

  #tmux new-session -s eyeglasses
  ~/.conda/envs/myenv/bin/python scripts/TEST_Damrich.py \
  --data data_sqrt.h5ad \
  --dataset eyeglasses \
  --n_qubits 6 \
  --num_layers 3 --max_layers 3 \
  --iterations 90 \
  --loss_subsample 100 \
  --opt_samples 2000 \
  --seed 42 \
  --no_timestamp \
  --output /data/gpfs/projects/punim0613/zuzana/qphate/results/kernel_bench_out

  #tmux new-session -s inter_circles
  ~/.conda/envs/myenv/bin/python scripts/TEST_Damrich.py \
  --data data_sqrt.h5ad \
  --dataset inter_circles \
  --n_qubits 7 \
  --num_layers 3 --max_layers 3 \
  --iterations 90 \
  --loss_subsample 100 \
  --opt_samples 2000 \
  --seed 42 \
  --no_timestamp \
  --output /data/gpfs/projects/punim0613/zuzana/qphate/results/kernel_bench_out

  #tmux new-session -s toy_sphere
  ~/.conda/envs/myenv/bin/python scripts/TEST_Damrich.py \
  --data data_sqrt.h5ad \
  --dataset toy_sphere \
  --n_qubits 5 \
  --num_layers 3 --max_layers 3 \
  --iterations 90 \
  --loss_subsample 100 \
  --opt_samples 2000 \
  --seed 42 \
  --no_timestamp \
  --output /data/gpfs/projects/punim0613/zuzana/qphate/results/kernel_bench_out

  ##### Traning 8th April #####
  #tmux attach-session -t  toy_circle
  ~/.conda/envs/myenv/bin/python scripts/TEST_Damrich.py \
  --data data_sqrt.h5ad \
  --dataset toy_circle \
  --n_qubits 5 \
  --num_layers 3 --max_layers 3 \
  --iterations 90 \
  --loss_subsample 100 \
  --opt_samples 2000 \
  --seed 42 \
  --no_timestamp \
  --output /data/gpfs/projects/punim0613/zuzana/qphate/results/kernel_bench_out

  #tmux new-session -s eyeglasses
  ~/.conda/envs/myenv/bin/python scripts/TEST_Damrich.py \
  --data data_sqrt.h5ad \
  --dataset eyeglasses \
  --n_qubits 6 \
  --num_layers 3 --max_layers 3 \
  --iterations 90 \
  --loss_subsample 100 \
  --opt_samples 2000 \
  --seed 42 \
  --no_timestamp \
  --output /data/gpfs/projects/punim0613/zuzana/qphate/results/kernel_bench_out

  #tmux new-session -s inter_circles
  cd /home/leova3397/projects/squlearn
  ~/.conda/envs/myenv/bin/python scripts/TEST_Damrich.py \
  --data data_sqrt.h5ad \
  --dataset inter_circles \
  --n_qubits 8 \
  --num_layers 2 --max_layers 2 \
  --iterations 90 \
  --loss_subsample 100 \
  --opt_samples 2000 \
  --seed 42 \
  --no_timestamp \
  --output /data/gpfs/projects/punim0613/zuzana/qphate/results/kernel_bench_out

  #tmux new-session -s toy_sphere
  ~/.conda/envs/myenv/bin/python scripts/TEST_Damrich.py \
  --data data_sqrt.h5ad \
  --dataset toy_sphere \
  --n_qubits 5 \
  --num_layers 3 --max_layers 3 \
  --iterations 90 \
  --loss_subsample 100 \
  --opt_samples 2000 \
  --seed 42 \
  --no_timestamp \
  --output /data/gpfs/projects/punim0613/zuzana/qphate/results/kernel_bench_out

### 9th April 2026

tmux new-session -s inter_stress
tmux attach-session -t inter_stress
cd /home/leova3397/projects/squlearn
~/.conda/envs/myenv/bin/python scripts/run_inter_circles_stress_test.py \
  --data '/data/gpfs/projects/punim0613/zuzana/qphate/data/inter_circles_raw.npz' \
  --outdir /data/gpfs/projects/punim0613/zuzana/qphate/results/inter_circles_stress_single_job \
  --mode one-factor \
  --baseline-pcs 10 \
  --optimizer-method torch_adam \
  --opt-iterations 80 \
  --sample-size 250 \
  --baseline-qubits 8 \
  --baseline-layers 2 \
  --baseline-gamma 6 \
  --seed 42

tmux new-session -s eyeglasses
tmux attach-session -t eyeglasses
cd /home/leova3397/projects/squlearn
~/.conda/envs/myenv/bin/python scripts/run_inter_circles_stress_test.py \
  --data /data/gpfs/projects/punim0613/zuzana/qphate/data/eyeglasses_raw.npz \
  --outdir /data/gpfs/projects/punim0613/zuzana/qphate/results/stress_full_eyeglasses \
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
  --baseline-qubits 6 \
  --baseline-layers 3 \
  --optimizer-method torch_adam \
  --opt-iterations 100 \
  --adam-iter-multiplier 1 \
  --spsa-samples 1 \
  --diag-rbf-gamma config \
  --diag-approx-rank 200 \
  --torch-device cuda \
  --seed 42

tmux new-session -s toy_data_x
tmux attach-session -t toy_data_x
cd /home/leova3397/projects/squlearn

~/.conda/envs/myenv/bin/python scripts/run_inter_circles_stress_test.py \
  --data /data/gpfs/projects/punim0613/zuzana/qphate/data/toy_data_x.npz \
  --outdir /data/gpfs/projects/punim0613/zuzana/qphate/results/stress_full_toy_circle \
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
  --baseline-qubits 5 \
  --baseline-layers 3 \
  --optimizer-method torch_adam \
  --opt-iterations 100 \
  --adam-iter-multiplier 1 \
  --spsa-samples 1 \
  --diag-rbf-gamma config \
  --diag-approx-rank 200 \
  --torch-device cuda \
  --seed 42

tmux new-session -s toy_sphere
tmux attach-session -t toy_sphere
cd /home/leova3397/projects/squlearn

~/.conda/envs/myenv/bin/python scripts/run_inter_circles_stress_test.py \
  --data /data/gpfs/projects/punim0613/zuzana/qphate/data/toy_sphere_raw.npz \
  --outdir /data/gpfs/projects/punim0613/zuzana/qphate/results/stress_full_toy_sphere \
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
  --baseline-qubits 5 \
  --baseline-layers 3 \
  --optimizer-method torch_adam \
  --opt-iterations 100 \
  --adam-iter-multiplier 1 \
  --spsa-samples 1 \
  --diag-rbf-gamma config \
  --diag-approx-rank 200 \
  --torch-device cuda \
  --seed 42

  ##### 10th April
tmux new-session -s toy_sphere_pipeline
tmux attach-session -t toy_sphere_pipeline
cd /home/leova3397/projects/squlearn
bash run_full_pqc_pipeline.sh --dataset toy_sphere --torch-device cuda --seed 42 --opt-iterations 100 --adam-multiplier 1 --spsa-samples 1 --top-k 4

tmux new-session -s toy_data_pipeline
tmux attach-session -t toy_data_pipeline
cd /home/leova3397/projects/squlearn
bash run_full_pqc_pipeline.sh --dataset toy_data_x.npz --torch-device cuda --seed 42 --opt-iterations 100 --adam-multiplier 1 --spsa-samples 1 --top-k 4