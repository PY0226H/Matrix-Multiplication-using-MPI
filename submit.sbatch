#!/bin/bash
#SBATCH --job-name=2.5Dmul
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --mem-per-cpu=512
#SBATCH --export=ALL
#SBATCH --reservation=comp422

# Load the SLURM‑aware MPI toolchain
module purge
module use /projects/comp422/modulefiles
module load assignment3
module load gompi/2023b

# Executables
EXEC=./multiply_2_5d
TRACE_EXEC=./multiply_2_5d-trace

# List of total ranks to test, and matching replication factors c
P_LIST=(4 9 16 25 36 49 64)
declare -A CVALS=(
  [4]=1
  [9]=1
  [16]=4
  [25]=5
  [36]=9
  [49]=7
  [64]=16
)

echo "=== 2.5D Matrix Multiply Performance Runs ==="
for P in "${P_LIST[@]}"; do
  c=${CVALS[$P]}
  echo ">> Running with P=$P ranks, c=$c replicas"
  mpirun -np $P $EXEC 7500 $c
done

echo ""
echo "=== MPE Trace Collection (16 ranks, N=2000) ==="
mpirun -np 16 $TRACE_EXEC 2000 ${CVALS[16]}
