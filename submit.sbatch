#!/bin/bash
#SBATCH --job-name=2.5Dmul
#SBATCH --nodes=4                 # upper bound: 4 nodes × 16 cores = 64 ranks max
#SBATCH --time=01:00:00
#SBATCH --mem-per-cpu=512
#SBATCH --export=ALL
# #SBATCH --reservation=comp422   # uncomment during reserved sessions

module purge
module use /projects/comp422/modulefiles
module load assignment3 gompi/2023b   # OpenMPI + GCC + OpenBLAS + MPE

EXEC=./multiply_2_5d          # release build (BLAS optional)
N=7500

# ---- table of valid replication factors -----------------------------------
#  c must satisfy  P = p²·c   with p integer  AND   N % p == 0
declare -A CVAL=(
  [4]=1   # p=2
  [9]=1   # p=3
  [16]=4  # p=2
  [25]=1  # p=5
  [36]=9  # p=2
  [49]=49 # p=1  (full replication; 7 would be illegal)
  [64]=16 # p=2
)

echo "=== 2.5D scalability (explicit c, adaptive nodes) ==="
for P in 4 9 16 25 36 49 64; do
  c=${CVAL[$P]}
  # how many nodes do we actually need for this run?
  nodes=$(( (P + 15) / 16 ))   # 16 MPI ranks per node on NOTS
  echo ">> P=$P  c=$c  (nodes=$nodes)"
  srun -N $nodes -n $P ${EXEC} ${N} ${c}
done
